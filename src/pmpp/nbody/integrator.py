import jax
from jax import vjp, value_and_grad
import jax.numpy as jnp

from ..core.configuration import Configuration
from ..cosmology.models import (
    E2, H_deriv, add_cosmology_cotangents, cosmology_param_cotangent, cosmology_param_names, cosmology_param_values,
    project_cosmology_param_cotangent, scale_cosmology_cotangent, sub_cosmology_cotangents, replace_cosmology_params,
    zero_cosmology_param_cotangent,
)
from .gravity import gravity, duplicate_slot_counts
from ..cosmology.growth import growth
from .particles import Particles
from ..corrections import (
    add_nbody_correction_cotangents, apply_local_pair_correction, apply_phase_space_correction,
    has_phase_space_correction, local_pair_correction, long_range_correction, phase_space_correction,
    prepare_phase_space_context,
)
from ..core.utils import raise_error


def _assert_halo_move_succeeded(has_failed, max_ptcl_moved):
    """Fail closed if a particle mover reports truncated routing."""
    _ = jax.lax.cond(
        jnp.any(has_failed), lambda _: raise_error(
            "[ERROR] Particle migration reported a static-capacity failure. "
            "max_particles_moved={x}.", x=jnp.max(max_ptcl_moved),
        ), lambda _: None, operand=None,
    )


def _halo_move_float_outputs(ptcl, disp, vel, acc, conf):
    """Run halo movement and expose only floating outputs to JAX VJP."""
    _, disp, vel, acc, _, _, _, _ = conf.mGPU_halo_moving(
        ptcl.pmid, ptcl.disp, disp, vel, acc, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    return disp, vel, acc


def _halo_move_float_outputs_with_aux(ptcl, disp, vel, acc, conf):
    """Run halo movement with integer/bool routing metadata returned as aux."""
    pmid, disp, vel, acc, halo_mask, unused_indexes, _, _ = conf.mGPU_halo_moving(
        ptcl.pmid, ptcl.disp, disp, vel, acc, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    return (disp, vel, acc), (pmid, halo_mask, unused_indexes)


def _halo_move_outputs_vjp_with_aux(ptcl, disp, vel, acc, conf):
    """Create a VJP for halo movement while preserving the new particle layout."""
    (float_outputs, halo_move_vjp, aux) = vjp(
        lambda disp_in, vel_in, acc_in: _halo_move_float_outputs_with_aux(ptcl, disp_in, vel_in, acc_in, conf), disp,
        vel, acc, has_aux=True,
    )
    return float_outputs, aux, halo_move_vjp


def _halo_move_vjp(ptcl, disp, vel, acc, disp_cot, vel_cot, acc_cot, conf):
    """Pull cotangents through the halo-movement operation."""
    _, halo_move_vjp = vjp(
        lambda disp_in, vel_in, acc_in: _halo_move_float_outputs(ptcl, disp_in, vel_in, acc_in, conf), disp, vel, acc,
    )
    return halo_move_vjp((disp_cot, vel_cot, acc_cot))


def partition_duplicate_slot_cot(ptcl, ptcl_cot, conf):
    """Convert duplicated slot cotangents into a partitioned per-slot representation.

    Parameters
    ----------
    ptcl : Particles
        Particle state whose slots may include duplicates.
    ptcl_cot : Particles
        Cotangent in duplicated-slot convention.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    Particles
        Cotangent reweighted into a partitioned per-slot convention.
    """
    counts = jax.lax.stop_gradient(duplicate_slot_counts(ptcl, conf)).astype(ptcl_cot.disp.dtype)
    return ptcl_cot.replace(
        disp=jnp.where(counts != 0, ptcl_cot.disp / counts, 0), vel=jnp.where(counts != 0, ptcl_cot.vel / counts, 0),
        acc=jnp.where(counts != 0, ptcl_cot.acc / counts, 0),
    )


def duplicate_partitioned_slot_cot(ptcl, ptcl_cot, conf):
    """Expand a partitioned per-slot cotangent back to duplicated-slot convention.

    Parameters
    ----------
    ptcl : Particles
        Particle state whose slots may include duplicates.
    ptcl_cot : Particles
        Cotangent in partitioned-slot convention.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    Particles
        Cotangent expanded to the duplicated-slot convention.
    """
    counts = jax.lax.stop_gradient(duplicate_slot_counts(ptcl, conf)).astype(ptcl_cot.disp.dtype)
    return ptcl_cot.replace(disp=ptcl_cot.disp * counts, vel=ptcl_cot.vel * counts, acc=ptcl_cot.acc * counts, )


def _G_D(a, cosmo, conf):
    """Growth factor of ZA canonical velocity in [H_0]."""
    return a**2 * jnp.sqrt(E2(a, cosmo)) * growth(a, cosmo, conf, deriv=1)


def _G_K(a, cosmo, conf):
    """Growth factor of ZA accelerations in [H_0^2]."""
    return a**3 * E2(a, cosmo
                     ) * (growth(a, cosmo, conf, deriv=2) + (2 + H_deriv(a, cosmo)) * growth(a, cosmo, conf, deriv=1))


def drift_factor(a_vel, a_prev, a_next, cosmo, conf):
    """Return the drift time-step factor in ``[1 / H_0]``.

    Parameters
    ----------
    a_vel : float
        Scale factor at which the velocity is defined.
    a_prev, a_next : float
        Start and end scale factors of the drift substep.
    cosmo : Cosmology
        Cosmology providing the growth functions.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Drift factor in solver float precision.
    """
    factor = growth(a_next, cosmo, conf) - growth(a_prev, cosmo, conf)
    factor /= _G_D(a_vel, cosmo, conf)
    return factor


def kick_factor(a_acc, a_prev, a_next, cosmo, conf):
    """Return the kick time-step factor in ``[1 / H_0]``.

    Parameters
    ----------
    a_acc : float
        Scale factor at which the acceleration is defined.
    a_prev, a_next : float
        Start and end scale factors of the kick substep.
    cosmo : Cosmology
        Cosmology providing the growth functions.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Kick factor in solver float precision.
    """
    factor = _G_D(a_next, cosmo, conf) - _G_D(a_prev, cosmo, conf)
    factor /= _G_K(a_acc, cosmo, conf)
    return factor


def _drift_factor_param_grad(a_vel, a_prev, a_next, cosmo, conf):
    """Return a drift factor and its cotangent with respect to cosmology params."""
    if not conf.nbody_cosmo_grad:
        return drift_factor(a_vel, a_prev, a_next, cosmo, conf), zero_cosmology_param_cotangent(cosmo)

    param_names = cosmology_param_names(cosmo)
    param_values = cosmology_param_values(cosmo, param_names)
    factor, param_cot = value_and_grad(
        lambda params: drift_factor(
            a_vel, a_prev, a_next, replace_cosmology_params(cosmo, param_names, params), conf,
        )
    )(param_values)
    return factor, cosmology_param_cotangent(cosmo, param_names, param_cot)


def _kick_factor_param_grad(a_acc, a_prev, a_next, cosmo, conf):
    """Return a kick factor and its cotangent with respect to cosmology params."""
    if not conf.nbody_cosmo_grad:
        return kick_factor(a_acc, a_prev, a_next, cosmo, conf), zero_cosmology_param_cotangent(cosmo)

    param_names = cosmology_param_names(cosmo)
    param_values = cosmology_param_values(cosmo, param_names)
    factor, param_cot = value_and_grad(
        lambda params: kick_factor(a_acc, a_prev, a_next, replace_cosmology_params(cosmo, param_names, params), conf,
                                   )
    )(param_values)
    return factor, cosmology_param_cotangent(cosmo, param_names, param_cot)


def drift(a_vel, a_prev, a_next, ptcl: Particles, cosmo, conf: Configuration):
    """Apply one drift substep and update ownership if particles crossed slabs.

    Parameters
    ----------
    a_vel : float
        Velocity scale factor for the drift.
    a_prev, a_next : float
        Start and end scale factors of the drift substep.
    ptcl : Particles
        Input particle state.
    cosmo : Cosmology
        Cosmology providing the drift factor.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    Particles
        Post-drift particle state.
    """
    factor = drift_factor(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    disp = ptcl.disp + ptcl.vel * factor

    if conf.use_mGPU and (conf.replicated_mesh or conf.static_mesh_halo_width > 0):
        return ptcl.replace(disp=disp)

    if not conf.use_mGPU or conf.mGPU_halo_moving is None:
        return ptcl.replace(disp=disp)

    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl.pmid, ptcl.disp, disp, ptcl.vel, ptcl.acc, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    _assert_halo_move_succeeded(has_failed, max_ptcl_moved)
    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=acc, halo_mask=halo_mask, unused_index=unused_indexes)


def drift_for_force(
    a_vel, a_prev, a_next, ptcl: Particles, cosmo, conf: Configuration, correction=None, apply_phase=False,
):
    """Apply a drift whose output is immediately consumed by a force stage.

    Parameters
    ----------
    a_vel : float
        Velocity scale factor for the drift.
    a_prev, a_next : float
        Start and end scale factors of the drift substep.
    ptcl : Particles
        Input particle state.
    cosmo : Cosmology
        Cosmology providing the drift factor.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    Particles
        Post-drift particle state specialized for an immediately following
        force stage.

    Notes
    -----
    On the mesh-halo fast path this can skip preserving acceleration data when
    the next operation will overwrite ``ptcl.acc`` anyway.
    """
    phase = phase_space_correction(correction) if apply_phase else None
    phase_context = None
    if phase is not None:
        # Mesh-halo scatter/gather requires particles to reside on their owning
        # slab.  Cache any mesh-local phase features while the pre-drift state
        # still satisfies that invariant; the residual itself is applied only
        # after the raw drift below.
        phase_context = prepare_phase_space_context(
            phase, a_next, ptcl, cosmo, conf, local_pair=local_pair_correction(correction),
        )

    factor = drift_factor(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    disp = ptcl.disp + ptcl.vel * factor

    if phase is not None:
        drifted = ptcl.replace(disp=disp)
        drifted = apply_phase_space_correction(
            phase, a_next, drifted, cosmo, conf, factor, local_pair=local_pair_correction(correction),
            context=phase_context,
        )
        disp = drifted.disp
        vel = drifted.vel
        acc = drifted.acc

        # The direct map is deliberately evaluated before this ownership
        # operation.  Passing the pre-drift displacement and final corrected
        # displacement to the mover handles phase-induced slab crossings once.
        if conf.use_mGPU and (conf.replicated_mesh or conf.static_mesh_halo_width > 0):
            return drifted
        if not conf.use_mGPU or conf.mGPU_halo_moving is None:
            return drifted

        no_acc_move = getattr(conf, "mGPU_halo_moving_no_acc", None)
        if no_acc_move is not None and conf.multigpu_mode == "mesh_halo":
            pmid, disp, vel, halo_mask, unused_indexes, has_failed, max_ptcl_moved = no_acc_move(
                ptcl.pmid, ptcl.disp, disp, vel, conf.halo_start, conf.halo_end, ptcl.unused_index,
            )
            _assert_halo_move_succeeded(has_failed, max_ptcl_moved)
            return drifted.replace(
                pmid=pmid, disp=disp, vel=vel, acc=jnp.zeros_like(vel), halo_mask=halo_mask,
                unused_index=unused_indexes,
            )

        pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
            ptcl.pmid, ptcl.disp, disp, vel, acc, conf.halo_start, conf.halo_end, ptcl.unused_index,
        )
        _assert_halo_move_succeeded(has_failed, max_ptcl_moved)
        return drifted.replace(
            pmid=pmid, disp=disp, vel=vel, acc=acc, halo_mask=halo_mask, unused_index=unused_indexes,
        )

    if conf.use_mGPU and (conf.replicated_mesh or conf.static_mesh_halo_width > 0):
        return ptcl.replace(disp=disp)

    no_acc_move = getattr(conf, "mGPU_halo_moving_no_acc", None)
    if not conf.use_mGPU or no_acc_move is None or conf.multigpu_mode != "mesh_halo":
        return drift(a_vel, a_prev, a_next, ptcl, cosmo, conf)

    pmid, disp, vel, halo_mask, unused_indexes, has_failed, max_ptcl_moved = no_acc_move(
        ptcl.pmid, ptcl.disp, disp, ptcl.vel, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    _assert_halo_move_succeeded(has_failed, max_ptcl_moved)
    acc = jnp.zeros_like(vel)
    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=acc, halo_mask=halo_mask, unused_index=unused_indexes)


def drift_adj(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Reverse a drift stage when the pre-drift state is available.

    Parameters
    ----------
    a_vel : float
        Velocity scale factor for the drift.
    a_prev, a_next : float
        Start and end scale factors of the drift substep.
    ptcl : Particles
        Pre-drift particle state.
    ptcl_cot : Particles
        Cotangent with respect to the post-drift state.
    cosmo : Cosmology
        Cosmology used in the forward drift.
    cosmo_cot : Cosmology
        Incoming cosmology cotangent accumulator.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    tuple
        ``(ptcl_out, ptcl_cot_out, cosmo_cot_out)`` after reversing the drift.
    """
    factor, cosmo_cot_drift = _drift_factor_param_grad(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    disp_before_halo = ptcl.disp + ptcl.vel * factor
    vel_before_halo = ptcl.vel
    acc_before_halo = ptcl.acc

    if ((not conf.use_mGPU) or conf.replicated_mesh or conf.static_mesh_halo_width > 0
        or conf.mGPU_halo_moving is None):
        ptcl_out = ptcl.replace(disp=disp_before_halo)
        disp = ptcl_cot.disp
        vel = ptcl_cot.vel
        acc = ptcl_cot.acc
        vel_cot = vel - disp * factor
        ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel_cot, acc=acc)
        cosmo_cot_drift = scale_cosmology_cotangent(cosmo_cot_drift, (disp * vel_before_halo).sum())
        cosmo_cot = sub_cosmology_cotangents(cosmo_cot, cosmo_cot_drift)
        return ptcl_out, ptcl_cot, cosmo_cot

    (disp_out, vel_out, acc_out), (pmid, halo_mask, unused_indexes), halo_move_vjp = (
        _halo_move_outputs_vjp_with_aux(ptcl, disp_before_halo, vel_before_halo, acc_before_halo, conf,
                                        )
    )
    ptcl_out = ptcl.replace(
        pmid=pmid, disp=disp_out, vel=vel_out, acc=acc_out, halo_mask=halo_mask, unused_index=unused_indexes,
    )

    disp, vel, acc = halo_move_vjp((ptcl_cot.disp, ptcl_cot.vel, ptcl_cot.acc))
    vel_cot = vel - disp * factor
    ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel_cot, acc=acc)

    cosmo_cot_drift = scale_cosmology_cotangent(cosmo_cot_drift, (disp * vel_before_halo).sum())
    cosmo_cot = sub_cosmology_cotangents(cosmo_cot, cosmo_cot_drift)

    return ptcl_out, ptcl_cot, cosmo_cot


def drift_adj_from_output(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Pull cotangents through a drift stage given only the post-drift state.

    Parameters
    ----------
    a_vel : float
        Velocity scale factor for the drift.
    a_prev, a_next : float
        Start and end scale factors of the drift substep.
    ptcl : Particles
        Post-drift particle state.
    ptcl_cot : Particles
        Cotangent with respect to the post-drift state.
    cosmo : Cosmology
        Cosmology used in the forward drift.
    cosmo_cot : Cosmology
        Incoming cosmology cotangent accumulator.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    tuple
        ``(ptcl_in, ptcl_cot_out, cosmo_cot_out)`` after reversing the drift.

    Notes
    -----
    For migrating multi-GPU runs this reconstructs the canonical pre-drift
    particle layout before applying the halo-movement pullback.
    """
    factor, cosmo_cot_drift = _drift_factor_param_grad(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    if ((not conf.use_mGPU) or conf.replicated_mesh or conf.static_mesh_halo_width > 0
        or conf.mGPU_reconstruct_pre_drift is None):
        vel_before_halo = ptcl.vel
        acc_before_halo = ptcl.acc
        disp_input = ptcl.disp - vel_before_halo * factor
        ptcl = ptcl.replace(disp=disp_input, vel=vel_before_halo, acc=acc_before_halo)
        disp = ptcl_cot.disp
        vel = ptcl_cot.vel
        acc = ptcl_cot.acc
        vel_cot = vel + disp * factor
        ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel_cot, acc=acc)
        cosmo_cot_drift = scale_cosmology_cotangent(cosmo_cot_drift, (disp * vel_before_halo).sum())
        cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_drift)
        return ptcl, ptcl_cot, cosmo_cot

    fused_pullback = getattr(conf, "mGPU_reconstruct_pre_drift_pullback", None)
    if fused_pullback is not None:
        (pmid, disp_input, vel_before_halo, acc_before_halo, unused_indexes, halo_mask_input, disp, vel, acc,
         ) = fused_pullback(
             ptcl.pmid, ptcl.disp, ptcl.vel, ptcl.acc, ptcl.unused_index, factor, ptcl_cot.disp, ptcl_cot.vel,
             ptcl_cot.acc,
         )
        ptcl = ptcl.replace(
            pmid=pmid, disp=disp_input, vel=vel_before_halo, acc=acc_before_halo, halo_mask=halo_mask_input,
            unused_index=unused_indexes,
        )
    else:
        # The adjoint scan only has the post-drift state, so rebuild the exact
        # canonical pre-drift layout before pulling the cotangent through halo move.
        pmid, disp_input, vel_before_halo, acc_before_halo, unused_indexes, halo_mask_input = conf.mGPU_reconstruct_pre_drift(
            ptcl.pmid, ptcl.disp, ptcl.vel, ptcl.acc, conf.halo_start, conf.halo_end, ptcl.unused_index, factor,
        )
        ptcl = ptcl.replace(
            pmid=pmid, disp=disp_input, vel=vel_before_halo, acc=acc_before_halo, halo_mask=halo_mask_input,
            unused_index=unused_indexes,
        )
        disp_before_halo = disp_input + vel_before_halo * factor

        disp, vel, acc = conf.mGPU_halo_move_pullback(
            ptcl.pmid, ptcl.disp, disp_before_halo, vel_before_halo, acc_before_halo, conf.halo_end, ptcl.unused_index,
            ptcl_cot.disp, ptcl_cot.vel, ptcl_cot.acc,
        )
    vel_cot = vel + disp * factor
    ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel_cot, acc=acc)

    cosmo_cot_drift = scale_cosmology_cotangent(cosmo_cot_drift, (disp * vel_before_halo).sum())
    cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_drift)

    return ptcl, ptcl_cot, cosmo_cot


def kick(a_acc, a_prev, a_next, ptcl, cosmo, conf):
    """Apply one kick substep to particle velocities.

    Parameters
    ----------
    a_acc : float
        Acceleration scale factor for the kick.
    a_prev, a_next : float
        Start and end scale factors of the kick substep.
    ptcl : Particles
        Input particle state.
    cosmo : Cosmology
        Cosmology providing the kick factor.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    Particles
        Post-kick particle state.
    """
    factor = kick_factor(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    vel = ptcl.vel + ptcl.acc * factor

    return ptcl.replace(vel=vel)


def kick_adj(a_acc, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Reverse a kick stage and accumulate particle/cosmology cotangents.

    Parameters
    ----------
    a_acc : float
        Acceleration scale factor for the kick.
    a_prev, a_next : float
        Start and end scale factors of the kick substep.
    ptcl : Particles
        Post-kick particle state.
    ptcl_cot : Particles
        Cotangent with respect to the post-kick state.
    cosmo : Cosmology
        Cosmology used in the forward kick.
    cosmo_cot : Cosmology
        Incoming cosmology cotangent accumulator.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    tuple
        ``(ptcl_in, ptcl_cot_out, cosmo_cot_out)`` after reversing the kick.
    """
    factor, cosmo_cot_kick = _kick_factor_param_grad(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    # kick
    vel = ptcl.vel - ptcl.acc * factor
    ptcl = ptcl.replace(vel=vel)

    # Kick only updates velocity, so the velocity cotangent feeds both the
    # input velocity and the input acceleration.
    vel_out_cot = ptcl_cot.vel
    acc_cot = ptcl_cot.acc + vel_out_cot * factor
    ptcl_cot = ptcl_cot.replace(vel=vel_out_cot, acc=acc_cot)

    cosmo_cot_kick = scale_cosmology_cotangent(cosmo_cot_kick, (vel_out_cot * ptcl.acc).sum())
    cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_kick)

    return ptcl, ptcl_cot, cosmo_cot


def force_acceleration(a, ptcl, cosmo, conf, correction=None):
    """Evaluate long- and short-range acceleration branches."""
    acc = gravity(a, ptcl, cosmo, conf, correction=long_range_correction(correction), )
    local = local_pair_correction(correction)
    if local is not None:
        acc = acc + apply_local_pair_correction(local, a, ptcl, cosmo, conf)
    return acc


def force(a, ptcl, cosmo, conf, correction=None):
    """Overwrite ``ptcl.acc`` with gravitational acceleration at scale factor ``a``.

    Parameters
    ----------
    a : float
        Scale factor of the force evaluation.
    ptcl : Particles
        Input particle state.
    cosmo : Cosmology
        Cosmology used for the gravity solve.
    conf : Configuration
        Active simulation configuration.
    correction : optional
        Potential-correction object applied in the gravity solve.

    Returns
    -------
    Particles
        Particle state with refreshed acceleration.
    """
    acc = force_acceleration(a, ptcl, cosmo, conf, correction=correction)
    return ptcl.replace(acc=acc)


def force_adj(a, ptcl, ptcl_cot, cosmo, conf, correction=None):
    """Differentiate one force evaluation with respect to particles and cosmology.

    Parameters
    ----------
    a : float
        Scale factor of the force evaluation.
    ptcl : Particles
        Post-force particle state.
    ptcl_cot : Particles
        Cotangent with respect to the post-force state.
    cosmo : Cosmology
        Cosmology used in the forward force evaluation.
    conf : Configuration
        Active simulation configuration.
    correction : optional
        Potential-correction object used in the forward gravity solve.

    Returns
    -------
    tuple
        ``(ptcl_in, ptcl_cot_out, cosmo_cot, correction_cot)``.
    """
    if correction is None:
        if conf.nbody_cosmo_grad:
            acc, gravity_vjp = vjp(force_acceleration, a, ptcl, cosmo, conf)
        else:
            acc, gravity_vjp = vjp(lambda ptcl_in: force_acceleration(a, ptcl_in, cosmo, conf), ptcl)
        correction_cot_force = None
    else:
        if conf.nbody_cosmo_grad:
            acc, gravity_vjp = vjp(
                lambda ptcl_in, cosmo_in, correction_in:
                force_acceleration(a, ptcl_in, cosmo_in, conf, correction=correction_in), ptcl, cosmo, correction,
            )
        else:
            acc, gravity_vjp = vjp(
                lambda ptcl_in, correction_in: force_acceleration(a, ptcl_in, cosmo, conf, correction=correction_in),
                ptcl, correction,
            )

    ptcl = ptcl.replace(acc=acc)

    # The force output only differs from the input in the acceleration field.
    # Pull the acceleration cotangent through gravity and pass the untouched
    # displacement / velocity cotangents straight through.
    acc_out_cot = ptcl_cot.acc
    if correction is None:
        if conf.nbody_cosmo_grad:
            _, ptcl_cot_force, cosmo_cot_force, _ = gravity_vjp(acc_out_cot)
        else:
            (ptcl_cot_force, ) = gravity_vjp(acc_out_cot)
            cosmo_cot_force = zero_cosmology_param_cotangent(cosmo)
    else:
        if conf.nbody_cosmo_grad:
            ptcl_cot_force, cosmo_cot_force, correction_cot_force = gravity_vjp(acc_out_cot)
        else:
            ptcl_cot_force, correction_cot_force = gravity_vjp(acc_out_cot)
            cosmo_cot_force = zero_cosmology_param_cotangent(cosmo)
    disp_cot_force = ptcl_cot_force.disp
    disp_cot = ptcl_cot.disp + disp_cot_force
    vel_cot = ptcl_cot.vel
    acc_cot = jnp.zeros_like(ptcl.acc)
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot, acc=acc_cot)
    cosmo_cot_force = project_cosmology_param_cotangent(cosmo_cot_force)

    return ptcl, ptcl_cot, cosmo_cot_force, correction_cot_force


def integrate(a_prev, a_next, ptcl, cosmo, conf, correction=None):
    """Advance one macro-step with the configured symplectic splitting.

    Parameters
    ----------
    a_prev, a_next : float
        Start and end scale factors of the macro-step.
    ptcl : Particles
        Input particle state.
    cosmo : Cosmology
        Cosmology used by the substeps.
    conf : Configuration
        Active simulation configuration.
    correction : optional
        Potential-correction object applied in force stages.

    Returns
    -------
    Particles
        Post-step particle state.
    """
    D = K = 0
    a_disp = a_vel = a_acc = a_prev
    last_drift_index = max(i for i, (d, _k) in enumerate(conf.symp_splits) if d != 0)
    for split_index, (d, k) in enumerate(conf.symp_splits):
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ptcl = drift_for_force(
                a_vel, a_disp, a_disp_next, ptcl, cosmo, conf, correction=correction,
                apply_phase=split_index == last_drift_index,
            )
            a_disp = a_disp_next
            ptcl = force(a_disp, ptcl, cosmo, conf, correction=correction)
            a_acc = a_disp

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            ptcl = kick(a_acc, a_vel, a_vel_next, ptcl, cosmo, conf)
            a_vel = a_vel_next

    return ptcl


def _integrate_stage_schedule(a_prev, a_next, conf):
    """Record the forward substep times so the adjoint can traverse them exactly."""
    D = K = 0
    a_disp = a_vel = a_acc = a_prev
    stages = []
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            stages.append(("drift", a_vel, a_disp, a_disp_next, a_acc))
            a_disp = a_disp_next
            a_acc = a_disp

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            stages.append(("kick", a_acc, a_vel, a_vel_next))
            a_vel = a_vel_next
    return stages


def integrate_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf, correction=None, correction_cot=None, ):
    """Reverse one macro-step of the symplectic integrator.

    Parameters
    ----------
    a_prev, a_next : float
        Start and end scale factors of the macro-step.
    ptcl : Particles
        Post-step particle state.
    ptcl_cot : Particles
        Cotangent with respect to the post-step state.
    cosmo : Cosmology
        Cosmology used in the forward step.
    cosmo_cot : Cosmology
        Incoming cosmology cotangent accumulator.
    conf : Configuration
        Active simulation configuration.
    correction : optional
        Potential-correction object used in the forward step.
    correction_cot : optional
        Incoming correction cotangent accumulator.

    Returns
    -------
    tuple
        ``(ptcl_in, ptcl_cot_out, cosmo_cot_out, correction_cot_out)``.
    """
    if has_phase_space_correction(correction):
        raise ValueError(
            "integrate_adj cannot reconstruct a direct phase-space map; "
            "differentiate nbody() to use exact rematerialized autodiff."
        )
    correction_cot = add_nbody_correction_cotangents(None, correction_cot)

    for stage in reversed(_integrate_stage_schedule(a_prev, a_next, conf)):
        if stage[0] == "kick":
            _, a_acc_stage, a_vel_stage, a_vel_next = stage
            ptcl, ptcl_cot, cosmo_cot = kick_adj(
                a_acc_stage, a_vel_stage, a_vel_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf,
            )
            continue

        _, a_vel_stage, a_disp_stage, a_disp_next, a_acc_in = stage
        ptcl, ptcl_cot, cosmo_cot_force_stage, correction_cot_force_stage = force_adj(
            a_disp_next, ptcl, ptcl_cot, cosmo, conf, correction=correction,
        )
        cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_force_stage)
        correction_cot = add_nbody_correction_cotangents(correction_cot, correction_cot_force_stage)
        ptcl, ptcl_cot, cosmo_cot = drift_adj_from_output(
            a_vel_stage, a_disp_stage, a_disp_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf,
        )

        # The force stage overwrites acceleration, so restore the incoming
        # acceleration field before any earlier kick adjoint consumes it.
        ptcl = force(a_acc_in, ptcl, cosmo, conf, correction=correction)

    return ptcl, ptcl_cot, cosmo_cot, correction_cot
