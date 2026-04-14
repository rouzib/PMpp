import jax
from jax import vjp, value_and_grad
import jax.numpy as jnp

from .configuration import Configuration
from .cosmo import (
    E2,
    H_deriv,
    add_cosmology_cotangents,
    cosmology_param_cotangent,
    cosmology_param_names,
    cosmology_param_values,
    project_cosmology_param_cotangent,
    scale_cosmology_cotangent,
    sub_cosmology_cotangents,
    replace_cosmology_params,
)
from .gravity import gravity, duplicate_slot_counts
from .growth import growth
from .particles import Particles
from .corrections import add_potential_correction_cotangents


def _halo_move_float_outputs(ptcl, disp, vel, acc, conf):
    _, disp, vel, acc, _, _, _, _ = conf.mGPU_halo_moving(
        ptcl.pmid,
        ptcl.disp,
        disp,
        vel,
        acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.unused_index,
    )
    return disp, vel, acc


def _halo_move_float_outputs_with_aux(ptcl, disp, vel, acc, conf):
    pmid, disp, vel, acc, halo_mask, unused_indexes, _, _ = conf.mGPU_halo_moving(
        ptcl.pmid,
        ptcl.disp,
        disp,
        vel,
        acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.unused_index,
    )
    return (disp, vel, acc), (pmid, halo_mask, unused_indexes)


def _halo_move_outputs_vjp_with_aux(ptcl, disp, vel, acc, conf):
    (float_outputs, halo_move_vjp, aux) = vjp(
        lambda disp_in, vel_in, acc_in: _halo_move_float_outputs_with_aux(
            ptcl, disp_in, vel_in, acc_in, conf
        ),
        disp,
        vel,
        acc,
        has_aux=True,
    )
    return float_outputs, aux, halo_move_vjp


def _halo_move_vjp(ptcl, disp, vel, acc, disp_cot, vel_cot, acc_cot, conf):
    _, halo_move_vjp = vjp(
        lambda disp_in, vel_in, acc_in: _halo_move_float_outputs(
            ptcl, disp_in, vel_in, acc_in, conf
        ),
        disp,
        vel,
        acc,
    )
    return halo_move_vjp((disp_cot, vel_cot, acc_cot))


def partition_duplicate_slot_cot(ptcl, ptcl_cot, conf):
    """Convert duplicated slot cotangents into a partitioned per-slot representation."""
    counts = jax.lax.stop_gradient(duplicate_slot_counts(ptcl, conf)).astype(ptcl_cot.disp.dtype)
    return ptcl_cot.replace(
        disp=jnp.where(counts != 0, ptcl_cot.disp / counts, 0),
        vel=jnp.where(counts != 0, ptcl_cot.vel / counts, 0),
        acc=jnp.where(counts != 0, ptcl_cot.acc / counts, 0),
    )


def duplicate_partitioned_slot_cot(ptcl, ptcl_cot, conf):
    """Expand a partitioned per-slot cotangent back to duplicated-slot convention."""
    counts = jax.lax.stop_gradient(duplicate_slot_counts(ptcl, conf)).astype(ptcl_cot.disp.dtype)
    return ptcl_cot.replace(
        disp=ptcl_cot.disp * counts,
        vel=ptcl_cot.vel * counts,
        acc=ptcl_cot.acc * counts,
    )


def _G_D(a, cosmo, conf):
    """Growth factor of ZA canonical velocity in [H_0]."""
    return a ** 2 * jnp.sqrt(E2(a, cosmo)) * growth(a, cosmo, conf, deriv=1)


def _G_K(a, cosmo, conf):
    """Growth factor of ZA accelerations in [H_0^2]."""
    return a ** 3 * E2(a, cosmo) * (
            growth(a, cosmo, conf, deriv=2)
            + (2 + H_deriv(a, cosmo)) * growth(a, cosmo, conf, deriv=1)
    )


def drift_factor(a_vel, a_prev, a_next, cosmo, conf):
    """Drift time step factor of conf.float_dtype in [1/H_0]."""
    factor = growth(a_next, cosmo, conf) - growth(a_prev, cosmo, conf)
    factor /= _G_D(a_vel, cosmo, conf)
    return factor


def kick_factor(a_acc, a_prev, a_next, cosmo, conf):
    """Kick time step factor of conf.float_dtype in [1/H_0]."""
    factor = _G_D(a_next, cosmo, conf) - _G_D(a_prev, cosmo, conf)
    factor /= _G_K(a_acc, cosmo, conf)
    return factor


def _drift_factor_param_grad(a_vel, a_prev, a_next, cosmo, conf):
    param_names = cosmology_param_names(cosmo)
    param_values = cosmology_param_values(cosmo, param_names)
    factor, param_cot = value_and_grad(
        lambda params: drift_factor(
            a_vel,
            a_prev,
            a_next,
            replace_cosmology_params(cosmo, param_names, params),
            conf,
        )
    )(param_values)
    return factor, cosmology_param_cotangent(cosmo, param_names, param_cot)


def _kick_factor_param_grad(a_acc, a_prev, a_next, cosmo, conf):
    param_names = cosmology_param_names(cosmo)
    param_values = cosmology_param_values(cosmo, param_names)
    factor, param_cot = value_and_grad(
        lambda params: kick_factor(
            a_acc,
            a_prev,
            a_next,
            replace_cosmology_params(cosmo, param_names, params),
            conf,
        )
    )(param_values)
    return factor, cosmology_param_cotangent(cosmo, param_names, param_cot)


def drift(a_vel, a_prev, a_next, ptcl: Particles, cosmo, conf: Configuration):
    """Drift."""
    factor = drift_factor(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    disp = ptcl.disp + ptcl.vel * factor

    if not conf.use_mGPU or conf.mGPU_halo_moving is None:
        return ptcl.replace(disp=disp)

    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl.pmid,
        ptcl.disp,
        disp,
        ptcl.vel,
        ptcl.acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.unused_index,
    )
    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=acc, halo_mask=halo_mask, unused_index=unused_indexes)


def drift_adj(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Drift stage adjoint from the pre-drift particle state."""
    factor, cosmo_cot_drift = _drift_factor_param_grad(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    disp_before_halo = ptcl.disp + ptcl.vel * factor
    vel_before_halo = ptcl.vel
    acc_before_halo = ptcl.acc

    if not conf.use_mGPU or conf.mGPU_halo_moving is None:
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
        _halo_move_outputs_vjp_with_aux(
            ptcl,
            disp_before_halo,
            vel_before_halo,
            acc_before_halo,
            conf,
        )
    )
    ptcl_out = ptcl.replace(
        pmid=pmid,
        disp=disp_out,
        vel=vel_out,
        acc=acc_out,
        halo_mask=halo_mask,
        unused_index=unused_indexes,
    )

    disp, vel, acc = halo_move_vjp((ptcl_cot.disp, ptcl_cot.vel, ptcl_cot.acc))
    vel_cot = vel - disp * factor
    ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel_cot, acc=acc)

    cosmo_cot_drift = scale_cosmology_cotangent(cosmo_cot_drift, (disp * vel_before_halo).sum())
    cosmo_cot = sub_cosmology_cotangents(cosmo_cot, cosmo_cot_drift)

    return ptcl_out, ptcl_cot, cosmo_cot


def drift_adj_from_output(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Drift stage adjoint from the post-drift particle state."""
    factor, cosmo_cot_drift = _drift_factor_param_grad(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    if not conf.use_mGPU or conf.mGPU_reconstruct_pre_drift is None:
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
        (
            pmid,
            disp_input,
            vel_before_halo,
            acc_before_halo,
            unused_indexes,
            halo_mask_input,
            disp,
            vel,
            acc,
        ) = fused_pullback(
            ptcl.pmid,
            ptcl.disp,
            ptcl.vel,
            ptcl.acc,
            ptcl.unused_index,
            factor,
            ptcl_cot.disp,
            ptcl_cot.vel,
            ptcl_cot.acc,
        )
        ptcl = ptcl.replace(
            pmid=pmid,
            disp=disp_input,
            vel=vel_before_halo,
            acc=acc_before_halo,
            halo_mask=halo_mask_input,
            unused_index=unused_indexes,
        )
    else:
        # The adjoint scan only has the post-drift state, so rebuild the exact
        # canonical pre-drift layout before pulling the cotangent through halo move.
        pmid, disp_input, vel_before_halo, acc_before_halo, unused_indexes, halo_mask_input = conf.mGPU_reconstruct_pre_drift(
            ptcl.pmid,
            ptcl.disp,
            ptcl.vel,
            ptcl.acc,
            conf.halo_start,
            conf.halo_end,
            ptcl.unused_index,
            factor,
        )
        ptcl = ptcl.replace(
            pmid=pmid,
            disp=disp_input,
            vel=vel_before_halo,
            acc=acc_before_halo,
            halo_mask=halo_mask_input,
            unused_index=unused_indexes,
        )
        disp_before_halo = disp_input + vel_before_halo * factor

        disp, vel, acc = conf.mGPU_halo_move_pullback(
            ptcl.pmid,
            ptcl.disp,
            disp_before_halo,
            vel_before_halo,
            acc_before_halo,
            conf.halo_end,
            ptcl.unused_index,
            ptcl_cot.disp,
            ptcl_cot.vel,
            ptcl_cot.acc,
        )
    vel_cot = vel + disp * factor
    ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel_cot, acc=acc)

    cosmo_cot_drift = scale_cosmology_cotangent(cosmo_cot_drift, (disp * vel_before_halo).sum())
    cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_drift)

    return ptcl, ptcl_cot, cosmo_cot


def kick(a_acc, a_prev, a_next, ptcl, cosmo, conf):
    """Kick."""
    factor = kick_factor(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    vel = ptcl.vel + ptcl.acc * factor

    return ptcl.replace(vel=vel)


def kick_adj(a_acc, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Kick, and particle and cosmology adjoints."""
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


def force(a, ptcl, cosmo, conf, correction=None):
    """Force."""
    acc = gravity(a, ptcl, cosmo, conf, correction=correction)
    return ptcl.replace(acc=acc)


def force_adj(a, ptcl, ptcl_cot, cosmo, conf, correction=None):
    """Force, and particle and cosmology vjp."""
    if correction is None:
        acc, gravity_vjp = vjp(gravity, a, ptcl, cosmo, conf)
        correction_cot_force = None
    else:
        acc, gravity_vjp = vjp(
            lambda ptcl_in, cosmo_in, correction_in: gravity(
                a, ptcl_in, cosmo_in, conf, correction=correction_in
            ),
            ptcl,
            cosmo,
            correction,
        )

    ptcl = ptcl.replace(acc=acc)

    # The force output only differs from the input in the acceleration field.
    # Pull the acceleration cotangent through gravity and pass the untouched
    # displacement / velocity cotangents straight through.
    acc_out_cot = ptcl_cot.acc
    if correction is None:
        _, ptcl_cot_force, cosmo_cot_force, _ = gravity_vjp(acc_out_cot)
    else:
        ptcl_cot_force, cosmo_cot_force, correction_cot_force = gravity_vjp(acc_out_cot)
    disp_cot_force = ptcl_cot_force.disp
    disp_cot = ptcl_cot.disp + disp_cot_force
    vel_cot = ptcl_cot.vel
    acc_cot = jnp.zeros_like(ptcl.acc)
    ptcl_cot = ptcl_cot.replace(disp=disp_cot, vel=vel_cot, acc=acc_cot)
    cosmo_cot_force = project_cosmology_param_cotangent(cosmo_cot_force)

    return ptcl, ptcl_cot, cosmo_cot_force, correction_cot_force


def integrate(a_prev, a_next, ptcl, cosmo, conf, correction=None):
    """Symplectic integration for one step."""
    D = K = 0
    a_disp = a_vel = a_acc = a_prev
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ptcl = drift(a_vel, a_disp, a_disp_next, ptcl, cosmo, conf)
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


def integrate_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf, correction=None, correction_cot=None):
    """Symplectic integration adjoint for one step."""
    correction_cot = add_potential_correction_cotangents(None, correction_cot)

    for stage in reversed(_integrate_stage_schedule(a_prev, a_next, conf)):
        if stage[0] == "kick":
            _, a_acc_stage, a_vel_stage, a_vel_next = stage
            ptcl, ptcl_cot, cosmo_cot = kick_adj(
                a_acc_stage,
                a_vel_stage,
                a_vel_next,
                ptcl,
                ptcl_cot,
                cosmo,
                cosmo_cot,
                conf,
            )
            continue

        _, a_vel_stage, a_disp_stage, a_disp_next, a_acc_in = stage
        ptcl, ptcl_cot, cosmo_cot_force_stage, correction_cot_force_stage = force_adj(
            a_disp_next,
            ptcl,
            ptcl_cot,
            cosmo,
            conf,
            correction=correction,
        )
        cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_force_stage)
        correction_cot = add_potential_correction_cotangents(correction_cot, correction_cot_force_stage)
        ptcl, ptcl_cot, cosmo_cot = drift_adj_from_output(
            a_vel_stage,
            a_disp_stage,
            a_disp_next,
            ptcl,
            ptcl_cot,
            cosmo,
            cosmo_cot,
            conf,
        )

        # The force stage overwrites acceleration, so restore the incoming
        # acceleration field before any earlier kick adjoint consumes it.
        ptcl = force(a_acc_in, ptcl, cosmo, conf, correction=correction)

    return ptcl, ptcl_cot, cosmo_cot, correction_cot
