"""Differentiable N-body solver and its custom adjoint."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, lax, shard_map
from jax.sharding import PartitionSpec as P
from jax.tree_util import tree_map

from ..core.utils import AXIS_NAME
from ..cosmology.models import Cosmology, add_cosmology_cotangents, zero_cosmology_param_cotangent
from .particles import Particles
from ..corrections import (
    add_nbody_correction_cotangents, has_phase_space_correction, phase_space_is_invertible,
    zero_nbody_correction_cotangent,
)
from .integrator import (force, force_adj, integrate, integrate_adj, integrate_low_memory, )


def nbody_init(a, ptcl, cosmo, conf, correction=None):
    """Initialize the leapfrog state by computing the starting acceleration.

    Parameters
    ----------
    a : float
        Initial scale factor.
    ptcl : Particles
        Input particle state.
    cosmo : Cosmology
        Cosmology used for the gravity solve.
    conf : Configuration
        Active simulation configuration.
    correction : optional
        Potential-correction object applied in the force evaluation.

    Returns
    -------
    Particles
        Particle state with initialized acceleration.
    """
    ptcl = force(a, ptcl, cosmo, conf, correction=correction)
    return ptcl


@jax.jit
def nbody_step(a_prev, a_next, ptcl, cosmo, conf, correction=None):
    """Advance one N-body macro-step between adjacent scale factors.

    Parameters
    ----------
    a_prev
        Scale factor at the start of the integration interval.
    a_next
        Scale factor at the end of the integration interval.
    ptcl
        Particle state passed through the solver.
    cosmo
        Cosmology object supplying density, growth, and transfer parameters.
    conf
        Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
    correction
        Potential-correction pytree or ``None`` for the uncorrected PM force."""
    ptcl = integrate(a_prev, a_next, ptcl, cosmo, conf, correction=correction)
    return ptcl


def _nbody_scale_factors(conf, reverse):
    """Return the integration scale-factor schedule in forward or reverse order."""
    return conf.a_nbody[::-1] if reverse else conf.a_nbody


def _validate_reverse_correction(reverse, correction):
    """Reject reverse integration for a non-invertible direct phase map."""
    if reverse and has_phase_space_correction(correction) and not phase_space_is_invertible(correction):
        raise ValueError("reverse=True is not supported for a non-invertible phase-space correction.")


def _validate_low_memory_nbody(conf, reverse, correction):
    """Validate the deliberately narrow streamed-gravity solver contract."""
    if reverse:
        raise ValueError("low-memory N-body is forward-only; reverse must be False.")
    if correction is not None:
        raise ValueError("low-memory N-body currently requires correction=None.")
    if conf.dim != 3:
        raise ValueError(f"low-memory N-body requires three dimensions, got dim={conf.dim}.")
    if conf.compute_mesh is not None:
        if conf.multigpu_mode != "mesh_halo":
            raise ValueError("distributed low-memory N-body requires multigpu mode='mesh_halo'.")
        if conf.mGPU_irfftn_transposed is None:
            raise ValueError("distributed low-memory N-body requires the scalar distributed inverse FFT.")


def _max_authoritative_occupancy(ptcl, conf):
    """Return the largest authoritative particle count on any device."""
    if ptcl.unused_index is None and ptcl.halo_mask is None:
        return jnp.asarray(ptcl.disp.shape[0] // int(conf.num_devices or 1), dtype=jnp.int32)

    if ptcl.unused_index is None:
        unused_index = jnp.zeros_like(ptcl.halo_mask)
    else:
        unused_index = ptcl.unused_index
    halo_mask = jnp.zeros_like(unused_index) if ptcl.halo_mask is None else ptcl.halo_mask
    if conf.compute_mesh is None:
        return jnp.sum((~unused_index) & (~halo_mask), dtype=jnp.int32)

    @partial(shard_map, mesh=conf.compute_mesh, in_specs=(P(AXIS_NAME), P(AXIS_NAME)), out_specs=P(), check_vma=False, )
    def distributed_max(local_unused, local_halo):
        local_count = jnp.sum((~local_unused) & (~local_halo), dtype=jnp.int32)
        return lax.pmax(local_count, AXIS_NAME)

    return distributed_max(unused_index, halo_mask)


@partial(jax.jit, static_argnums=(3, 5, 6))
def nbody_collect(ptcl, cosmo, conf, collector, collector_state, reverse=False, return_final=False, correction=None):
    """Run forward N-body integration while updating caller-managed state.

    Parameters
    ----------
    ptcl : Particles
        Initial particle state.
    cosmo : Cosmology
        Cosmology used for force and time-step factors.
    conf : Configuration
        Active runtime configuration.
    collector : callable
        Pure JAX function with signature
        ``collector(state, a_prev, a_next, ptcl, cosmo, conf) -> new_state``.
    collector_state : PyTree
        Initial collector state carried through the integration.
    reverse : bool, optional
        Whether to traverse ``conf.a_nbody`` in reverse.
    return_final : bool, optional
        If True, also return the final particle state.
    correction : optional
        Potential-correction object passed through to force evaluation.

    Returns
    -------
    collector_state : PyTree
        Final collector state after all N-body steps.
    tuple[Particles, PyTree]
        Returned instead when ``return_final=True``.

    Notes
    -----
    This helper is forward-only. It is intended for diagnostics, saved maps,
    and other side-car computations that should stay outside the custom N-body
    adjoint.
    """
    _validate_reverse_correction(reverse, correction)
    a = _nbody_scale_factors(conf, reverse)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf, correction=correction)

    def body(carry, ab):
        """Advance one scan body step for the enclosing N-body integration."""
        ptcl_state, state = carry
        a_prev, a_next = ab
        ptcl_state = nbody_step(a_prev, a_next, ptcl_state, cosmo, conf, correction=correction)
        state = collector(state, a_prev, a_next, ptcl_state, cosmo, conf)
        return (ptcl_state, state), None

    (ptcl_final, collector_state), _ = lax.scan(body, (ptcl, collector_state), (a[:-1], a[1:]))
    if return_final:
        return ptcl_final, collector_state
    return collector_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def nbody_observe(ptcl, cosmo, conf, observer, reverse=False, include_start=False, return_final=False, correction=None):
    """Run forward N-body integration and stack one observation per step.

    Parameters
    ----------
    ptcl : Particles
        Initial particle state.
    cosmo : Cosmology
        Cosmology used for the integration.
    conf : Configuration
        Active runtime configuration.
    observer : callable
        Pure JAX function with signature
        ``observer(a, ptcl, cosmo, conf) -> observation_pytree``.
    reverse : bool, optional
        Whether to traverse ``conf.a_nbody`` in reverse.
    include_start : bool, optional
        Whether to prepend the observation at the initial scale factor.
    return_final : bool, optional
        If True, also return the final particle state.
    correction : optional
        Potential-correction object passed through to force evaluation.

    Returns
    -------
    observations : PyTree
        Observation tree stacked along a leading time axis.
    tuple[Particles, PyTree]
        Returned instead when ``return_final=True``.

    Notes
    -----
    This helper materializes one observation tree per saved step and is meant
    for forward diagnostics such as projections or summary statistics.
    """
    _validate_reverse_correction(reverse, correction)
    a = _nbody_scale_factors(conf, reverse)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf, correction=correction)
    first_obs = observer(a[0], ptcl, cosmo, conf) if include_start else None

    def body(ptcl_state, ab):
        """Advance one scan body step for the enclosing N-body integration."""
        a_prev, a_next = ab
        ptcl_state = nbody_step(a_prev, a_next, ptcl_state, cosmo, conf, correction=correction)
        obs = observer(a_next, ptcl_state, cosmo, conf)
        return ptcl_state, obs

    ptcl_final, observations = lax.scan(body, ptcl, (a[:-1], a[1:]))
    if include_start:
        observations = tree_map(
            lambda start, rest: jax.numpy.concatenate((start[jax.numpy.newaxis], rest), axis=0), first_obs,
            observations,
        )
    if return_final:
        return ptcl_final, observations
    return observations


def nbody_kappa(ptcl, cosmo, conf, reverse=False):
    """Compatibility wrapper for the legacy saved-map N-body path.

    Parameters
    ----------
    ptcl : Particles
        Initial particle state.
    cosmo : Cosmology
        Cosmology used for the forward solve.
    conf : Configuration
        Active simulation configuration.
    reverse : bool, optional
        Whether to integrate in reverse scale-factor order.

    Returns
    -------
    object
        Same return value as :func:`pmpp.nbody.nbody_kappa`.
    """
    from .observers import nbody_kappa as _nbody_kappa

    return _nbody_kappa(ptcl, cosmo, conf, reverse=reverse)


def _nbody_impl(ptcl, cosmo, conf, reverse=False, correction=None):
    """Plain N-body time integration body used by the custom VJP wrapper."""
    a = _nbody_scale_factors(conf, reverse)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf, correction=correction)

    def body(ptcl, ab):
        """Advance one scan body step for the enclosing N-body integration.

        Parameters
        ----------
        ptcl
            Particle state passed through the solver.
        ab
            Pair of adjacent scale factors for one scan iteration.
        """
        a_prev, a_next = ab
        ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf, correction=correction)
        return ptcl, None

    ptcl, _ = lax.scan(body, ptcl, (a[:-1], a[1:]))
    return ptcl


def _nbody_low_memory_impl(ptcl, cosmo, conf, reverse=False, correction=None):
    """Run the primal N-body scan with sequential force-component FFTs."""
    a = _nbody_scale_factors(conf, reverse)
    max_occupancy = _max_authoritative_occupancy(ptcl, conf)
    max_migration = jnp.int32(0)
    max_invalid_count = jnp.int32(0)
    ptcl = force(a[0], ptcl, cosmo, conf, correction=correction, streamed_gravity=True)

    def body(carry, ab):
        ptcl_state, high_water, migration_high_water, invalid_high_water = carry
        a_prev, a_next = ab
        ptcl_state, moved, invalid = integrate_low_memory(a_prev, a_next, ptcl_state, cosmo, conf)
        high_water = jnp.maximum(high_water, _max_authoritative_occupancy(ptcl_state, conf))
        migration_high_water = jnp.maximum(migration_high_water, moved)
        invalid_high_water = jnp.maximum(invalid_high_water, invalid)
        return (ptcl_state, high_water, migration_high_water, invalid_high_water), None

    (ptcl, max_occupancy, max_migration,
     max_invalid_count), _ = lax.scan(body, (ptcl, max_occupancy, max_migration, max_invalid_count), (a[:-1], a[1:]),
                                      )
    return ptcl, max_occupancy, max_migration, max_invalid_count


def _nbody_remat_impl(ptcl, cosmo, conf, reverse=False, correction=None):
    """Run an exact autodiff path, rematerializing every macro-step.

    Direct phase-space maps cannot use the reversible hand-written adjoint: the
    final state is insufficient to reconstruct the pre-map state.  This path
    makes every integration step a rematerialization boundary, so JAX replays
    its exact forward arithmetic during the VJP instead of applying the
    algebraic drift reconstruction.
    """
    a = _nbody_scale_factors(conf, reverse)

    def initialize(ptcl_state, cosmo_state, correction_state):
        return nbody_init(a[0], ptcl_state, cosmo_state, conf, correction=correction_state)

    ptcl = jax.checkpoint(initialize)(ptcl, cosmo, correction)

    def body(ptcl_state, ab):
        a_prev, a_next = ab

        def step(state, cosmo_state, correction_state, start, end):
            return integrate(start, end, state, cosmo_state, conf, correction=correction_state, )

        ptcl_state = jax.checkpoint(step)(ptcl_state, cosmo, correction, a_prev, a_next, )
        return ptcl_state, None

    ptcl, _ = lax.scan(body, ptcl, (a[:-1], a[1:]))
    return ptcl


def _ptcl_state(ptcl):
    """Flatten a ``Particles`` object into the differentiable custom-VJP state."""
    return (ptcl.pmid, ptcl.disp, ptcl.vel, ptcl.acc, ptcl.unused_index, ptcl.halo_mask, ptcl.attr, )


def _state_to_ptcl(conf, state):
    """Rebuild ``Particles`` from the flat custom-VJP particle state."""
    pmid, disp, vel, acc, unused_index, halo_mask, attr = state
    return Particles(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask, attr=attr, )


def _cosmo_state(cosmo):
    """Flatten ``Cosmology`` so the custom VJP can return parameter cotangents."""
    return (
        cosmo.A_s_1e9, cosmo.n_s, cosmo.Omega_m, cosmo.Omega_b, cosmo.h, cosmo.Omega_k_, cosmo.w_0_, cosmo.w_a_,
        cosmo.transfer, cosmo.growth, cosmo.varlin,
    )


def _state_to_cosmo(conf, state):
    """Rebuild ``Cosmology`` from the flat custom-VJP cosmology state."""
    A_s_1e9, n_s, Omega_m, Omega_b, h, Omega_k_, w_0_, w_a_, transfer, growth, varlin = state
    return Cosmology(
        conf, A_s_1e9, n_s, Omega_m, Omega_b, h, Omega_k_=Omega_k_, w_0_=w_0_, w_a_=w_a_, transfer=transfer,
        growth=growth, varlin=varlin,
    )


def _nbody_state_impl(conf, reverse, pmid, disp, vel, acc, unused_index, halo_mask, attr, cosmo, correction=None):
    """Run N-body on flat particle inputs and return a flat particle state."""
    ptcl_in = _state_to_ptcl(conf, (pmid, disp, vel, acc, unused_index, halo_mask, attr))
    ptcl_out = _nbody_impl(ptcl_in, cosmo, conf, reverse=reverse, correction=correction)
    return _ptcl_state(ptcl_out)


@partial(jax.jit, static_argnums=(0, 1))
def _nbody_flat_impl(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=None):
    """Jitted bridge from flat custom-VJP arguments to the solver body."""
    cosmo = _state_to_cosmo(conf, cosmo_state)
    return _nbody_state_impl(
        conf, reverse, pmid, disp, vel, acc, unused_index, halo_mask, attr, cosmo, correction=correction,
    )


@partial(jax.jit, static_argnums=(0, 1), donate_argnums=(6, 7, 8))
def _nbody_low_memory_flat_impl(
    conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=None,
):
    """Jitted flat bridge for the forward-only low-memory solver."""
    cosmo = _state_to_cosmo(conf, cosmo_state)
    ptcl_in = _state_to_ptcl(conf, (pmid, disp, vel, acc, unused_index, halo_mask, attr))
    ptcl_out, max_occupancy, max_migration, max_invalid_count = _nbody_low_memory_impl(
        ptcl_in, cosmo, conf, reverse=reverse, correction=correction,
    )
    return _ptcl_state(ptcl_out), max_occupancy, max_migration, max_invalid_count


def nbody_adj(ptcl, ptcl_cot, cosmo, conf, reverse=False, correction=None):
    """Sweep the hand-written N-body adjoint from the final particle state.

    Parameters
    ----------
    ptcl : Particles
        Final particle state from the forward N-body solve.
    ptcl_cot : Particles
        Cotangent with respect to that final state.
    cosmo : Cosmology
        Cosmology used in the forward solve.
    conf : Configuration
        Active runtime configuration.
    reverse : bool, optional
        Whether the paired forward solve integrated in reverse time order.
    correction : optional
        Potential-correction object used in the forward solve.

    Returns
    -------
    ptcl : Particles
        Reconstructed initial particle state.
    ptcl_cot : Particles
        Cotangent with respect to the reconstructed initial state.
    cosmo_cot : Cosmology
        Accumulated cosmology parameter cotangent.
    correction_cot : optional
        Cotangent for the correction object, if one is active.
    """
    if has_phase_space_correction(correction):
        raise ValueError(
            "The reversible N-body adjoint cannot be used with a direct phase-space correction; "
            "differentiate the public nbody() function to use exact rematerialized autodiff."
        )
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody

    cosmo_cot = zero_cosmology_param_cotangent(cosmo)
    correction_cot = zero_nbody_correction_cotangent(correction)

    def body(carry, ab):
        ptcl, ptcl_cot, cosmo_cot, correction_cot = carry
        a_prev, a_next = ab
        return integrate_adj(
            a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf, correction=correction,
            correction_cot=correction_cot,
        ), None

    reverse_steps = (a_nbody[:-1][::-1], a_nbody[1:][::-1])
    (ptcl, ptcl_cot, cosmo_cot,
     correction_cot), _ = lax.scan(body, (ptcl, ptcl_cot, cosmo_cot, correction_cot), reverse_steps,
                                   )

    # The forward initialization computes the acceleration at the first
    # scale factor before the first macro-step.  Pull that force through last.
    ptcl, ptcl_cot, cosmo_cot_force, correction_cot_force = force_adj(
        a_nbody[0], ptcl, ptcl_cot, cosmo, conf, correction=correction,
    )
    cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_force)
    correction_cot = add_nbody_correction_cotangents(correction_cot, correction_cot_force)
    return ptcl, ptcl_cot, cosmo_cot, correction_cot


@partial(custom_vjp, nondiff_argnums=(0, 1))
def _nbody_state(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=None):
    """Flat custom-VJP primitive underlying the public ``nbody`` call."""
    # Keep the public nbody entry point flat so the backward can start from the
    # final particle state without carrying a full-step replay tape.
    return _nbody_flat_impl(
        conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=correction,
    )


def nbody_adjoint_fwd(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=None):
    """Forward rule for the N-body custom VJP.

    Parameters
    ----------
    conf : Configuration
        Active simulation configuration.
    reverse : bool
        Whether the paired forward solve runs in reverse time order.
    pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state
        Flattened particle and cosmology state used by the custom VJP bridge.
    correction : optional
        Potential-correction object used in the forward solve.

    Returns
    -------
    tuple
        Primal output state plus residuals needed by the backward rule.

    Notes
    -----
    Only the final state and static option flags are saved. The backward rule
    reconstructs the adjoint trajectory by sweeping the symplectic steps in
    reverse, avoiding a full tape of every intermediate particle state.
    """
    cosmo = _state_to_cosmo(conf, cosmo_state)
    ptcl_in = _state_to_ptcl(conf, (pmid, disp, vel, acc, unused_index, halo_mask, attr))
    ptcl_out = _nbody_impl(ptcl_in, cosmo, conf, reverse=reverse, correction=correction)
    state_out = _ptcl_state(ptcl_out)
    input_optionals = (vel is None, acc is None, unused_index is None, halo_mask is None, attr is None, )
    return state_out, (state_out, cosmo_state, input_optionals, correction)


def nbody_adjoint_bwd(conf, reverse, res, cotangents):
    """Backward rule for the N-body custom VJP.

    Parameters
    ----------
    conf : Configuration
        Active simulation configuration.
    reverse : bool
        Whether the paired forward solve ran in reverse time order.
    res : tuple
        Residuals produced by :func:`nbody_adjoint_fwd`.
    cotangents : tuple
        Cotangents with respect to the flat custom-VJP output state.

    Returns
    -------
    tuple
        Cotangents with respect to the flat custom-VJP inputs.
    """
    state_out, cosmo_state, input_optionals, correction = res
    vel_is_none, acc_is_none, _, _, _ = input_optionals

    ptcl_out = _state_to_ptcl(conf, state_out)
    cosmo = _state_to_cosmo(conf, cosmo_state)
    _, disp_cot, vel_cot, acc_cot, _, _, _ = cotangents
    ptcl_out_cot = ptcl_out.replace(disp=disp_cot, vel=vel_cot, acc=acc_cot)

    ptcl_in, ptcl_in_cot, cosmo_cot, correction_cot = nbody_adj(
        ptcl_out, ptcl_out_cot, cosmo, conf, reverse=reverse, correction=correction,
    )

    return (
        None, None, None, None, ptcl_in_cot.disp, None if vel_is_none else ptcl_in_cot.vel,
        None if acc_is_none else ptcl_in_cot.acc, _cosmo_state(cosmo_cot), correction_cot,
    )


_nbody_state.defvjp(nbody_adjoint_fwd, nbody_adjoint_bwd)


@partial(custom_vjp, nondiff_argnums=(0, 1))
def _nbody_low_memory_state(
    conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=None,
):
    """Flat forward-only primitive for the streamed-gravity solver."""
    return _nbody_low_memory_flat_impl(
        conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=correction,
    )


def _nbody_low_memory_state_fwd(
    conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=None,
):
    """Forward rule that deliberately saves no reverse-mode residuals."""
    state_out = _nbody_low_memory_flat_impl(
        conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state, correction=correction,
    )
    return state_out, ()


def _nbody_low_memory_state_bwd(conf, reverse, _res, _cotangents):
    """Reject differentiation instead of silently building a large tape."""
    del conf, reverse
    raise NotImplementedError("nbody_low_memory is forward-only. Use nbody for differentiated simulations.")


_nbody_low_memory_state.defvjp(_nbody_low_memory_state_fwd, _nbody_low_memory_state_bwd)


def nbody_low_memory(ptcl, cosmo, conf, reverse=False, correction=None):
    """Advance particles with sequential component FFTs and no AD contract.

    The standard :func:`nbody` entrypoint remains the differentiable default.
    This opt-in path is intended for memory-constrained forward simulations and
    rejects reverse integration, force corrections, and distributed modes
    other than ``mesh_halo``. Its particle-state buffers are donated to the
    result and must not be reused after the call starts.
    """
    ptcl, _ = nbody_low_memory_with_occupancy(ptcl, cosmo, conf, reverse=reverse, correction=correction)
    return ptcl


def nbody_low_memory_with_occupancy(ptcl, cosmo, conf, reverse=False, correction=None):
    """Run low-memory N-body and return its authoritative-occupancy high-water."""
    ptcl, max_occupancy, _, _ = nbody_low_memory_with_telemetry(
        ptcl, cosmo, conf, reverse=reverse, correction=correction,
    )
    return ptcl, max_occupancy


def nbody_low_memory_with_telemetry(ptcl, cosmo, conf, reverse=False, correction=None):
    """Run low-memory N-body and return occupancy and routing high-waters."""
    _validate_low_memory_nbody(conf, reverse, correction)
    cosmo_state = _cosmo_state(cosmo)
    state_out, max_occupancy, max_migration, max_invalid_count = _nbody_low_memory_state(
        conf, reverse, ptcl.pmid, ptcl.unused_index, ptcl.halo_mask, ptcl.attr, ptcl.disp, ptcl.vel, ptcl.acc,
        cosmo_state, correction,
    )
    return _state_to_ptcl(conf, state_out), max_occupancy, max_migration, max_invalid_count


def lower_nbody_low_memory(ptcl, cosmo, conf):
    """Lower the forward-only solver for HLO and compiled-memory inspection.

    This helper performs no execution or compilation. Callers may invoke
    ``.compile()`` on the returned JAX ``Lowered`` object when they explicitly
    want executable or memory-analysis artifacts.
    """
    _validate_low_memory_nbody(conf, reverse=False, correction=None)
    return _nbody_low_memory_flat_impl.lower(
        conf, False, ptcl.pmid, ptcl.unused_index, ptcl.halo_mask, ptcl.attr, ptcl.disp, ptcl.vel, ptcl.acc,
        _cosmo_state(cosmo), None,
    )


def nbody(ptcl, cosmo, conf, reverse=False, correction=None):
    """Advance particles through the configured N-body schedule.

    Parameters
    ----------
    ptcl : Particles
        Input particle state, typically produced by LPT or a prior segment.
    cosmo : Cosmology
        Cosmology with precomputed transfer and growth tables.
    conf : Configuration
        Active runtime configuration.
    reverse : bool, optional
        Whether to integrate over ``conf.a_nbody`` in reverse order.
    correction : optional
        Potential-correction object applied inside each force evaluation.

    Returns
    -------
    Particles
        Final particle state after the N-body integration.

    Notes
    -----
    Phase-free runs use the custom VJP that reconstructs the adjoint sweep from
    the final state.  A direct phase-space correction instead uses exact native
    autodiff with one rematerialization boundary per macro-step, because its
    pre-map state cannot be reconstructed algebraically.
    """
    _validate_reverse_correction(reverse, correction)
    if has_phase_space_correction(correction):
        return _nbody_remat_impl(ptcl, cosmo, conf, reverse=reverse, correction=correction, )

    cosmo_state = _cosmo_state(cosmo)
    state_out = _nbody_state(
        conf, reverse, ptcl.pmid, ptcl.unused_index, ptcl.halo_mask, ptcl.attr, ptcl.disp, ptcl.vel, ptcl.acc,
        cosmo_state, correction,
    )
    return _state_to_ptcl(conf, state_out)


def nbody_static_halo_scheduled(ptcl, cosmo, confs, reverse=False, correction=None):
    """Run N-body through multiple configuration segments.

    Parameters
    ----------
    ptcl : Particles
        Input particle state for the first segment.
    cosmo : Cosmology
        Cosmology shared by all segments.
    confs : sequence of Configuration
        Segment configurations, each with its own time schedule and static-halo
        settings.
    reverse : bool, optional
        Whether to execute the segment list and each internal N-body schedule in
        reverse order.
    correction : optional
        Potential-correction object applied in every segment.

    Returns
    -------
    Particles
        Final particle state after all segments complete.

    Notes
    -----
    This experimental helper is mainly for the static-owner mesh-halo path.
    Differentiation still happens segment-by-segment through public ``nbody``;
    each segment selects the fast reversible adjoint or exact rematerialized
    phase-space path as appropriate.
    """
    _validate_reverse_correction(reverse, correction)
    ordered_confs = confs[::-1] if reverse else confs
    for segment_conf in ordered_confs:
        ptcl = nbody(ptcl, cosmo, segment_conf, reverse=reverse, correction=correction)
    return ptcl
