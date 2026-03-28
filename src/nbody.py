from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, lax

from .cosmo import Cosmology, add_cosmology_cotangents, zero_cosmology_param_cotangent
from .particles import Particles
from .steps import (
    force,
    integrate,
    force_adj,
    integrate_adj,
)


def nbody_init(a, ptcl, cosmo, conf):
    ptcl = force(a, ptcl, cosmo, conf)
    return ptcl


@jax.jit
def nbody_step(a_prev, a_next, ptcl, cosmo, conf):
    ptcl = integrate(a_prev, a_next, ptcl, cosmo, conf)
    return ptcl


def _nbody_scale_factors(conf, reverse):
    return conf.a_nbody[::-1] if reverse else conf.a_nbody


@partial(jax.jit, static_argnums=(3, 5, 6))
def nbody_collect(ptcl, cosmo, conf, collector, collector_state, reverse=False, return_final=False):
    """Run forward N-body integration while updating a caller-defined collector state.

    The default `nbody(...)` path stays unchanged. This helper is for forward-only uses
    such as saving projections, maps, or other diagnostics without baking those
    concerns into the adjoint solver itself.
    """
    a = _nbody_scale_factors(conf, reverse)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf)

    def body(carry, ab):
        ptcl_state, state = carry
        a_prev, a_next = ab
        ptcl_state = nbody_step(a_prev, a_next, ptcl_state, cosmo, conf)
        state = collector(state, a_prev, a_next, ptcl_state, cosmo, conf)
        return (ptcl_state, state), None

    (ptcl_final, collector_state), _ = lax.scan(body, (ptcl, collector_state), (a[:-1], a[1:]))
    if return_final:
        return ptcl_final, collector_state
    return collector_state


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def nbody_observe(ptcl, cosmo, conf, observer, reverse=False, include_start=False, return_final=False):
    """Run forward N-body integration and stack one observation per saved step.

    `observer` must be a pure JAX function with signature
    `(a, ptcl, cosmo, conf) -> observation_pytree`.
    """
    a = _nbody_scale_factors(conf, reverse)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf)
    first_obs = observer(a[0], ptcl, cosmo, conf) if include_start else None

    def body(ptcl_state, ab):
        a_prev, a_next = ab
        ptcl_state = nbody_step(a_prev, a_next, ptcl_state, cosmo, conf)
        obs = observer(a_next, ptcl_state, cosmo, conf)
        return ptcl_state, obs

    ptcl_final, observations = lax.scan(body, ptcl, (a[:-1], a[1:]))
    if include_start:
        observations = tree_map(
            lambda start, rest: jax.numpy.concatenate((start[jax.numpy.newaxis], rest), axis=0),
            first_obs,
            observations,
        )
    if return_final:
        return ptcl_final, observations
    return observations


def nbody_kappa(ptcl, cosmo, conf, reverse=False):
    """Compatibility wrapper for the legacy saved-map N-body path."""
    from .nbody_observers import nbody_kappa as _nbody_kappa

    return _nbody_kappa(ptcl, cosmo, conf, reverse=reverse)


def _nbody_impl(ptcl, cosmo, conf, reverse=False):
    """Plain N-body time integration body used by the custom VJP wrapper."""
    a = _nbody_scale_factors(conf, reverse)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf)

    def body(ptcl, ab):
        a_prev, a_next = ab
        ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf)
        return ptcl, None

    ptcl, _ = lax.scan(body, ptcl, (a[:-1], a[1:]))
    return ptcl


def _ptcl_state(ptcl):
    return (
        ptcl.pmid,
        ptcl.disp,
        ptcl.vel,
        ptcl.acc,
        ptcl.unused_index,
        ptcl.halo_mask,
        ptcl.attr,
    )


def _state_to_ptcl(conf, state):
    pmid, disp, vel, acc, unused_index, halo_mask, attr = state
    return Particles(
        conf,
        pmid,
        disp,
        vel=vel,
        acc=acc,
        unused_index=unused_index,
        halo_mask=halo_mask,
        attr=attr,
    )


def _cosmo_state(cosmo):
    return (
        cosmo.A_s_1e9,
        cosmo.n_s,
        cosmo.Omega_m,
        cosmo.Omega_b,
        cosmo.h,
        cosmo.Omega_k_,
        cosmo.w_0_,
        cosmo.w_a_,
        cosmo.transfer,
        cosmo.growth,
        cosmo.varlin,
    )


def _state_to_cosmo(conf, state):
    A_s_1e9, n_s, Omega_m, Omega_b, h, Omega_k_, w_0_, w_a_, transfer, growth, varlin = state
    return Cosmology(
        conf,
        A_s_1e9,
        n_s,
        Omega_m,
        Omega_b,
        h,
        Omega_k_=Omega_k_,
        w_0_=w_0_,
        w_a_=w_a_,
        transfer=transfer,
        growth=growth,
        varlin=varlin,
    )


def _nbody_state_impl(conf, reverse, pmid, disp, vel, acc, unused_index, halo_mask, attr, cosmo):
    ptcl_in = _state_to_ptcl(conf, (pmid, disp, vel, acc, unused_index, halo_mask, attr))
    ptcl_out = _nbody_impl(ptcl_in, cosmo, conf, reverse=reverse)
    return _ptcl_state(ptcl_out)


@partial(jax.jit, static_argnums=(0, 1))
def _nbody_flat_impl(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state):
    cosmo = _state_to_cosmo(conf, cosmo_state)
    return _nbody_state_impl(conf, reverse, pmid, disp, vel, acc, unused_index, halo_mask, attr, cosmo)


def nbody_adj(ptcl, ptcl_cot, cosmo, conf, reverse=False):
    """Run the hand-written adjoint sweep from the final nbody state."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody

    cosmo_cot = zero_cosmology_param_cotangent(cosmo)

    def body(carry, ab):
        ptcl, ptcl_cot, cosmo_cot = carry
        a_prev, a_next = ab
        carry = integrate_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf)
        return carry, None

    (ptcl, ptcl_cot, cosmo_cot), _ = lax.scan(
        body,
        (ptcl, ptcl_cot, cosmo_cot),
        (a_nbody[-2::-1], a_nbody[:0:-1]),
    )
    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a_nbody[0], ptcl, ptcl_cot, cosmo, conf)
    cosmo_cot = add_cosmology_cotangents(cosmo_cot, cosmo_cot_force)
    return ptcl, ptcl_cot, cosmo_cot


@partial(custom_vjp, nondiff_argnums=(0, 1))
def _nbody_state(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state):
    # Keep the public nbody entry point flat so the backward can start from the
    # final particle state without carrying a full-step replay tape.
    return _nbody_flat_impl(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state)


def nbody_adjoint_fwd(conf, reverse, pmid, unused_index, halo_mask, attr, disp, vel, acc, cosmo_state):
    cosmo = _state_to_cosmo(conf, cosmo_state)
    ptcl_in = _state_to_ptcl(conf, (pmid, disp, vel, acc, unused_index, halo_mask, attr))
    ptcl_out = _nbody_impl(ptcl_in, cosmo, conf, reverse=reverse)
    state_out = _ptcl_state(ptcl_out)
    input_optionals = (
        vel is None,
        acc is None,
        unused_index is None,
        halo_mask is None,
        attr is None,
    )
    return state_out, (state_out, cosmo_state, input_optionals)


def nbody_adjoint_bwd(conf, reverse, res, cotangents):
    state_out, cosmo_state, input_optionals = res
    vel_is_none, acc_is_none, _, _, _ = input_optionals

    ptcl_out = _state_to_ptcl(conf, state_out)
    cosmo = _state_to_cosmo(conf, cosmo_state)
    _, disp_cot, vel_cot, acc_cot, _, _, _ = cotangents
    ptcl_out_cot = ptcl_out.replace(disp=disp_cot, vel=vel_cot, acc=acc_cot)

    ptcl_in, ptcl_in_cot, cosmo_cot = nbody_adj(
        ptcl_out,
        ptcl_out_cot,
        cosmo,
        conf,
        reverse=reverse,
    )

    return (
        None,
        None,
        None,
        None,
        ptcl_in_cot.disp,
        None if vel_is_none else ptcl_in_cot.vel,
        None if acc_is_none else ptcl_in_cot.acc,
        _cosmo_state(cosmo_cot),
    )


_nbody_state.defvjp(nbody_adjoint_fwd, nbody_adjoint_bwd)


def nbody(ptcl, cosmo, conf, reverse=False):
    """N-body time integration with the hand-written adjoint backward."""
    cosmo_state = _cosmo_state(cosmo)
    state_out = _nbody_state(
        conf,
        reverse,
        ptcl.pmid,
        ptcl.unused_index,
        ptcl.halo_mask,
        ptcl.attr,
        ptcl.disp,
        ptcl.vel,
        ptcl.acc,
        cosmo_state,
    )
    return _state_to_ptcl(conf, state_out)
