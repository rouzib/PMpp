from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, lax
from jax.tree_util import tree_map

from .cosmo import Cosmology
from .particles import Particles
from .scatter import scatter
from .steps import (
    force,
    integrate,
    force_adj,
    integrate_adj,
)
from .utils import wraparound_slice


def nbody_init(a, ptcl, cosmo, conf):
    ptcl = force(a, ptcl, cosmo, conf)
    return ptcl


@jax.jit
def nbody_step(a_prev, a_next, ptcl, cosmo, conf):
    ptcl = integrate(a_prev, a_next, ptcl, cosmo, conf)
    return ptcl


def nbody_kappa(ptcl, cosmo, conf, reverse=False):
    """N-body time integration with saving of intermediate states."""
    a = jnp.array(conf.a_nbody[::-1] if reverse else conf.a_nbody)
    ptcl = nbody_init(a[0], ptcl, cosmo, conf)

    if conf.to_save_a is None:
        def body(ptcl, ab):
            a_prev, a_next = ab
            ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf)
            return ptcl, None

        ptcl, _ = lax.scan(body, ptcl, (a[:-1], a[1:]))
        return ptcl

    # Create a pytree with the same structure as ptcl to store the saved states.
    # The leading dimension's size is the number of timesteps to save.
    saved_maps = jnp.zeros((len(conf.to_save_a), 3) + (conf.nMesh, conf.nMesh), dtype=conf.float_dtype)
    to_save_a = jnp.array(conf.to_save_a)

    max_slice_width = conf.max_slice_width

    def body(carry, ab):
        ptcl, saved_state = carry
        a_prev, a_next = ab
        ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf)

        # Check if the current 'a_next' is close to any of the values in 'to_save_a'.
        # A small tolerance is used for the comparison.
        is_close = jnp.isclose(a_next, to_save_a, atol=1e-6)

        # If there is a match, get the index for saving.
        # jnp.where will return the indices where is_close is True. We take the first one.
        match_indices = jnp.where(is_close, size=1, fill_value=-1)[0]
        match_index = match_indices[0]

        def save_op(saved_state):
            # If a match is found, update the corresponding slice of the saved_maps pytree.
            dens_tot = scatter(ptcl, conf)
            dens = jnp.stack([jnp.sum(
                wraparound_slice(dens_tot, conf.slice_to_save[match_index], conf.slice_to_save[match_index + 1],
                                 max_slice_width, axis=axis), axis=axis) for axis in range(3)], axis=0)
            return jax.tree_util.tree_map(
                lambda s, p: s.at[match_index].set(p), saved_state, dens
            )

        def no_op(saved_state):
            # If no match, return the saved_maps pytree as is.
            return saved_state

        # Conditionally apply the save operation.
        saved_maps = lax.cond(match_index > -1, save_op, no_op, saved_state)

        return (ptcl, saved_maps), None

    # The initial carry for the scan includes the initial particle state and the empty
    # structure for the saved states.
    (_, saved_maps), _ = lax.scan(body, (ptcl, saved_maps), (a[:-1], a[1:]))

    # The function now returns the saved states.
    return saved_maps


def _nbody_impl(ptcl, cosmo, conf, reverse=False):
    """Plain N-body time integration body used by the custom VJP wrapper."""
    a = conf.a_nbody[::-1] if reverse else conf.a_nbody
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


def _zeros_like_or_none(x):
    return jnp.zeros_like(x) if x is not None else None


def nbody_adj(ptcl, ptcl_cot, cosmo, conf, reverse=False):
    """Run the hand-written adjoint sweep from the final nbody state."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody

    cosmo_cot = tree_map(_zeros_like_or_none, cosmo)

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
    cosmo_cot += cosmo_cot_force
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
