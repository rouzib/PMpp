from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, lax
from jax.tree_util import tree_map

from .scatter import scatter
from .steps import force, integrate, force_adj, integrate_adj
from .utils import wraparound_slice


def nbody_init(a, ptcl, cosmo, conf):
    ptcl = force(a, ptcl, cosmo, conf)
    return ptcl


@jax.jit
def nbody_step(a_prev, a_next, ptcl, cosmo, conf):
    ptcl = integrate(a_prev, a_next, ptcl, cosmo, conf)
    return ptcl


# @partial(custom_vjp, nondiff_argnums=(3, ))
# def nbody(ptcl, cosmo, conf, reverse=False):
#     """N-body time integration."""
#     a = conf.a_nbody[::-1] if reverse else conf.a_nbody
#     ptcl = nbody_init(a[0], ptcl, cosmo, conf)
#
#     def body(ptcl, ab):
#         a_prev, a_next = ab
#         ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf)
#         return ptcl, None
#
#     ptcl, _ = lax.scan(body, ptcl, (a[:-1], a[1:]))
#     return ptcl

@partial(jax.custom_vjp, nondiff_argnums=(3,))
def nbody(ptcl, cosmo, conf, reverse=False):
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
    saved_maps = jnp.zeros((len(conf.to_save_a),) + (conf.nMesh, conf.nMesh), dtype=conf.float_dtype)
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
            dens = scatter(ptcl, conf)
            dens = wraparound_slice(dens, conf.slice_to_save[match_index], conf.slice_to_save[match_index + 1],
                                    max_slice_width, axis=0)
            dens = jnp.sum(dens, axis=0)
            temp = jnp.ones((128, 128, 128), dtype=jnp.int16)
            temp = wraparound_slice(temp, conf.slice_to_save[match_index], conf.slice_to_save[match_index + 1],
                                    max_slice_width, axis=0)
            jax.debug.print(
                "at a={x} (index)={index}, saving density: {y} , temp.sum(axis=0).shape={z}, positions={i}, {j}",
                x=to_save_a[match_index], index=match_index,
                y=temp.sum(axis=0)[0, 0], z=temp.sum(axis=0).shape, i=conf.slice_to_save[match_index],
                j=conf.slice_to_save[match_index + 1])
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


@jax.jit
def nbody_adj_init(a, ptcl, ptcl_cot, cosmo, conf):
    ptcl, ptcl_cot, cosmo_cot_force = force_adj(a, ptcl, ptcl_cot, cosmo, conf)
    cosmo_cot = tree_map(jnp.zeros_like, cosmo)
    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


@jax.jit
def nbody_adj_step(a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = integrate_adj(
        a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force


def nbody_adj(ptcl, ptcl_cot, cosmo, conf, reverse=False):
    """N-body time integration with adjoint equation."""
    a_nbody = conf.a_nbody[::-1] if reverse else conf.a_nbody

    print(a_nbody[-1], ptcl, ptcl_cot, cosmo, conf)
    ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_init(
        a_nbody[-1], ptcl, ptcl_cot, cosmo, conf)
    for a_prev, a_next in zip(a_nbody[:0:-1], a_nbody[-2::-1]):
        ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force = nbody_adj_step(
            a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf)
    return ptcl, ptcl_cot, cosmo_cot


def nbody_fwd(ptcl, cosmo, conf, reverse):
    ptcl = nbody(ptcl, cosmo, conf, reverse)
    return ptcl, (ptcl, cosmo, conf)


def nbody_bwd(reverse, res, cotangents):
    ptcl, cosmo, conf = res
    ptcl_cot = cotangents

    ptcl, ptcl_cot, cosmo_cot = nbody_adj(ptcl, ptcl_cot, cosmo, conf,
                                          reverse=reverse)
    return ptcl_cot, cosmo_cot, None


nbody.defvjp(nbody_fwd, nbody_bwd)
