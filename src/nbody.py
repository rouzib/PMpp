from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, lax
from jax.tree_util import tree_map

from .steps import force, integrate, force_adj, integrate_adj


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
    saved_ptcls = jax.tree_util.tree_map(
        lambda x: jnp.zeros((len(conf.to_save_a),) + x.shape, dtype=x.dtype), ptcl
    )
    to_save_a = jnp.array(conf.to_save_a)

    def body(carry, ab):
        ptcl, saved_ptcls = carry
        a_prev, a_next = ab
        ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf)

        # Check if the current 'a_next' is close to any of the values in 'to_save_a'.
        # A small tolerance is used for the comparison.
        is_close = jnp.isclose(a_next, to_save_a, atol=1e-6)

        # If there is a match, get the index for saving.
        # jnp.where will return the indices where is_close is True. We take the first one.
        match_indices = jnp.where(is_close, size=1, fill_value=-1)[0]
        match_index = match_indices[0]

        def save_op(saved_ptcls):
            # If a match is found, update the corresponding slice of the saved_ptcls pytree.
            return jax.tree_util.tree_map(
                lambda s, p: s.at[match_index].set(p), saved_ptcls, ptcl
            )

        def no_op(saved_ptcls):
            # If no match, return the saved_ptcls pytree as is.
            return saved_ptcls

        # Conditionally apply the save operation.
        saved_ptcls = lax.cond(match_index > -1, save_op, no_op, saved_ptcls)

        return (ptcl, saved_ptcls), None

    # The initial carry for the scan includes the initial particle state and the empty
    # structure for the saved states.
    (ptcl, saved_ptcls), _ = lax.scan(body, (ptcl, saved_ptcls), (a[:-1], a[1:]))

    # The function now returns the final state of ptcl and the pytree with the saved states.
    return ptcl, saved_ptcls


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
