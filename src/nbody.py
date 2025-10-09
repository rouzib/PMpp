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


@partial(custom_vjp, nondiff_argnums=(3, ))
def nbody(ptcl, cosmo, conf, reverse=False):
    """N-body time integration."""
    a = conf.a_nbody[::-1] if reverse else conf.a_nbody
    ptcl = nbody_init(a[0], ptcl, cosmo, conf)

    def body(ptcl, ab):
        a_prev, a_next = ab
        ptcl = nbody_step(a_prev, a_next, ptcl, cosmo, conf)
        return ptcl, None

    ptcl, _ = lax.scan(body, ptcl, (a[:-1], a[1:]))
    return ptcl


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
