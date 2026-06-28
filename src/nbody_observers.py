"""Forward-only N-body observers and collectors.

This module keeps analysis / movie / map-export logic out of the core adjoint solver.
The default `src.nbody.nbody(...)` path stays particle-state focused, while callers that
need per-step products can opt into the observer helpers defined here.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from .nbody import nbody, nbody_collect
from .scatter import scatter
from .utils import wraparound_slice


def density_projection_observer(axis: int, normalize: bool = False):
    """Build an observer that returns a projected density image per step.

    Parameters
    ----------
    axis : int
        Spatial axis to sum over after scattering particles to the mesh.
    normalize : bool, optional
        Whether to divide each projection by its mean value.

    Returns
    -------
    callable
        Observer function with signature ``(a, ptcl, cosmo, conf) -> image``.
    """

    def observer(a, ptcl, cosmo, conf):
        del a, cosmo
        dens = scatter(ptcl, conf)
        proj = dens.sum(axis=axis)
        if normalize:
            proj = proj / jnp.mean(proj)
        return proj

    return observer


def nbody_kappa(ptcl, cosmo, conf, reverse: bool = False):
    """Legacy saved-map path implemented on top of the generic collector API.

    Parameters
    ----------
    ptcl : Particles
        Initial particle state.
    cosmo : Cosmology
        Cosmology used for the forward solve.
    conf : Configuration
        Active simulation configuration. ``to_save_a`` and related save fields
        control whether maps are recorded.
    reverse : bool, optional
        Whether to integrate in reverse scale-factor order.

    Returns
    -------
    Particles or jax.Array
        If ``conf.to_save_a`` is ``None``, returns the final particle state via
        :func:`src.nbody.nbody`. Otherwise returns the stacked saved maps.
    """
    if conf.to_save_a is None:
        return nbody(ptcl, cosmo, conf, reverse=reverse)

    saved_maps = jnp.zeros((len(conf.to_save_a), 3, conf.nMesh, conf.nMesh), dtype=conf.float_dtype)
    to_save_a = jnp.asarray(conf.to_save_a, dtype=conf.cosmo_dtype)
    max_slice_width = conf.max_slice_width

    def collector(saved_state, a_prev, a_next, ptcl_step, cosmo_step, conf_step):
        del a_prev, cosmo_step
        is_close = jnp.isclose(a_next, to_save_a, atol=1e-6)
        match_index = jnp.where(is_close, size=1, fill_value=-1)[0][0]

        def save_op(state):
            dens_tot = scatter(ptcl_step, conf_step)
            dens = jnp.stack(
                [
                    jnp.sum(
                        wraparound_slice(
                            dens_tot,
                            conf.slice_to_save[match_index],
                            conf.slice_to_save[match_index + 1],
                            max_slice_width,
                            axis=axis,
                        ),
                        axis=axis,
                    )
                    for axis in range(3)
                ],
                axis=0,
            )
            return state.at[match_index].set(dens)

        return lax.cond(match_index > -1, save_op, lambda state: state, saved_state)

    return nbody_collect(
        ptcl,
        cosmo,
        conf,
        collector,
        saved_maps,
        reverse=reverse,
        return_final=False,
    )
