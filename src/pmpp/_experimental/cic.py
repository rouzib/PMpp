"""Reference and experimental binned CIC operators.

The plan is an ephemeral index object.  It never changes PM++ canonical
``pmid`` ordering, and all output cotangents are returned in original particle
order.  The pure-JAX binned implementation is intentionally suitable for
logical-device tests; a CUDA worker can replace the final tile accumulation
while preserving this contract.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CICPlan(NamedTuple):
    positions: jax.Array
    base: jax.Array
    coordinates: jax.Array
    weights: jax.Array
    valid: jax.Array
    tile_key: jax.Array
    particle_order: jax.Array
    mesh_shape: tuple[int, int, int]
    tile_size: int


def make_cic_plan(
    positions,
    mesh_shape: tuple[int, int, int],
    *,
    valid_mask=None,
    tile_size: int = 4,
) -> CICPlan:
    """Create deterministic periodic eight-corner CIC indices."""

    if len(mesh_shape) != 3:
        raise ValueError("CIC plan requires three spatial dimensions")
    if tile_size not in (4, 8):
        raise ValueError("binned CIC tile_size must be 4 or 8")
    positions = jnp.asarray(positions)
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {positions.shape}")
    n = positions.shape[0]
    valid = jnp.ones((n,), dtype=jnp.bool_) if valid_mask is None else jnp.asarray(valid_mask, dtype=jnp.bool_)
    if valid.shape != (n,):
        raise ValueError(f"valid_mask must have shape {(n,)}, got {valid.shape}")
    base = jnp.floor(positions).astype(jnp.int32)
    fraction = positions - jnp.floor(positions)
    bits = jnp.asarray(
        [[(corner >> axis) & 1 for axis in range(3)] for corner in range(8)], dtype=jnp.int32
    )
    coordinates = jnp.mod(base[:, None, :] + bits[None, :, :], jnp.asarray(mesh_shape, jnp.int32))
    axis_weights = jnp.where(bits[None, :, :] == 1, fraction[:, None, :], 1.0 - fraction[:, None, :])
    weights = jnp.prod(axis_weights, axis=-1)
    tile_base = jnp.floor(base / tile_size).astype(jnp.int32)
    tile_shape = tuple((int(size) + tile_size - 1) // tile_size for size in mesh_shape)
    tile_key = (tile_base[:, 0] % tile_shape[0]) * tile_shape[1] * tile_shape[2]
    tile_key = tile_key + (tile_base[:, 1] % tile_shape[1]) * tile_shape[2] + tile_base[:, 2] % tile_shape[2]
    source = jnp.arange(n, dtype=jnp.int32)
    order = jnp.argsort(tile_key * max(n, 1) + source, stable=True)
    return CICPlan(
        positions,
        base,
        coordinates,
        weights,
        valid,
        tile_key,
        order,
        tuple(int(size) for size in mesh_shape),
        int(tile_size),
    )


def _scatter_from_indices(coordinates, weights, values, valid, mesh_shape):
    values = jnp.asarray(values)
    n = coordinates.shape[0]
    flat_coordinates = coordinates.reshape((n * 8, 3))
    flat_weights = (weights * valid.astype(weights.dtype)[:, None]).reshape((n * 8,))
    flat_values = jnp.repeat(values, 8, axis=0)
    if values.ndim == 1:
        updates = flat_values * flat_weights
        return jnp.zeros(mesh_shape, dtype=values.dtype).at[
            flat_coordinates[:, 0], flat_coordinates[:, 1], flat_coordinates[:, 2]
        ].add(updates)
    updates = flat_values * flat_weights[:, None]
    return jnp.zeros(mesh_shape + (values.shape[-1],), dtype=values.dtype).at[
        flat_coordinates[:, 0], flat_coordinates[:, 1], flat_coordinates[:, 2]
    ].add(updates)


def cic_scatter_reference(plan: CICPlan, values):
    """Reference atomic-style CIC scatter in canonical particle order."""

    return _scatter_from_indices(plan.coordinates, plan.weights, values, plan.valid, plan.mesh_shape)


def cic_scatter_binned(plan: CICPlan, values):
    """Ephemeral tile-binned scatter with original-order semantics.

    The sort and grouping are explicit here even on CPU.  CUDA implementations
    can consume ``particle_order`` and flush one shared tile at a time without
    changing the public function or its VJP.
    """

    order = plan.particle_order
    values = jnp.asarray(values)
    return _scatter_from_indices(
        plan.coordinates[order],
        plan.weights[order],
        values[order],
        plan.valid[order],
        plan.mesh_shape,
    )


def _gather_from_indices(mesh, coordinates, weights, valid):
    mesh = jnp.asarray(mesh)
    values = mesh[coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]]
    weighted = values * weights[(...,) + (None,) * (mesh.ndim - 3)]
    # The validity mask applies to the eight-corner axis, not to the particle
    # axis alone.  Keep an explicit corner singleton for scalar and channel
    # meshes alike.
    valid_factor = valid[:, None]
    if mesh.ndim > 3:
        valid_factor = valid_factor[..., None]
    weighted = weighted * valid_factor
    return weighted.sum(axis=1)


def cic_gather_reference(plan: CICPlan, mesh):
    """Reference three-dimensional periodic CIC gather."""

    return _gather_from_indices(mesh, plan.coordinates, plan.weights, plan.valid)


def cic_gather_binned(plan: CICPlan, mesh):
    """Gather using the same ephemeral tile order, then restore source order."""

    order = plan.particle_order
    gathered = _gather_from_indices(mesh, plan.coordinates[order], plan.weights[order], plan.valid[order])
    inverse = jnp.argsort(order, stable=True)
    return gathered[inverse]


def cic_scatter_vjp(plan: CICPlan, values, mesh_cotangent, *, binned: bool = False):
    """Return value and position cotangents for a scatter objective."""

    fn = cic_scatter_binned if binned else cic_scatter_reference
    _, pullback = jax.vjp(lambda v, p: fn(make_cic_plan(p, plan.mesh_shape, valid_mask=plan.valid, tile_size=plan.tile_size), v), values, plan.positions)
    value_cotangent, position_cotangent = pullback(mesh_cotangent)
    return value_cotangent, position_cotangent


def cic_gather_vjp(plan: CICPlan, mesh, output_cotangent, *, binned: bool = False):
    """Return mesh and position cotangents for a gather objective."""

    fn = cic_gather_binned if binned else cic_gather_reference
    _, pullback = jax.vjp(lambda m, p: fn(make_cic_plan(p, plan.mesh_shape, valid_mask=plan.valid, tile_size=plan.tile_size), m), mesh, plan.positions)
    mesh_cotangent, position_cotangent = pullback(output_cotangent)
    return mesh_cotangent, position_cotangent


__all__ = [
    "CICPlan",
    "cic_gather_binned",
    "cic_gather_reference",
    "cic_gather_vjp",
    "cic_scatter_binned",
    "cic_scatter_reference",
    "cic_scatter_vjp",
    "make_cic_plan",
]
