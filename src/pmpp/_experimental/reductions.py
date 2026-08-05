"""Fused phase-space reduction candidates."""

from __future__ import annotations

import jax.numpy as jnp


def _count(mask, dtype):
    return jnp.maximum(jnp.sum(mask.astype(dtype)), jnp.asarray(1, dtype=dtype))


def fused_particle_statistics(displacement, velocity, mask):
    """Reduce displacement and velocity means/peak norms in one logical pass."""

    displacement = jnp.asarray(displacement)
    velocity = jnp.asarray(velocity, dtype=displacement.dtype)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if displacement.shape != velocity.shape or displacement.ndim < 2:
        raise ValueError("displacement and velocity must have the same (..., 3) shape")
    if mask.shape != displacement.shape[:-1]:
        raise ValueError("mask must match the vector leading shape")
    dtype = displacement.dtype
    count = _count(mask, dtype)
    vectors = jnp.stack([displacement, velocity], axis=0)
    weights = mask.astype(dtype)
    sums = jnp.sum(vectors * weights[None, ..., None], axis=tuple(range(1, vectors.ndim - 1)))
    means = sums / count
    peak_norm = jnp.sqrt(
        jnp.max(jnp.sum(vectors * vectors, axis=-1) * weights[None, ...], axis=tuple(range(1, mask.ndim + 1)), initial=jnp.asarray(0, dtype=dtype))
    )
    return count, means[0], means[1], peak_norm[0], peak_norm[1]


def fused_phase_space_reductions(bases, mask):
    """Compute one count and all basis sum-of-squares values together."""

    bases = jnp.asarray(bases)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if bases.ndim < 3 or bases.shape[-1] != 3 or bases.shape[1:-1] != mask.shape:
        raise ValueError("bases must have shape (B, ... , 3), with mask shape (...)")
    dtype = bases.dtype
    count = _count(mask, dtype)
    weights = mask.astype(dtype)
    sum_sq = jnp.sum(bases * bases * weights[None, ..., None], axis=tuple(range(1, bases.ndim - 1)))
    return count, sum_sq


__all__ = ["fused_particle_statistics", "fused_phase_space_reductions"]
