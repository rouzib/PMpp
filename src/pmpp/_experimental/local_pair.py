"""Exact radial-shell and ratio-two local-pair reference candidates."""

from __future__ import annotations

from functools import lru_cache
import math

import jax.numpy as jnp
import numpy as np

from .cic import CICPlan, _gather_from_indices, _scatter_from_indices, make_cic_plan


@lru_cache(maxsize=None)
def shell_layout(cutoff_cells: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return offset shell IDs, shell counts, and valid offset rows."""

    cutoff = float(cutoff_cells)
    if not np.isfinite(cutoff) or cutoff < 1.0:
        raise ValueError("cutoff_cells must be finite and at least one mesh cell")
    radius = int(math.floor(cutoff))
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    xx, yy, zz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    squared = xx * xx + yy * yy + zz * zz
    valid = squared <= cutoff * cutoff + 1e-7
    shells = np.unique(squared[valid])
    shell_index = np.full(squared.shape, -1, dtype=np.int32)
    counts = np.zeros((shells.size,), dtype=np.float32)
    rows = []
    for shell_id, radius2 in enumerate(shells):
        mask = valid & (squared == radius2)
        shell_index[mask] = shell_id
        counts[shell_id] = float(np.count_nonzero(mask))
        rows.extend(np.stack(np.where(mask), axis=1).tolist())
    rows = np.asarray(rows, dtype=np.int32) - radius
    return shell_index, counts, rows


def radial_shell_average(source, cutoff_cells: float):
    """Calculate every exact radial-shell average with periodic shifts."""

    source = jnp.asarray(source)
    if source.ndim not in (3, 4):
        raise ValueError("source must have shape (X,Y,Z) or (X,Y,Z,C)")
    _, counts, rows = shell_layout(float(cutoff_cells))
    averages = []
    for row, count in zip(rows, np.repeat(counts, counts.astype(np.int32))):
        shifted = jnp.roll(source, tuple(int(value) for value in row), axis=(0, 1, 2))
        averages.append(shifted / jnp.asarray(count, dtype=source.dtype))
    # ``rows`` is ordered shell by shell, so reshape back to (..., shell, ...).
    # Using a second shell loop avoids relying on a Python-side segment operation.
    shell_values = []
    cursor = 0
    for count in counts.astype(np.int32):
        shell_values.append(jnp.sum(jnp.stack(averages[cursor:cursor + count]), axis=0))
        cursor += int(count)
    return jnp.stack(shell_values, axis=3 if source.ndim == 4 else 3)


def shell_local_pair_convolution(source, shell_weights, bias=None, *, cutoff_cells: float = 2.5):
    """Apply shell matrices after one exact shell-average collection.

    ``shell_weights`` has shape ``(shell, input_channels, output_channels)``.
    A scalar source is accepted as a convenience and receives one input
    channel.
    """

    source = jnp.asarray(source)
    scalar = source.ndim == 3
    if scalar:
        source = source[..., None]
    averages = radial_shell_average(source, cutoff_cells)
    weights = jnp.asarray(shell_weights, dtype=source.dtype)
    if averages.shape[-2] != weights.shape[0]:
        raise ValueError(f"shell dimension mismatch: averages={averages.shape}, weights={weights.shape}")
    out = jnp.einsum("xyzsi,sio->xyzo", averages, weights)
    if bias is not None:
        out = out + jnp.asarray(bias, dtype=out.dtype)
    return out[..., 0] if scalar and out.shape[-1] == 1 else out


def dense_local_pair_convolution(source, shell_weights, bias=None, *, cutoff_cells: float = 2.5):
    """Reference dense-offset form of the same radial operator."""

    source = jnp.asarray(source)
    scalar = source.ndim == 3
    if scalar:
        source = source[..., None]
    _, counts, rows = shell_layout(float(cutoff_cells))
    weights = jnp.asarray(shell_weights, dtype=source.dtype)
    out = jnp.zeros(source.shape[:-1] + (weights.shape[-1],), dtype=source.dtype)
    cursor = 0
    for shell_id, count in enumerate(counts.astype(np.int32)):
        shell_sum = jnp.zeros_like(source)
        for row in rows[cursor:cursor + int(count)]:
            shell_sum = shell_sum + jnp.roll(source, tuple(int(value) for value in row), axis=(0, 1, 2))
        out = out + jnp.einsum("xyzi,io->xyzo", shell_sum / count, weights[shell_id])
        cursor += int(count)
    if bias is not None:
        out = out + jnp.asarray(bias, dtype=out.dtype)
    return out[..., 0] if scalar and out.shape[-1] == 1 else out


def shell_fused_halo_network(source, layers, *, cutoff_cells: float = 2.5):
    """Apply a sequence of shell layers through one ownership-preserving API.

    On CPU this is an exact periodic reference.  A distributed kernel can stage
    the total receptive halo once and use the same list of ``layers`` without
    changing the numerical contract.
    """

    value = jnp.asarray(source)
    for weights, bias, activation in layers:
        value = shell_local_pair_convolution(value, weights, bias, cutoff_cells=cutoff_cells)
        if activation is not None:
            value = activation(value)
    return value


def ratio_two_coarse_deposit(positions, values, coarse_shape, *, valid_mask=None, tile_size=4):
    """Directly deposit fine-grid CIC corners into a ratio-two coarse mesh.

    This is algebraically identical to fine deposition followed by a ``2^3``
    block mean.  Duplicate fine corners are deliberately combined by the
    coarse indexed update.
    """

    coarse_shape = tuple(int(value) for value in coarse_shape)
    fine_shape = tuple(2 * value for value in coarse_shape)
    plan = make_cic_plan(positions, fine_shape, valid_mask=valid_mask, tile_size=tile_size)
    coarse_coordinates = plan.coordinates // 2
    coarse_plan = CICPlan(
        plan.positions,
        plan.base,
        coarse_coordinates,
        plan.weights / 8.0,
        plan.valid,
        plan.tile_key,
        plan.particle_order,
        coarse_shape,
        plan.tile_size,
    )
    return _scatter_from_indices(
        coarse_plan.coordinates,
        coarse_plan.weights,
        values,
        coarse_plan.valid,
        coarse_shape,
    )


def ratio_two_coarse_gather(positions, coarse_mesh, *, valid_mask=None, tile_size=4):
    """Gather from a coarse force mesh without materializing three repeats."""

    coarse_mesh = jnp.asarray(coarse_mesh)
    coarse_shape = tuple(int(value) for value in coarse_mesh.shape[:3])
    fine_shape = tuple(2 * value for value in coarse_shape)
    plan = make_cic_plan(positions, fine_shape, valid_mask=valid_mask, tile_size=tile_size)
    coarse_coordinates = plan.coordinates // 2
    return _gather_from_indices(coarse_mesh, coarse_coordinates, plan.weights, plan.valid)


__all__ = [
    "dense_local_pair_convolution",
    "radial_shell_average",
    "ratio_two_coarse_deposit",
    "ratio_two_coarse_gather",
    "shell_fused_halo_network",
    "shell_layout",
    "shell_local_pair_convolution",
]
