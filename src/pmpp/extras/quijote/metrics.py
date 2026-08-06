"""Canonical PM++ particle identity and fixed-grid QUIJOTE metrics.

The force mesh is not the particle identity grid.  In particular, a
``mesh_shape=2`` configuration stores the regular Lagrangian particles at
every second force-mesh point.  :func:`pmpp.core.pmid_to_idx`
therefore remains an internal force-mesh key and must not be used to align a
PM++ state with ID-canonicalized QUIJOTE snapshots.

This module provides the separate particle-grid key, verified gathering of
authoritative particles into that order, and analysis metrics which do not
depend on the force-mesh ratio.  The numerical primitives use JAX arrays so
they can participate in training losses.  Bijection validation is an eager,
host-side setup check; compiled callers should validate a state once and then
pass ``validate=False`` to :func:`gather_authoritative_particles`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AcceptanceSummary", "DenseParticleState", "FixedGridSpectra", "PeriodicPositionMetrics",
    "authoritative_particle_mask", "fixed_grid_spectra", "gather_authoritative_particles", "interlaced_cic_density",
    "periodic_position_metrics", "pmid_to_lagrangian_idx", "summarize_acceptance", "validate_lagrangian_bijection",
]


class DenseParticleState(NamedTuple):
    """Authoritative PM++ fields gathered in Lagrangian-grid order."""

    position: jax.Array
    velocity: jax.Array | None
    acceleration: jax.Array | None
    counts: jax.Array


class FixedGridSpectra(NamedTuple):
    """Interlaced-CIC shell spectra on a force-mesh-independent grid."""

    k: jax.Array
    r: jax.Array
    pk_cross: jax.Array
    pk_reference: jax.Array
    pk_candidate: jax.Array
    nmodes: jax.Array


class PeriodicPositionMetrics(NamedTuple):
    """Minimum-image, row-matched particle-position errors."""

    rmse: jax.Array
    p99: jax.Array
    rmse_cells: jax.Array
    p99_cells: jax.Array
    distances: jax.Array
    distances_cells: jax.Array


class AcceptanceSummary(NamedTuple):
    """The cross-correlation and position hard gates for one snapshot."""

    min_r: jax.Array
    rmse_cells: jax.Array
    p99_cells: jax.Array
    evaluated_shells: jax.Array
    cross_pass: jax.Array
    positions_pass: jax.Array
    passed: jax.Array


def pmid_to_lagrangian_idx(pmid, conf, unused_index=None, dtype=jnp.int32):
    """Ravel PM++ particle anchors on the Lagrangian particle grid.

    Unlike :func:`pmpp.core.pmid_to_idx`, this mapping uses
    ``conf.ptcl_grid_shape`` for the returned key.  Force-mesh anchors are
    mapped back to their nearest regular particle-grid coordinate before
    raveling in C order (z fastest).  This is exact for the supported force
    mesh ratios one and two and also follows PM++'s nearest-anchor convention
    for other mesh refinements.

    Parameters
    ----------
    pmid
        Integer force-mesh anchor coordinates with shape ``(..., dim)``.
    conf
        PM++ configuration defining ``mesh_shape`` and ``ptcl_grid_shape``.
    unused_index
        Optional padding mask.  Masked keys are returned as ``-1``.
    dtype
        Signed integer dtype used for the packed key.

    Returns
    -------
    jax.Array
        Lagrangian-grid indices in ``[0, conf.ptcl_num)`` (or ``-1`` for
        explicitly masked slots).
    """

    pmid = jnp.asarray(pmid)
    particle_shape = tuple(int(size) for size in conf.ptcl_grid_shape)
    mesh_shape = tuple(int(size) for size in conf.mesh_shape)
    if pmid.ndim < 1 or pmid.shape[-1] != len(particle_shape):
        raise ValueError(
            "pmid must have final dimension equal to the particle-grid "
            f"dimension {len(particle_shape)}; got shape {pmid.shape}"
        )
    if not jnp.issubdtype(jnp.dtype(dtype), jnp.signedinteger):
        raise ValueError("dtype must be a signed integer dtype")

    # Use integer round-half-up arithmetic.  The ratio-1 and ratio-2 anchors
    # are exact multiples, so no rounding ambiguity is present in the two
    # acceptance configurations.
    coordinates = []
    for axis, (particle_size, mesh_size) in enumerate(zip(particle_shape, mesh_shape)):
        wrapped = jnp.mod(pmid[..., axis].astype(dtype), dtype(mesh_size))
        numerator = wrapped * dtype(particle_size)
        coordinate = jnp.floor_divide(2 * numerator + dtype(mesh_size), dtype(2 * mesh_size))
        coordinates.append(jnp.mod(coordinate, dtype(particle_size)))

    idx = jnp.zeros(pmid.shape[:-1], dtype=dtype)
    for coordinate, particle_size in zip(coordinates, particle_shape):
        idx = idx * dtype(particle_size) + coordinate

    if unused_index is not None:
        unused_index = jnp.asarray(unused_index, dtype=jnp.bool_)
        if unused_index.shape != idx.shape:
            raise ValueError(
                "unused_index must match pmid leading dimensions; "
                f"got {unused_index.shape} and {pmid.shape[:-1]}"
            )
        idx = jnp.where(unused_index, dtype(-1), idx)
    return idx


def authoritative_particle_mask(ptcl) -> jax.Array:
    """Return slots containing physical, non-halo PM++ particles."""

    shape = ptcl.pmid.shape[:-1]
    mask = jnp.ones(shape, dtype=jnp.bool_)
    if ptcl.unused_index is not None:
        unused = jnp.asarray(ptcl.unused_index, dtype=jnp.bool_)
        if unused.shape != shape:
            raise ValueError("unused_index must match particle slots; "
                             f"got {unused.shape} and {shape}")
        mask = mask & ~unused
    if ptcl.halo_mask is not None:
        halo = jnp.asarray(ptcl.halo_mask, dtype=jnp.bool_)
        if halo.shape != shape:
            raise ValueError(f"halo_mask must match particle slots; got {halo.shape} and {shape}")
        mask = mask & ~halo
    return mask


def _preview_indices(indices: np.ndarray, limit: int = 5) -> str:
    values = [int(value) for value in indices[:limit]]
    suffix = ", ..." if indices.size > limit else ""
    return "[" + ", ".join(map(str, values)) + suffix + "]"


def validate_lagrangian_bijection(indices, authoritative_mask, particle_count: int):
    """Assert exactly one authoritative particle for every Lagrangian key.

    This function intentionally performs an eager host check.  It should run
    while loading/building a trajectory, before a compiled training loop.

    Returns
    -------
    numpy.ndarray
        Per-key counts (all ones when validation succeeds).

    Raises
    ------
    ValueError
        If shapes, index range, duplicates, or missing particle identities are
        found.
    """

    if isinstance(particle_count, (bool, np.bool_)):
        raise ValueError("particle_count must be a positive integer")
    try:
        count = int(particle_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("particle_count must be a positive integer") from error
    if count <= 0 or count != particle_count:
        raise ValueError("particle_count must be a positive integer")

    try:
        keys = np.asarray(jax.device_get(indices))
        mask = np.asarray(jax.device_get(authoritative_mask), dtype=bool)
    except Exception as error:
        raise ValueError(
            "bijection validation must run eagerly outside jax.jit; validate "
            "once before compilation and use validate=False in traced code"
        ) from error
    if keys.shape != mask.shape:
        raise ValueError(
            f"indices and authoritative_mask must have the same shape; got "
            f"{keys.shape} and {mask.shape}"
        )
    if not np.issubdtype(keys.dtype, np.integer):
        raise ValueError(f"indices must use an integer dtype; got {keys.dtype}")

    authoritative = keys[mask].reshape(-1)
    invalid = authoritative[(authoritative < 0) | (authoritative >= count)]
    if invalid.size:
        raise ValueError(
            "authoritative Lagrangian indices are out of range "
            f"[0, {count}): {_preview_indices(invalid)}"
        )

    counts = np.bincount(authoritative.astype(np.int64, copy=False), minlength=count)
    missing = np.flatnonzero(counts == 0)
    duplicates = np.flatnonzero(counts > 1)
    if missing.size or duplicates.size:
        raise ValueError(
            "authoritative particles are not a bijection over the Lagrangian "
            f"grid: authoritative={authoritative.size}, expected={count}, "
            f"missing={_preview_indices(missing)}, "
            f"duplicates={_preview_indices(duplicates)}"
        )
    return counts


def _dense_masked_add(values, indices, mask, particle_count):
    values = jnp.asarray(values)
    safe_indices = jnp.where(mask, indices, jnp.zeros_like(indices))
    expand = mask.reshape(mask.shape + (1, ) * (values.ndim - mask.ndim))
    masked_values = jnp.where(expand, values, jnp.zeros_like(values))
    dense_shape = (particle_count, ) + values.shape[mask.ndim:]
    return jnp.zeros(dense_shape, dtype=values.dtype).at[safe_indices].add(masked_values)


def gather_authoritative_particles(ptcl, conf=None, *, validate: bool = True):
    """Gather a PM++ state into canonical Lagrangian-grid order.

    Halo copies and padding slots are excluded.  With the default
    ``validate=True``, the particle identity is checked eagerly before fields
    are gathered.  Set ``validate=False`` only in compiled code after that
    setup check; the gather and its position/velocity gradients remain normal
    JAX operations.
    """

    if conf is None:
        conf = ptcl.conf
    particle_count = int(math.prod(tuple(int(size) for size in conf.ptcl_grid_shape)))
    mask = authoritative_particle_mask(ptcl)
    indices = pmid_to_lagrangian_idx(ptcl.pmid, conf)
    if validate:
        validate_lagrangian_bijection(indices, mask, particle_count)

    position_dtype = ptcl.disp.dtype
    box = jnp.asarray(conf.box_size, dtype=position_dtype)
    position = jnp.mod(
        ptcl.pmid.astype(position_dtype) * jnp.asarray(conf.cell_size, position_dtype) + ptcl.disp, box,
    )
    dense_position = _dense_masked_add(position, indices, mask, particle_count)
    dense_velocity = (None if ptcl.vel is None else _dense_masked_add(ptcl.vel, indices, mask, particle_count))
    dense_acceleration = (None if ptcl.acc is None else _dense_masked_add(ptcl.acc, indices, mask, particle_count))
    counts = jnp.zeros((particle_count, ),
                       dtype=jnp.int32).at[jnp.where(mask, indices,
                                                     jnp.zeros_like(indices))].add(mask.astype(jnp.int32))
    return DenseParticleState(dense_position, dense_velocity, dense_acceleration, counts, )


def _analysis_grid_size(analysis_grid) -> int:
    if isinstance(analysis_grid, (bool, np.bool_)):
        raise ValueError("analysis_grid must be a positive integer")
    try:
        grid = int(analysis_grid)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("analysis_grid must be a positive integer") from error
    if grid <= 1 or grid != analysis_grid:
        raise ValueError("analysis_grid must be an integer greater than one")
    return grid


def _box_vector(box_size, dtype):
    box = jnp.asarray(box_size, dtype=dtype)
    if box.ndim == 0:
        box = jnp.repeat(box[None], 3)
    if box.shape != (3, ):
        raise ValueError(f"box_size must be a scalar or length-3 vector; got {box.shape}")
    return box


def _validate_positions_shape(positions, label):
    positions = jnp.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3); got {positions.shape}")
    if positions.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one particle")
    if not jnp.issubdtype(positions.dtype, jnp.floating):
        positions = positions.astype(jnp.float32)
    return positions


def _cic_density(positions, box, grid: int, shift_cells: float):
    dtype = positions.dtype
    cell = box / jnp.asarray(grid, dtype=dtype)
    grid_position = jnp.mod(positions, box) / cell - jnp.asarray(shift_cells, dtype=dtype)
    lower_float = jnp.floor(grid_position)
    lower = lower_float.astype(jnp.int32)
    fraction = grid_position - lower_float
    flat_density = jnp.zeros((grid**3, ), dtype=dtype)

    for offset_x in (0, 1):
        weight_x = fraction[:, 0] if offset_x else 1 - fraction[:, 0]
        index_x = jnp.mod(lower[:, 0] + offset_x, grid)
        for offset_y in (0, 1):
            weight_y = fraction[:, 1] if offset_y else 1 - fraction[:, 1]
            index_y = jnp.mod(lower[:, 1] + offset_y, grid)
            for offset_z in (0, 1):
                weight_z = fraction[:, 2] if offset_z else 1 - fraction[:, 2]
                index_z = jnp.mod(lower[:, 2] + offset_z, grid)
                flat_index = (index_x * grid + index_y) * grid + index_z
                flat_density = flat_density.at[flat_index].add(weight_x * weight_y * weight_z)

    mean_count = jnp.asarray(positions.shape[0] / (grid**3), dtype=dtype)
    return flat_density.reshape((grid, grid, grid)) / mean_count


def _mode_vectors(grid: int, dtype):
    mode_x = jnp.fft.fftfreq(grid, d=1 / grid).astype(dtype)
    mode_y = jnp.fft.fftfreq(grid, d=1 / grid).astype(dtype)
    mode_z = jnp.fft.rfftfreq(grid, d=1 / grid).astype(dtype)
    return mode_x, mode_y, mode_z


def _interlaced_cic_delta_spectrum(positions, box, grid: int):
    density_base = _cic_density(positions, box, grid, 0.0)
    density_shifted = _cic_density(positions, box, grid, 0.5)
    spectral_base = jnp.fft.rfftn(density_base - 1)
    spectral_shifted = jnp.fft.rfftn(density_shifted - 1)

    mode_x, mode_y, mode_z = _mode_vectors(grid, positions.dtype)
    phase_argument = -jnp.pi * (mode_x[:, None, None] + mode_y[None, :, None] +
                                mode_z[None, None, :]) / jnp.asarray(grid, dtype=positions.dtype)
    phase = jnp.exp(1j * phase_argument).astype(spectral_base.dtype)
    return 0.5 * (spectral_base + phase * spectral_shifted)


def interlaced_cic_density(positions, box_size, *, analysis_grid: int = 256):
    """Deposit positions onto an interlaced CIC analysis grid.

    The default is the fixed ``256^3`` QUIJOTE acceptance grid.  A smaller
    explicit value is supported for unit tests and lower-resolution training
    curricula.  The returned density is normalized to mean one.
    """

    grid = _analysis_grid_size(analysis_grid)
    positions = _validate_positions_shape(positions, "positions")
    box = _box_vector(box_size, positions.dtype)
    spectrum = _interlaced_cic_delta_spectrum(positions, box, grid)
    delta = jnp.fft.irfftn(spectrum, s=(grid, grid, grid)).real
    return delta + jnp.asarray(1, dtype=delta.dtype)


def _deconvolve_cic(spectrum, grid: int, dtype):
    mode_x, mode_y, mode_z = _mode_vectors(grid, dtype)
    window = (
        jnp.sinc(mode_x[:, None, None] / grid)**2 * jnp.sinc(mode_y[None, :, None] / grid)**2 *
        jnp.sinc(mode_z[None, None, :] / grid)**2
    )
    floor = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    return spectrum / jnp.maximum(window, floor).astype(spectrum.dtype)


def fixed_grid_spectra(
    reference_positions, candidate_positions, box_size, *, analysis_grid: int = 256, deconvolve_cic: bool = True,
    eps: float = 1e-30,
):
    """Compute interlaced-CIC power and cross spectra on one fixed grid.

    Both position arrays are deposited with identical window treatment and
    reduced with identical integer-radius Fourier bins.  Consequently the
    result is directly comparable between PM++ force-mesh ratios one and two.
    QUIJOTE acceptance should retain the default ``analysis_grid=256``.
    """

    grid = _analysis_grid_size(analysis_grid)
    reference = _validate_positions_shape(reference_positions, "reference_positions")
    candidate = _validate_positions_shape(candidate_positions, "candidate_positions")
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference_positions and candidate_positions must be ID-matched "
            f"arrays with the same shape; got {reference.shape} and {candidate.shape}"
        )
    dtype = jnp.result_type(reference.dtype, candidate.dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.dtype(jnp.float32)
    reference = reference.astype(dtype)
    candidate = candidate.astype(dtype)
    box = _box_vector(box_size, dtype)

    reference_spectrum = _interlaced_cic_delta_spectrum(reference, box, grid)
    candidate_spectrum = _interlaced_cic_delta_spectrum(candidate, box, grid)
    if deconvolve_cic:
        reference_spectrum = _deconvolve_cic(reference_spectrum, grid, dtype)
        candidate_spectrum = _deconvolve_cic(candidate_spectrum, grid, dtype)

    mode_x, mode_y, mode_z = _mode_vectors(grid, dtype)
    mode_sq = (mode_x[:, None, None]**2 + mode_y[None, :, None]**2 + mode_z[None, None, :]**2)
    shell = jnp.floor(jnp.sqrt(mode_sq) + jnp.asarray(1e-6, dtype=dtype)).astype(jnp.int32)
    num_shells = math.floor(math.sqrt(3) * (grid // 2)) + 1

    multiplicity_z = jnp.full((grid // 2 + 1, ), 2, dtype=jnp.int32)
    multiplicity_z = multiplicity_z.at[0].set(1)
    if grid % 2 == 0:
        multiplicity_z = multiplicity_z.at[-1].set(1)
    multiplicity = jnp.broadcast_to(multiplicity_z, shell.shape)
    weights = multiplicity.astype(dtype)
    shell_flat = shell.reshape(-1)

    auto_reference = (reference_spectrum.real**2 + reference_spectrum.imag**2).astype(dtype)
    auto_candidate = (candidate_spectrum.real**2 + candidate_spectrum.imag**2).astype(dtype)
    cross = (reference_spectrum * jnp.conj(candidate_spectrum)).real.astype(dtype)

    def reduce(values):
        return jnp.bincount(shell_flat, weights=(values * weights).reshape(-1), length=num_shells, )

    reference_sum = reduce(auto_reference)
    candidate_sum = reduce(auto_candidate)
    cross_sum = reduce(cross)
    nmodes = jnp.bincount(shell_flat, weights=weights.reshape(-1), length=num_shells, )

    kx = 2 * jnp.pi * mode_x[:, None, None] / box[0]
    ky = 2 * jnp.pi * mode_y[None, :, None] / box[1]
    kz = 2 * jnp.pi * mode_z[None, None, :] / box[2]
    mode_k = jnp.sqrt(kx**2 + ky**2 + kz**2)
    k_sum = jnp.bincount(shell_flat, weights=(mode_k * weights).reshape(-1), length=num_shells, )
    safe_modes = jnp.maximum(nmodes, jnp.asarray(1, dtype=nmodes.dtype))
    k = k_sum / safe_modes

    box_volume = jnp.prod(box)
    normalization = box_volume / jnp.asarray(grid**6, dtype=dtype)
    pk_reference = reference_sum / safe_modes * normalization
    pk_candidate = candidate_sum / safe_modes * normalization
    pk_cross = cross_sum / safe_modes * normalization
    denominator = jnp.sqrt(jnp.maximum(pk_reference * pk_candidate, jnp.asarray(eps, dtype=dtype)))
    r = jnp.clip(pk_cross / denominator, -1, 1)

    # The DC shell is not a physical density-fluctuation comparison.
    return FixedGridSpectra(
        k[1:], r[1:], pk_cross[1:], pk_reference[1:], pk_candidate[1:], nmodes[1:].astype(jnp.int32),
    )


def periodic_position_metrics(predicted_positions, target_positions, box_size, particle_cell_size, ):
    """Compute periodic ID-matched RMSE and p99 position errors.

    Rows must already share canonical particle identity, normally by combining
    :func:`gather_authoritative_particles` with
    :class:`pmpp.extras.quijote.QuijoteCanonicalization`.
    """

    predicted = _validate_positions_shape(predicted_positions, "predicted_positions")
    target = _validate_positions_shape(target_positions, "target_positions")
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted_positions and target_positions must be ID-matched arrays "
            f"with the same shape; got {predicted.shape} and {target.shape}"
        )
    dtype = jnp.result_type(predicted.dtype, target.dtype)
    predicted = predicted.astype(dtype)
    target = target.astype(dtype)
    box = _box_vector(box_size, dtype)
    cell = jnp.asarray(particle_cell_size, dtype=dtype)
    if cell.ndim != 0:
        raise ValueError("particle_cell_size must be a scalar")

    delta = jnp.mod(predicted - target + 0.5 * box, box) - 0.5 * box
    distances = jnp.linalg.norm(delta, axis=-1)
    distances_cells = distances / cell
    rmse = jnp.sqrt(jnp.mean(distances**2))
    p99 = jnp.quantile(distances, jnp.asarray(0.99, dtype=dtype))
    rmse_cells = jnp.sqrt(jnp.mean(distances_cells**2))
    p99_cells = jnp.quantile(distances_cells, jnp.asarray(0.99, dtype=dtype))
    return PeriodicPositionMetrics(rmse, p99, rmse_cells, p99_cells, distances, distances_cells, )


def summarize_acceptance(
    spectra: FixedGridSpectra, positions: PeriodicPositionMetrics, *, k_max, cross_threshold: float = 0.999,
    rmse_cells_max: float = 0.01, p99_cells_max: float = 0.05,
):
    """Summarize the held-out cross and particle-position hard gates."""

    k_max = jnp.asarray(k_max, dtype=spectra.k.dtype)
    valid = (
        jnp.isfinite(spectra.k)
        & jnp.isfinite(spectra.r)
        & (spectra.k > 0)
        & (spectra.k <= k_max)
        & (spectra.nmodes > 0)
    )
    evaluated_shells = jnp.sum(valid.astype(jnp.int32))
    minimum = jnp.min(jnp.where(valid, spectra.r, jnp.inf))
    min_r = jnp.where(evaluated_shells > 0, minimum, jnp.nan)
    cross_pass = (evaluated_shells > 0) & (min_r >= cross_threshold)
    positions_pass = (
        jnp.isfinite(positions.rmse_cells)
        & jnp.isfinite(positions.p99_cells)
        & (positions.rmse_cells <= rmse_cells_max)
        & (positions.p99_cells <= p99_cells_max)
    )
    return AcceptanceSummary(
        min_r, positions.rmse_cells, positions.p99_cells, evaluated_shells, cross_pass, positions_pass,
        cross_pass & positions_pass,
    )
