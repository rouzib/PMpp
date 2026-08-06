"""Canonical particle ordering helpers for QUIJOTE snapshots.

QUIJOTE particle IDs identify the same particles across snapshots, but their
numeric order is not the regular Lagrangian-grid order expected by PM++.
This module reconstructs that grid order from an initial snapshot and applies
it consistently to any later ID-labelled particle arrays.

Only NumPy is required.  In particular, these helpers are intended for host-
side dataset preparation rather than for a JAX-traced simulation step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["QuijoteCanonicalization", "build_quijote_canonicalization", "canonicalize_quijote_arrays", ]


@dataclass(frozen=True)
class QuijoteCanonicalization:
    """Verified mapping from particle IDs to the regular QUIJOTE q grid.

    Parameters stored in this object are produced by
    :func:`build_quijote_canonicalization`; users should not normally
    instantiate it directly.

    Attributes
    ----------
    grid_size
        Number of Lagrangian cells along each axis.
    box_size
        Periodic box size in the same position units used at construction.
    a_start
        Initial scale factor used to undo the first-order velocity shift.
    expansion_rate
        Dimensionless expansion rate ``E(a_start) = H(a_start) / H0``.
    sorted_ids
        Particle IDs in ascending numeric order.
    q_by_sorted_id
        Reconstructed integer ``(qx, qy, qz)`` for ``sorted_ids``.
    q_permutation
        Indices that transform an ID-sorted array to canonical q-grid order.
    max_q_residual
        Largest absolute distance, in grid-cell units, between reconstructed
        q coordinates and their nearest integer coordinates.
    """

    grid_size: int
    box_size: float
    a_start: float
    expansion_rate: float
    sorted_ids: np.ndarray
    q_by_sorted_id: np.ndarray
    q_permutation: np.ndarray
    max_q_residual: float

    @property
    def particle_count(self) -> int:
        """Return the number of particles represented by the mapping."""

        return int(self.sorted_ids.size)

    def permutation_for(self, ids: Any) -> np.ndarray:
        """Return indices that put an ID-labelled array in canonical q order.

        ``ids`` may be in any order, but must be unique and contain exactly
        the same ID set as the initial snapshot used to build this object.
        """

        return _permutation_for_matching_ids(self, ids)

    def apply(self, ids: Any, *arrays: Any) -> tuple[np.ndarray, ...]:
        """Canonicalize one or more arrays sharing the ordering of ``ids``."""

        return canonicalize_quijote_arrays(self, ids, *arrays)

    def canonical_q_coordinates(self, dtype: Any | None = None) -> np.ndarray:
        """Return the reconstructed q coordinates in canonical grid order.

        The returned rows follow NumPy C order: z varies fastest, followed by
        y and then x.  ``dtype=None`` preserves the compact integer dtype used
        internally.
        """

        q = self.q_by_sorted_id[self.q_permutation]
        if dtype is None:
            return q
        return q.astype(dtype, copy=False)


def build_quijote_canonicalization(
    ic_ids: Any, ic_pos: Any, ic_vel: Any, *, box_size: float, a_start: float, expansion_rate: float,
    grid_size: int | None = None, q_tolerance: float = 0.25,
) -> QuijoteCanonicalization:
    """Reconstruct and verify the regular q-grid ordering of QUIJOTE ICs.

    QUIJOTE initial positions include the velocity displacement associated
    with their starting scale factor.  After sorting by particle ID, the
    Lagrangian coordinates are reconstructed as

    ``q_est = (ic_pos - ic_vel / (a_start**2 * E(a_start))) / cell_size``.

    The nearest integer coordinates are reduced periodically into the box.
    Construction succeeds only when every coordinate is within
    ``q_tolerance`` cells of an integer and the reconstructed coordinates form
    an exact bijection over the ``grid_size**3`` regular grid.

    Parameters
    ----------
    ic_ids
        Unique one-dimensional particle IDs.
    ic_pos, ic_vel
        Initial particle positions and velocities with shape ``(N, 3)`` and
        matching raw particle order.
    box_size
        Periodic box size in the same units as ``ic_pos`` and the velocity
        displacement in the reconstruction formula.
    a_start
        Initial scale factor.
    expansion_rate
        Dimensionless ``E(a_start) = H(a_start) / H0``.
    grid_size
        Grid size per dimension.  If omitted, it is inferred from the exact
        integer cube root of ``N``.
    q_tolerance
        Maximum accepted reconstruction residual in grid-cell units.  It must
        lie in ``[0, 0.5)``.

    Returns
    -------
    QuijoteCanonicalization
        A reusable, verified ID-to-q mapping.

    Raises
    ------
    ValueError
        If shapes, IDs, scalar parameters, reconstruction tolerance, or the
        q-grid bijection are invalid.
    """

    ids, id_sort, sorted_ids = _validate_ids(ic_ids, label="IC particle IDs")
    particle_count = int(ids.size)
    n = _resolve_grid_size(particle_count, grid_size)

    pos = _validate_particle_vectors(ic_pos, particle_count, label="ic_pos")
    vel = _validate_particle_vectors(ic_vel, particle_count, label="ic_vel")
    box = _positive_finite_scalar(box_size, label="box_size")
    start = _positive_finite_scalar(a_start, label="a_start")
    e_start = _positive_finite_scalar(expansion_rate, label="expansion_rate")
    tolerance = _q_tolerance(q_tolerance)

    calculation_dtype = np.result_type(pos.dtype, vel.dtype, np.float32)
    pos_by_id = np.asarray(pos[id_sort], dtype=calculation_dtype)
    vel_by_id = np.asarray(vel[id_sort], dtype=calculation_dtype)
    cell_size = box / n
    velocity_factor = start * start * e_start
    q_est = (pos_by_id - vel_by_id / velocity_factor) / cell_size

    if not np.all(np.isfinite(q_est)):
        raise ValueError("reconstructed q coordinates contain non-finite values")

    q_nearest = np.rint(q_est)
    residual = np.abs(q_est - q_nearest)
    max_flat = int(np.argmax(residual))
    max_residual = float(residual.reshape(-1)[max_flat])
    if max_residual > tolerance:
        particle_row, axis = divmod(max_flat, 3)
        particle_id = sorted_ids[particle_row].item()
        raise ValueError(
            "q reconstruction exceeds q_tolerance: "
            f"maximum residual {max_residual:.8g} cells at particle ID "
            f"{particle_id}, axis {axis}; tolerance is {tolerance:.8g}"
        )

    q_dtype = np.min_scalar_type(n - 1)
    q_by_sorted_id = np.remainder(q_nearest, n).astype(q_dtype, copy=False)
    qx = q_by_sorted_id[:, 0].astype(np.int64, copy=False)
    qy = q_by_sorted_id[:, 1].astype(np.int64, copy=False)
    qz = q_by_sorted_id[:, 2].astype(np.int64, copy=False)
    q_linear = qz + n * (qy + n * qx)
    q_permutation = np.argsort(q_linear, kind="stable")
    ordered_linear = q_linear[q_permutation]
    expected_linear = np.arange(particle_count, dtype=np.int64)

    if not np.array_equal(ordered_linear, expected_linear):
        mismatch = int(np.flatnonzero(ordered_linear != expected_linear)[0])
        unique_count = int(np.unique(q_linear).size)
        raise ValueError(
            "reconstructed q coordinates are not a bijection over the regular "
            f"{n}^3 grid: found {unique_count} unique cells for "
            f"{particle_count} particles; first ordered mismatch is at linear "
            f"cell {mismatch} (found {int(ordered_linear[mismatch])})"
        )

    if particle_count <= np.iinfo(np.int32).max + 1:
        q_permutation = q_permutation.astype(np.int32, copy=False)

    sorted_ids = np.array(sorted_ids, copy=True)
    q_by_sorted_id = np.array(q_by_sorted_id, copy=True)
    q_permutation = np.array(q_permutation, copy=True)
    sorted_ids.flags.writeable = False
    q_by_sorted_id.flags.writeable = False
    q_permutation.flags.writeable = False

    return QuijoteCanonicalization(
        grid_size=n, box_size=box, a_start=start, expansion_rate=e_start, sorted_ids=sorted_ids,
        q_by_sorted_id=q_by_sorted_id, q_permutation=q_permutation, max_q_residual=max_residual,
    )


def canonicalize_quijote_arrays(canonicalization: QuijoteCanonicalization, ids: Any, *arrays: Any,
                                ) -> tuple[np.ndarray, ...]:
    """Put one or more ID-labelled particle arrays in canonical q-grid order.

    Each array must have particle count as its first dimension and share the
    raw ordering of ``ids``.  Trailing dimensions and dtypes are preserved.
    The function always returns a tuple, including when only one array is
    supplied.
    """

    if not isinstance(canonicalization, QuijoteCanonicalization):
        raise TypeError("canonicalization must be a QuijoteCanonicalization")
    if not arrays:
        raise ValueError("at least one particle array is required")

    permutation = canonicalization.permutation_for(ids)
    canonical_arrays: list[np.ndarray] = []
    for index, values in enumerate(arrays):
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != canonicalization.particle_count:
            raise ValueError(
                f"array {index} must have first dimension "
                f"{canonicalization.particle_count}; got shape {array.shape}"
            )
        canonical_arrays.append(array[permutation])
    return tuple(canonical_arrays)


def _permutation_for_matching_ids(canonicalization: QuijoteCanonicalization, ids: Any, ) -> np.ndarray:
    _, id_sort, sorted_ids = _validate_ids(ids, label="particle IDs", expected_count=canonicalization.particle_count, )
    if not np.array_equal(sorted_ids, canonicalization.sorted_ids):
        missing = np.setdiff1d(canonicalization.sorted_ids, sorted_ids)
        unexpected = np.setdiff1d(sorted_ids, canonicalization.sorted_ids)
        raise ValueError(
            "particle ID set does not match the canonical IC IDs; "
            f"missing={_id_preview(missing)}, "
            f"unexpected={_id_preview(unexpected)}"
        )
    return id_sort[canonicalization.q_permutation]


def _validate_ids(ids: Any, *, label: str, expected_count: int | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(ids)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{label} must use an integer dtype; got {array.dtype}")
    if expected_count is not None and array.size != expected_count:
        raise ValueError(f"{label} must contain {expected_count} entries; got {array.size}")
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")

    order = np.argsort(array, kind="stable")
    sorted_ids = array[order]
    duplicate_locations = np.flatnonzero(sorted_ids[1:] == sorted_ids[:-1])
    if duplicate_locations.size:
        duplicate = sorted_ids[int(duplicate_locations[0])].item()
        raise ValueError(f"{label} must be unique; duplicate ID {duplicate}")
    return array, order, sorted_ids


def _validate_particle_vectors(values: Any, count: int, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count, 3):
        raise ValueError(f"{label} must have shape ({count}, 3); got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{label} must contain real numeric values; got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    return array


def _resolve_grid_size(particle_count: int, grid_size: int | None) -> int:
    if grid_size is None:
        approximate = int(round(particle_count**(1.0 / 3.0)))
        candidates = range(max(1, approximate - 1), approximate + 2)
        exact = next((candidate for candidate in candidates if candidate**3 == particle_count), None, )
        if exact is None:
            raise ValueError(
                "particle count is not an exact integer cube; provide valid "
                f"grid_size for {particle_count} particles"
            )
        return exact

    if isinstance(grid_size, (bool, np.bool_)):
        raise ValueError("grid_size must be a positive integer")
    try:
        n = int(grid_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("grid_size must be a positive integer") from error
    if n <= 0 or n != grid_size:
        raise ValueError("grid_size must be a positive integer")
    if n**3 != particle_count:
        raise ValueError(f"grid_size={n} requires {n**3} particles; got {particle_count}")
    return n


def _positive_finite_scalar(value: Any, *, label: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a positive finite scalar") from error
    if not np.isfinite(scalar) or scalar <= 0:
        raise ValueError(f"{label} must be a positive finite scalar")
    return scalar


def _q_tolerance(value: Any) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("q_tolerance must be a finite scalar in [0, 0.5)") from error
    if not np.isfinite(tolerance) or tolerance < 0 or tolerance >= 0.5:
        raise ValueError("q_tolerance must be a finite scalar in [0, 0.5)")
    return tolerance


def _id_preview(ids: np.ndarray, limit: int = 3) -> str:
    values = [value.item() for value in ids[:limit]]
    suffix = ", ..." if ids.size > limit else ""
    return "[" + ", ".join(map(str, values)) + suffix + "]"
