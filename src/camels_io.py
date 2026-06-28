from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np

PLANCK18_OMEGA_B = 0.04897


@dataclass(frozen=True)
class CamelsMetadata:
    """Simulation metadata needed to build PM++ CAMELS comparison runs."""

    box_size: float
    omega_m: float
    omega_l: float
    omega_b: float
    h: float
    sigma8: float
    n_s: float
    a_start: float
    redshift: float
    grid_size: int


@dataclass(frozen=True)
class CamelsParticlePair:
    """Paired initial and final CAMELS particle arrays in a common ID order."""

    ic_pos: np.ndarray
    ic_vel: np.ndarray
    final_pos: np.ndarray
    final_vel: np.ndarray
    ids: np.ndarray
    metadata: CamelsMetadata


def periodic_wrap(pos, box_size):
    """Wrap positions into a periodic box.

    Parameters
    ----------
    pos : array-like
        Positions to wrap.
    box_size : float
        Periodic box size.

    Returns
    -------
    numpy.ndarray
        Wrapped positions in ``[0, box_size)``.
    """
    return np.mod(pos, box_size)


def periodic_delta(pos, anchor, box_size):
    """Shortest periodic displacement from ``anchor`` to ``pos``.

    Parameters
    ----------
    pos : array-like
        Target positions.
    anchor : array-like
        Reference positions.
    box_size : float
        Periodic box size.

    Returns
    -------
    numpy.ndarray
        Minimum-image displacement vectors.
    """
    return ((pos - anchor + 0.5 * box_size) % box_size) - 0.5 * box_size


def gadget_velocity_to_pmpp(vel, redshift):
    """Convert Gadget velocities into the notebook/npz PM units.

    The showcase notebook uses the saved `*.npz` CAMELS files directly. Those
    files store velocities in the PM units expected by PM++ / PMWD rather than
    raw Gadget km/s values. Empirically, that convention is:

    `v_pm = v_gadget / 100 * a`, with `a = 1 / (1 + z)`.
    
    Parameters
    ----------
    vel : array-like
        Gadget velocities.
    redshift : float
        Snapshot redshift.

    Returns
    -------
    numpy.ndarray
        Velocities converted into the PM units used by PM++ notebook CAMELS
        files.
    """
    a = 1.0 / (1.0 + float(redshift))
    return np.asarray(vel, dtype=np.float32) / 100.0 * a


def _infer_grid_size(num_particles):
    """Infer the side length of a cubic Lagrangian particle grid."""
    grid_size = round(num_particles ** (1 / 3))
    if grid_size ** 3 != num_particles:
        raise ValueError(f"Expected a cubic particle grid, got {num_particles} particles.")
    return grid_size


def _parse_two_lpt_param(path: Path) -> Dict[str, float]:
    """Parse the simple whitespace ``2LPT.param`` files shipped with CAMELS."""
    params = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, value = parts[0], parts[1]
            try:
                params[key] = float(value)
            except ValueError:
                continue
    return params


def _load_camels_snapshot_npz(path: Path):
    """Load the cached final CAMELS snapshot format used by local scripts."""
    with np.load(path, allow_pickle=True) as data:
        return {
            "pos": np.asarray(data["pos"], dtype=np.float32),
            "vel": np.asarray(data["vel"], dtype=np.float32),
            "ids": np.asarray(data["ids"], dtype=np.int64),
            "BoxSize": float(data["BoxSize"]),
            "Omega_m": float(data["Omega_m"]),
            "Omega_l": float(data["Omega_l"]),
            "redshift": float(data["redshift"]),
        }


def _load_camels_ics_npz(path: Path):
    """Load cached CAMELS IC arrays from ``ics.npz``."""
    with np.load(path, allow_pickle=True) as data:
        return {
            "pos": np.asarray(data["pos"], dtype=np.float32),
            "vel": np.asarray(data["vel"], dtype=np.float32),
            "ids": np.asarray(data["ids"], dtype=np.int64),
            "BoxSize": float(data["BoxSize"]),
            "Omega_m": float(data["Omega_m"]),
            "Omega_l": float(data["Omega_l"]),
            "redshift": float(data["redshift"]),
        }


def _load_camels_ics_hdf5(ics_dir: Path):
    """Load raw CAMELS HDF5 IC shards and sort particles by ID."""
    pos, vel, ids = [], [], []
    for path in sorted(ics_dir.glob("ics.*.hdf5")):
        with h5py.File(path, "r") as handle:
            part = handle["PartType1"]
            pos.append(np.asarray(part["Coordinates"], dtype=np.float32))
            vel.append(np.asarray(part["Velocities"], dtype=np.float32))
            ids.append(np.asarray(part["ParticleIDs"], dtype=np.int64))

    if not pos:
        raise FileNotFoundError(f"No CAMELS IC files were found in {ics_dir!s}.")

    params = _parse_two_lpt_param(ics_dir / "2LPT.param")
    redshift = float(params.get("Redshift", 127.0))
    pos = np.concatenate(pos, axis=0)
    vel = np.concatenate(vel, axis=0)
    ids = np.concatenate(ids, axis=0)
    order = np.argsort(ids - 1)
    return {
        "pos": np.asarray(pos[order] / 1e3, dtype=np.float32),
        "vel": gadget_velocity_to_pmpp(vel[order], redshift),
        "ids": np.asarray(ids[order] - 1, dtype=np.int64),
        "BoxSize": float(params.get("Box", 25000.0) / 1e3),
        "Omega_m": float(params.get("Omega", 0.3)),
        "Omega_l": float(params.get("OmegaLambda", 0.7)),
        "redshift": redshift,
    }


def load_camels_pair(base_dir):
    """Load CAMELS ICs and final snapshot as a paired supervised dataset.

    Parameters
    ----------
    base_dir : str or pathlib.Path
        Directory containing CAMELS IC and final snapshot files.

    Returns
    -------
    CamelsParticlePair
        Paired initial/final particle arrays with aligned IDs and metadata.
    """
    base_dir = Path(base_dir)
    snapshot_npz = base_dir / "snapshot_090.npz"
    if snapshot_npz.exists():
        snapshot = _load_camels_snapshot_npz(snapshot_npz)
    else:
        raise FileNotFoundError(f"Expected {snapshot_npz!s} to exist.")

    ics_npz = base_dir / "ics.npz"
    if ics_npz.exists():
        ics = _load_camels_ics_npz(ics_npz)
    else:
        ics = _load_camels_ics_hdf5(base_dir / "ICs")
    params = _parse_two_lpt_param(base_dir / "ICs" / "2LPT.param")

    ic_ids = np.asarray(ics["ids"], dtype=np.int64)
    final_ids = np.asarray(snapshot["ids"], dtype=np.int64)
    ic_ids = ic_ids - ic_ids.min()
    final_ids = final_ids - final_ids.min()

    ic_order = np.argsort(ic_ids)
    final_order = np.argsort(final_ids)
    sorted_ic_ids = ic_ids[ic_order]
    sorted_final_ids = final_ids[final_order]
    if not np.array_equal(sorted_ic_ids, sorted_final_ids):
        raise ValueError("CAMELS IC and final snapshot particle IDs do not match.")

    match_pos = np.searchsorted(sorted_final_ids, ic_ids)
    final_match = final_order[match_pos]

    grid_size = _infer_grid_size(ic_ids.shape[0])
    omega_b = float(params.get("OmegaBaryon", PLANCK18_OMEGA_B))
    if omega_b <= 0:
        omega_b = PLANCK18_OMEGA_B

    metadata = CamelsMetadata(
        box_size=float(snapshot["BoxSize"]),
        omega_m=float(params.get("Omega", snapshot["Omega_m"])),
        omega_l=float(params.get("OmegaLambda", snapshot["Omega_l"])),
        omega_b=omega_b,
        h=float(params.get("HubbleParam", 0.7)),
        sigma8=float(params.get("Sigma8", 0.8)),
        n_s=float(params.get("PrimordialIndex", 0.96)),
        a_start=1.0 / (1.0 + float(params.get("Redshift", 127.0))),
        redshift=float(snapshot["redshift"]),
        grid_size=grid_size,
    )

    return CamelsParticlePair(
        # Preserve the notebook's raw IC ordering, then align the final snapshot
        # to that same ID order for paired supervision.
        ic_pos=np.asarray(ics["pos"], dtype=np.float32),
        ic_vel=np.asarray(ics["vel"], dtype=np.float32),
        final_pos=np.asarray(snapshot["pos"][final_match], dtype=np.float32),
        final_vel=np.asarray(snapshot["vel"][final_match], dtype=np.float32),
        ids=ic_ids,
        metadata=metadata,
    )


def _grid_linear_index_from_pos(pos, box_size, grid_size):
    """Map regular-grid IC positions back to their Lagrangian linear index."""
    spacing = box_size / grid_size
    grid = np.rint(pos / spacing).astype(np.int64) % grid_size
    return grid[:, 2] + grid_size * (grid[:, 1] + grid_size * grid[:, 0])


def _reshape_on_lagrangian_grid(pair: CamelsParticlePair):
    """Reshape paired particle arrays onto the Lagrangian grid."""
    n = pair.metadata.grid_size
    count = pair.ids.shape[0]
    linear_index = np.asarray(pair.ids, dtype=np.int64)
    order = np.argsort(linear_index)
    if not np.array_equal(linear_index[order], np.arange(count, dtype=np.int64)):
        linear_index = _grid_linear_index_from_pos(pair.ic_pos, pair.metadata.box_size, n)
        order = np.argsort(linear_index)
        if not np.array_equal(linear_index[order], np.arange(count, dtype=np.int64)):
            raise ValueError("IC positions do not map cleanly back to a regular Lagrangian grid.")

    reshape = (n, n, n, 3)
    return (
        pair.ic_pos[order].reshape(reshape),
        pair.ic_vel[order].reshape(reshape),
        pair.final_pos[order].reshape(reshape),
        pair.final_vel[order].reshape(reshape),
    )


def _block_mean(array, factor):
    """Average a vector grid over cubic blocks."""
    n = array.shape[0]
    coarse_n = n // factor
    reshaped = array.reshape(coarse_n, factor, coarse_n, factor, coarse_n, factor, array.shape[-1])
    return reshaped.mean(axis=(1, 3, 5))


def coarsen_camels_pair(pair: CamelsParticlePair, factor: int):
    """Downsample a CAMELS pair by averaging displacements and velocities.

    Parameters
    ----------
    pair : CamelsParticlePair
        Input paired CAMELS dataset.
    factor : int
        Integer coarsening factor per spatial axis.

    Returns
    -------
    CamelsParticlePair
        Coarsened particle pair on the lower-resolution Lagrangian grid.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return pair

    n = pair.metadata.grid_size
    if n % factor != 0:
        raise ValueError(f"Grid size {n} is not divisible by downsample factor {factor}.")

    ic_pos_grid, ic_vel_grid, final_pos_grid, final_vel_grid = _reshape_on_lagrangian_grid(pair)
    fine_spacing = pair.metadata.box_size / n
    coarse_n = n // factor
    coarse_spacing = pair.metadata.box_size / coarse_n

    fine_index = np.indices((n, n, n), dtype=np.float32)
    fine_anchor = np.moveaxis(fine_index, 0, -1) * fine_spacing
    coarse_index = np.indices((coarse_n, coarse_n, coarse_n), dtype=np.float32)
    coarse_anchor = np.moveaxis(coarse_index, 0, -1) * coarse_spacing

    ic_disp_grid = periodic_delta(ic_pos_grid, fine_anchor, pair.metadata.box_size)
    coarse_ic_disp = _block_mean(ic_disp_grid, factor)
    coarse_ic_vel = _block_mean(ic_vel_grid, factor)
    coarse_ic_pos = periodic_wrap(coarse_anchor + coarse_ic_disp, pair.metadata.box_size)

    final_disp_grid = periodic_delta(final_pos_grid, fine_anchor, pair.metadata.box_size)
    coarse_final_disp = _block_mean(final_disp_grid, factor)
    coarse_final_pos = periodic_wrap(coarse_anchor + coarse_final_disp, pair.metadata.box_size)
    coarse_final_vel = _block_mean(final_vel_grid, factor)

    coarse_ids = np.arange(coarse_n ** 3, dtype=np.int64)
    metadata = replace(pair.metadata, grid_size=coarse_n)
    return CamelsParticlePair(
        ic_pos=coarse_ic_pos.reshape((-1, 3)).astype(np.float32),
        ic_vel=coarse_ic_vel.reshape((-1, 3)).astype(np.float32),
        final_pos=coarse_final_pos.reshape((-1, 3)).astype(np.float32),
        final_vel=coarse_final_vel.reshape((-1, 3)).astype(np.float32),
        ids=coarse_ids,
        metadata=metadata,
    )


def velocity_kms_to_canonical(vel_kms, conf, extra_scale=1.0):
    """Convert km/s velocities into PM++ canonical velocity units.

    Parameters
    ----------
    vel_kms : array-like
        Velocities in km/s.
    conf : Configuration
        Configuration defining PM++ code units.
    extra_scale : float, optional
        Additional multiplicative scale factor applied before conversion.

    Returns
    -------
    numpy.ndarray
        Velocities in PM++ canonical units ``[H_0 L]``.
    """
    km_per_second_per_code_unit = float(conf.V) / 1000.0
    return np.asarray(vel_kms, dtype=np.float32) / km_per_second_per_code_unit * extra_scale
