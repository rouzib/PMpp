import sys
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmpp.configuration import Configuration
from pmpp.modes import white_noise_nested
from pmpp.utils import create_compute_mesh

try:
    import pytest
except ImportError:
    pytest = None


def _init_conf(num_ptcl, devices):
    box_size = 100.0
    return Configuration(
        ptcl_spacing=box_size / num_ptcl,
        ptcl_grid_shape=(num_ptcl,) * 3,
        mesh_shape=1,
        compute_mesh=create_compute_mesh(devices),
        max_ptcl_per_slice=int(num_ptcl**3 / max(1, len(devices)) * 1.6),
        max_share_ptcl=4000,
        max_share_gather_ptcl=12000,
        a_start=1 / 60,
        a_stop=1 / 30,
        a_nbody_maxstep=1 / 60,
    )


def _shared_mode_index_map(coarse_size, fine_size):
    coarse_idx = np.arange(coarse_size, dtype=np.int32)
    signed = np.where(coarse_idx < (coarse_size + 1) // 2, coarse_idx, coarse_idx - coarse_size)
    if coarse_size % 2 == 0:
        keep = signed != -(coarse_size // 2)
        coarse_idx = coarse_idx[keep]
        signed = signed[keep]
    fine_idx = np.where(signed >= 0, signed, signed + fine_size)
    return coarse_idx.astype(np.int32), fine_idx.astype(np.int32)


def _extract_shared_block(fine_modes, coarse_shape, fine_shape):
    x_idx, x_map = _shared_mode_index_map(coarse_shape[0], fine_shape[0])
    y_idx, y_map = _shared_mode_index_map(coarse_shape[1], fine_shape[1])
    z_map = np.arange(coarse_shape[2] // 2 + 1, dtype=np.int32)
    if coarse_shape[2] % 2 == 0:
        z_map = z_map[:-1]
    return fine_modes[np.ix_(x_map, y_map, z_map)], x_idx, y_idx, z_map


def test_nested_white_noise_matches_shared_low_k_modes():
    devices = jax.devices()[: min(2, len(jax.devices()))]
    if not devices:
        if pytest is not None:
            pytest.skip("nested white noise test requires at least one JAX device")
        raise SystemExit("nested white noise test requires at least one JAX device")

    conf32 = _init_conf(32, devices)
    conf64 = _init_conf(64, devices)

    modes32 = np.asarray(jax.device_get(white_noise_nested(7, conf32)))
    modes64 = np.asarray(jax.device_get(white_noise_nested(7, conf64)))
    shared64, x_idx, y_idx, z_idx = _extract_shared_block(modes64, conf32.ptcl_grid_shape, conf64.ptcl_grid_shape)
    shared32 = modes32[np.ix_(x_idx, y_idx, z_idx)]

    np.testing.assert_array_equal(shared32, shared64)
