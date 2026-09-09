"""Reverse migration must preserve cotangent identity despite float32 rounding.

Set PMPP_TEST_DEVICES=4/8 with logical CPU devices to exercise wider rings;
the native variant requires physical GPUs and the routing extension.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh


@pytest.mark.parametrize("native", [False, True])
def test_reconstructed_cotangents_follow_particle_ids_at_roundoff_boundary(native):
    count = int(os.environ.get("PMPP_TEST_DEVICES", "2"))
    devices = jax.devices()[:count]
    if len(devices) != count or count < 2:
        pytest.skip(f"requires {count} devices")
    if native and devices[0].platform != "gpu":
        pytest.skip("native routing requires GPUs")
    mesh = create_compute_mesh(devices)
    n, capacity = count * 4, 12
    conf = Configuration(
        1., (n, ) * 3, mesh_shape=1, float_dtype=jnp.float32, pmid_dtype=jnp.int16,
        pallas_cic=False, multigpu=MultiGPUConfiguration(
            compute_mesh=mesh, mode="mesh_halo", cuda_routing=native, cuda_routing_backend="bidir_mergepath",
        ), max_ptcl_per_slice=capacity, max_share_ptcl=6, max_halo_share_ptcl=6, max_share_gather_ptcl=6,
    )
    pmid = np.zeros((count, capacity, 3), dtype=np.int16)
    disp = np.zeros((count, capacity, 3), dtype=np.float32)
    vel = np.zeros_like(disp)
    unused = np.ones((count, capacity), dtype=bool)
    for gpu in range(count):
        rows = [((gpu * 4 + i, 1, 1), 0., (.25, 0., -2.25)[i]) for i in range(3)]
        if gpu == count - 1:
            # This tiny negative displacement is lost when subtracting 1 in
            # float32. Replaying the drift moves this ID to GPU 0 and shifts
            # sorted slots, even though the saved post-state owns it here.
            rows.append(((0, 0, 0), -2.**-27, 1.))
        for row, (identity, d, v) in enumerate(sorted(rows)):
            pmid[gpu, row] = identity
            disp[gpu, row, 0] = d
            vel[gpu, row, 0] = v
            unused[gpu, row] = False
    keys = (pmid[..., 0].astype(np.int32) * n + pmid[..., 1]) * n + pmid[..., 2]
    cot = np.stack((keys + 1, 2 * keys + 3, -keys - 5), axis=-1).astype(np.float32)
    cot[unused] = 12345  # Padding must never leak into an active cotangent.

    def put(x):
        x = x.reshape((count * capacity,) + x.shape[2:])
        return jax.device_put(x, NamedSharding(mesh, P("gpus")))

    p, d, v, u, g = [put(x) for x in (pmid, disp, vel, unused, cot)]
    result = jax.jit(conf.mGPU_reconstruct_pre_drift_pullback)(
        p, d, v, jnp.zeros_like(v), u, jnp.float32(1), g, 2 * g, -3 * g,
    )
    result = jax.device_get(result)
    active = ~result[4]
    actual_ids = result[0][active].astype(np.int32)
    actual_keys = (actual_ids[:, 0] * n + actual_ids[:, 1]) * n + actual_ids[:, 2]
    np.testing.assert_array_equal(np.sort(actual_keys), np.sort(keys[~unused]))
    state_by_key = {int(key): (d - v, v) for key, d, v in zip(keys[~unused], disp[~unused], vel[~unused])}
    np.testing.assert_array_equal(result[1][active], np.stack([state_by_key[int(key)][0] for key in actual_keys]))
    np.testing.assert_array_equal(result[2][active], np.stack([state_by_key[int(key)][1] for key in actual_keys]))
    np.testing.assert_array_equal(result[3], 0)
    expected = np.stack((actual_keys + 1, 2 * actual_keys + 3, -actual_keys - 5), axis=-1)
    for actual, scale in zip(result[6:], (1, 2, -3)):
        np.testing.assert_array_equal(actual[active], scale * expected)
        np.testing.assert_array_equal(actual[~active], 0)
