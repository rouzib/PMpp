"""Device-level contract tests for the optional bidirectional routing ABI."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pmpp.distributed import cuda as cuda_routing


def _bidir_ready():
    try:
        cuda_routing._register_targets()
        return bool(jax.devices("gpu")) and bool(cuda_routing.extension_status().get("bidir_registered"))
    except Exception:
        return False


@pytest.mark.skipif(not _bidir_ready(), reason="requires the rebuilt CUDA bidirectional routing library")
@pytest.mark.parametrize("float_dtype", (jnp.float32, jnp.float64))
def test_native_bidir_pack_and_three_stream_merge_contract(float_dtype):
    if float_dtype == jnp.float64 and not cuda_routing.extension_status().get("float64_bidir_registered"):
        pytest.skip("loaded CUDA routing library has no float64 bidirectional ABI")
    pmid = jnp.asarray([[index, 0, 0] for index in range(12)], dtype=jnp.int32)
    disp = jnp.arange(36, dtype=float_dtype).reshape(12, 3)
    vel = disp + 100.0
    valid = jnp.ones((12, ), dtype=jnp.uint8)
    x_mod = jnp.arange(12, dtype=float_dtype) + jnp.asarray(0.1, dtype=float_dtype)

    def fn(pmid, disp, vel, valid, x_mod):
        packed = cuda_routing.route_pack_bidir_cuda(
            pmid, disp, vel, valid, x_mod, global_nmesh=12, mesh_shape=(12, 1, 1), owned_start=3, owned_end=9,
            slice_width=3, num_devices=4, capacity=12,
        )
        merged = cuda_routing.route_merge_bidir_cuda(
            pmid, disp, vel, packed[5], packed[6], packed[7], packed[0], packed[2], packed[1], packed[3],
            mesh_shape=(12, 1, 1), capacity=12,
        )
        return packed, merged

    packed, merged = jax.jit(fn)(pmid, disp, vel, valid, x_mod)
    np.testing.assert_array_equal(np.asarray(packed[2]), np.array(3, np.int32))
    np.testing.assert_array_equal(np.asarray(packed[3]), np.array(3, np.int32))
    np.testing.assert_array_equal(np.asarray(packed[7]), np.array(6, np.int32))
    np.testing.assert_array_equal(np.asarray(packed[4]), np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3], np.uint8), )
    np.testing.assert_array_equal(np.asarray(merged[-1]), np.array(12, np.int32))
    np.testing.assert_array_equal(np.asarray(merged[3]), np.ones((12, ), np.uint8))
    np.testing.assert_array_equal(np.asarray(merged[4]), np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2], np.uint8), )
    np.testing.assert_array_equal(np.asarray(merged[0])[:, 0], np.arange(12, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(merged[1]), np.asarray(disp))
    np.testing.assert_array_equal(np.asarray(merged[2]), np.asarray(vel))
