"""Device-level contract tests for the optional bidirectional routing ABI."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pmpp.distributed import cuda as cuda_routing
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh


def _bidir_ready():
    try:
        cuda_routing._register_targets()
        return bool(jax.devices("gpu")) and bool(cuda_routing.extension_status().get("bidir_registered"))
    except Exception:
        return False


def _fused_primal_ready():
    return _bidir_ready() and bool(cuda_routing.extension_status().get("fused_primal_registered"))


@pytest.mark.skipif(not _bidir_ready(), reason="requires the rebuilt CUDA bidirectional routing library")
@pytest.mark.parametrize("float_dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("pmid_dtype", (jnp.int16, jnp.int32))
def test_native_bidir_pack_and_three_stream_merge_contract(float_dtype, pmid_dtype):
    if float_dtype == jnp.float64 and not cuda_routing.extension_status().get("float64_bidir_registered"):
        pytest.skip("loaded CUDA routing library has no float64 bidirectional ABI")
    pmid = jnp.asarray([[index, 0, 0] for index in range(12)], dtype=pmid_dtype)
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
            pmid, disp, vel, packed[5], packed[6], packed[0], packed[2], packed[1], packed[3], mesh_shape=(12, 1, 1),
            capacity=12,
        )
        return packed, merged

    packed, merged = jax.jit(fn)(pmid, disp, vel, valid, x_mod)
    np.testing.assert_array_equal(np.asarray(packed[2]), np.array(3, np.int32))
    np.testing.assert_array_equal(np.asarray(packed[3]), np.array(3, np.int32))
    np.testing.assert_array_equal(np.asarray(packed[6]), np.array(6, np.int32))
    np.testing.assert_array_equal(np.asarray(packed[4]), np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3], np.uint8), )
    np.testing.assert_array_equal(np.asarray(merged[-1]), np.array(12, np.int32))
    np.testing.assert_array_equal(np.asarray(merged[3]), np.ones((12, ), np.uint8))
    np.testing.assert_array_equal(np.asarray(merged[4]), np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2], np.uint8), )
    np.testing.assert_array_equal(np.asarray(merged[0])[:, 0], np.arange(12, dtype=np.int32))
    assert merged[0].dtype == pmid.dtype
    np.testing.assert_array_equal(np.asarray(merged[1]), np.asarray(disp))
    np.testing.assert_array_equal(np.asarray(merged[2]), np.asarray(vel))


@pytest.mark.skipif(not _bidir_ready(), reason="requires the rebuilt CUDA bidirectional routing library")
def test_native_backends_preserve_keys_across_u32_boundary():
    pmid = jnp.asarray([[1023, 2047, 2047], [1024, 0, 0], [2047, 2047, 2047]], dtype=jnp.int16)
    disp = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    vel = disp + 10.0
    valid = jnp.ones((3, ), dtype=jnp.uint8)
    x_mod = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
    expected_keys = np.asarray([[0xffffffff, 0], [0, 1], [0xffffffff, 1]], dtype=np.uint32)

    packed_current = cuda_routing.route_pack(
        pmid, disp, vel, valid, x_mod, global_nmesh=2048, mesh_shape=(2048, 2048, 2048), owned_start=1024, owned_end=0,
        slice_width=1024, direction=-1, num_devices=2, capacity=3,
    )
    empty_pmid = jnp.zeros_like(pmid)
    empty_valid = jnp.zeros((3, ), dtype=jnp.uint8)
    merged_current = cuda_routing.route_merge(
        empty_pmid, jnp.zeros_like(disp), jnp.zeros_like(vel), empty_valid, packed_current[0], packed_current[1],
        mesh_shape=(2048, 2048, 2048), capacity=3,
    )

    packed_bidir = cuda_routing.route_pack_bidir_cuda(
        pmid, disp, vel, valid, x_mod, global_nmesh=2048, mesh_shape=(2048, 2048, 2048), owned_start=1024, owned_end=0,
        slice_width=1024, num_devices=2, capacity=3,
    )
    empty_records = jnp.zeros_like(packed_bidir[0])
    merged_bidir = cuda_routing.route_merge_bidir_cuda(
        empty_pmid, jnp.zeros_like(disp), jnp.zeros_like(vel), packed_bidir[5], packed_bidir[6], packed_bidir[0],
        packed_bidir[2], empty_records, jnp.int32(0), mesh_shape=(2048, 2048, 2048), capacity=3,
    )
    merged_primal = cuda_routing.route_merge_bidir_primal_i16(
        empty_pmid, jnp.zeros_like(disp), jnp.zeros_like(vel), packed_bidir[5], packed_bidir[6], packed_bidir[0],
        packed_bidir[2], empty_records, jnp.int32(0), mesh_shape=(2048, 2048, 2048), capacity=3,
    )

    np.testing.assert_array_equal(np.asarray(packed_current[0])[:, :2], expected_keys)
    np.testing.assert_array_equal(np.asarray(merged_current[0]), np.asarray(pmid, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(merged_bidir[0]), np.asarray(pmid))
    np.testing.assert_array_equal(np.asarray(merged_bidir[6]), expected_keys)
    np.testing.assert_array_equal(np.asarray(merged_primal[0]), np.asarray(pmid))
    np.testing.assert_array_equal(np.asarray(merged_primal[1]), np.asarray(disp))
    np.testing.assert_array_equal(np.asarray(merged_primal[2]), np.asarray(vel))


@pytest.mark.skipif(not _fused_primal_ready(), reason="requires the fused-drift primal CUDA routing ABI")
def test_fused_drift_primal_pack_and_merge_avoid_particle_sized_metadata():
    pmid = jnp.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=jnp.int16)
    disp = jnp.zeros((4, 3), dtype=jnp.float32)
    vel = jnp.asarray([[0, 0, 0], [-2, 0, 0], [600, 0, 0], [1200, 0, 0]], dtype=jnp.float32)
    valid = jnp.ones((4, ), dtype=jnp.bool_)

    def fn(pmid, disp, vel, valid, factor):
        packed = cuda_routing.route_pack_bidir_drift_primal_i16(
            pmid, disp, vel, valid, factor, disp_size=1.0, global_nmesh=2048, mesh_shape=(2048, 2048, 2048),
            owned_start=jnp.int32(0), owned_end=jnp.int32(512), slice_width=512, num_devices=4, capacity=4,
        )
        merged = cuda_routing.route_merge_bidir_drift_primal_i16(
            pmid, disp, vel, valid, factor, packed[4], packed[5], packed[0], packed[2], packed[1], packed[3],
            disp_size=1.0, global_nmesh=2048, mesh_shape=(2048, 2048, 2048), owned_start=jnp.int32(0),
            owned_end=jnp.int32(512), slice_width=512, num_devices=4, record_capacity=4, capacity=4,
        )
        return packed, merged

    compiled = jax.jit(fn)
    packed, merged = compiled(pmid, disp, vel, valid, jnp.float32(1))
    assert len(packed) == 7
    assert packed[4].shape == (1, )
    np.testing.assert_array_equal(np.asarray(packed[2:4]), np.asarray([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(packed[4]), np.asarray([1], dtype=np.uint32))
    assert int(np.asarray(packed[5])) == 1
    assert int(np.asarray(packed[6])) == 1
    assert len(merged) == 5
    assert merged[0].dtype == jnp.int16 and merged[3].dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(merged[0][:3, 0]), np.asarray([0, 1, 2], dtype=np.int16))
    np.testing.assert_allclose(np.asarray(merged[1][:3, 0]), np.asarray([0, -2, 600], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(merged[3]), np.asarray([True, True, True, False]))
    assert int(np.asarray(merged[4])) == 3
    _, merged_factor_zero = compiled(pmid, vel, vel, valid, jnp.float32(0))
    np.testing.assert_array_equal(np.asarray(merged_factor_zero[0]), np.asarray(merged[0]))
    np.testing.assert_array_equal(np.asarray(merged_factor_zero[1]), np.asarray(merged[1]))


@pytest.mark.skipif(not _fused_primal_ready(), reason="requires the fused-primal CUDA routing ABI")
def test_native_offset_arithmetic_crosses_signed_int32_without_allocation():
    row = np.iinfo(np.int32).max // 3
    limbs = np.asarray(cuda_routing.route_offset_probe(row, component=2, record_words=14))
    vector_offset = int(limbs[0]) | (int(limbs[1]) << 32)
    record_offset = int(limbs[2]) | (int(limbs[3]) << 32)
    assert vector_offset == row * 3 + 2
    assert vector_offset > np.iinfo(np.int32).max
    assert record_offset == row * 14
    assert record_offset > np.iinfo(np.uint32).max


@pytest.mark.skipif(
    not _fused_primal_ready() or len(jax.devices("gpu")) < 2,
    reason="requires two GPUs and the fused-primal CUDA routing ABI",
)
def test_fused_low_memory_shard_mover_replicates_diagnostics_and_routes_drift(monkeypatch):
    # The local validation environment predates the declared JAX qualification
    # floor but supports the typed FFI exercised by this device-level test.
    monkeypatch.setattr(cuda_routing, "_qualified_jax", lambda: True)
    conf = Configuration(
        1.0, (8, 1, 1), mesh_shape=1, pmid_dtype=jnp.int16, float_dtype=jnp.float32, pallas_cic=False,
        multigpu=MultiGPUConfiguration(
            compute_mesh=create_compute_mesh(jax.devices("gpu")[:2]), mode="mesh_halo", cuda_routing=True,
        ), max_ptcl_per_slice=8, max_share_ptcl=4, max_halo_share_ptcl=4, max_share_gather_ptcl=4,
    )
    assert cuda_routing._FUSED_PRIMAL_REGISTERED
    assert cuda_routing.supported_configuration(conf)
    assert cuda_routing.supported_bidir_configuration(conf)
    assert cuda_routing.requested_backend(conf) == "bidir_mergepath"
    assert jnp.dtype(conf.float_dtype) == jnp.dtype(jnp.float32)
    assert jnp.dtype(conf.pmid_dtype) == jnp.dtype(jnp.int16)
    assert cuda_routing.supported_fused_primal_configuration(conf)
    mover = conf.mGPU_halo_moving_low_memory
    assert mover is not None
    local_pmid = []
    local_disp = []
    local_vel = []
    local_unused = []
    for start in (0, 4):
        ids = np.zeros((8, 3), dtype=np.int16)
        ids[:4, 0] = np.arange(start, start + 4, dtype=np.int16)
        velocity = np.zeros((8, 3), dtype=np.float32)
        if start == 0:
            velocity[3, 0] = 2.0
        else:
            velocity[0, 0] = -2.0
        local_pmid.append(ids)
        local_disp.append(np.zeros((8, 3), dtype=np.float32))
        local_vel.append(velocity)
        local_unused.append(np.asarray([False] * 4 + [True] * 4))
    pmid = jnp.asarray(np.concatenate(local_pmid, axis=0))
    disp = jnp.asarray(np.concatenate(local_disp, axis=0))
    vel = jnp.asarray(np.concatenate(local_vel, axis=0))
    unused = jnp.asarray(np.concatenate(local_unused, axis=0))

    result = jax.jit(mover)(pmid, disp, vel, jnp.float32(1), unused)
    result = tuple(value.block_until_ready() for value in result)
    assert not bool(np.asarray(result[5]))
    assert int(np.asarray(result[6])) == 1
    assert int(np.asarray(result[7])) == 0
    np.testing.assert_array_equal(np.asarray(result[0])[:4, 0], np.asarray([0, 1, 2, 4], dtype=np.int16))
    np.testing.assert_array_equal(np.asarray(result[0])[8:12, 0], np.asarray([3, 5, 6, 7], dtype=np.int16))
    np.testing.assert_allclose(np.asarray(result[1])[:4, 0], np.asarray([0, 0, 0, -2], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(result[1])[8:12, 0], np.asarray([2, 0, 0, 0], dtype=np.float32))

    standard = jax.jit(conf.mGPU_halo_moving_no_acc
                       )(pmid, disp, disp + vel, vel, conf.halo_start, conf.halo_end, unused,
                         )
    standard = tuple(value.block_until_ready() for value in standard)
    np.testing.assert_array_equal(np.asarray(standard[0]), np.asarray(result[0]))
    np.testing.assert_array_equal(np.asarray(standard[1]), np.asarray(result[1]))
    np.testing.assert_array_equal(np.asarray(standard[2]), np.asarray(result[2]))
    np.testing.assert_array_equal(np.asarray(standard[4]), np.asarray(result[4]))
