"""Tests for the optional CUDA routing boundary and portable fallback."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from pmpp.core import Configuration
from pmpp.distributed import (
    MultiGPUConfiguration, build_multigpu_configuration, create_compute_mesh, cuda as cuda_routing, extension_status,
    supported_configuration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_routing_is_not_required_by_a_cpu_configuration():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1)
    assert supported_configuration(conf) is False
    assert extension_status()["backend"] == jax.default_backend()


def test_explicit_cuda_request_still_resolves_to_portable_fallback_without_qualification():
    mesh = create_compute_mesh(jax.devices()[:1])
    conf = Configuration(
        1.0, (4, 4, 4), mesh_shape=1,
        multigpu=MultiGPUConfiguration(compute_mesh=mesh, mode="mesh_halo", cuda_routing=True,
                                       ),
    )
    assert conf.cuda_routing is False
    assert conf.multigpu.cuda_routing is False


def test_cuda_record_and_build_contract_is_maintained():
    source = (ROOT / "cuda" / "route_kernels.cu").read_text(encoding="utf-8")
    cmake = (ROOT / "cuda" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "constexpr int kRecordWordsF32 = 8" in source
    assert "constexpr int kRecordWordsF64 = 14" in source
    assert "pmpp_route_pack_f64" in source
    assert "pmpp_route_merge_f64" in source
    assert "pmpp_route_merge_aux_f64" in source
    assert "pmpp_route_transpose_split_f64" in source
    assert "pmpp_route_transpose_scatter_f64" in source
    assert "pmpp_route_merge_aux" in source
    assert "pmpp_route_bidir_pack" in source
    assert "pmpp_route_merge_bidir" in source
    assert "MergePathBidirKernel" in source
    assert "LowerBoundRecordKeys" in source
    assert "UpperBoundRecordKeys" in source
    assert "LowerBoundKeys(left_records" not in source
    assert "LowerBoundKeys(right_records" not in source
    assert "UpperBoundKeys(left_records" not in source
    assert "cudaDeviceSynchronize(" not in source
    assert "cudaMalloc(" not in source
    assert "CUDA_ARCHITECTURES \"80;86;86-virtual\"" in cmake
    assert "--use_fast_math" not in cmake

    bidir_start = source.index("pmpp_route_bidir_pack")
    bidir_end = source.index("pmpp_route_merge,", bidir_start)
    bidir_binding = source[bidir_start:bidir_end]
    assert 'Attr<int32_t>("owned_start")' not in bidir_binding
    assert 'Attr<int32_t>("owned_end")' not in bidir_binding
    assert bidir_binding.count(".Arg<ffi::Buffer") == 7


def _gpu_extension_ready():
    try:
        return len(jax.devices("gpu")) >= 2 and cuda_routing._register_targets()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_extension_ready(), reason="requires two GPUs and the optional CUDA FFI library")
@pytest.mark.parametrize(("float_dtype", "record_words"), ((jnp.float32, 8), (jnp.float64, 14)), )
def test_cuda_route_merge_aux_and_overflow_are_device_safe(float_dtype, record_words):
    if float_dtype == jnp.float64 and not cuda_routing.extension_status().get("float64_registered"):
        pytest.skip("loaded CUDA routing library has no float64 ABI")
    devices = jax.devices("gpu")[:2]
    mesh = jax.sharding.Mesh(np.asarray(devices), ("i", ))
    perm = ((0, 1), (1, 0))

    pmid = jnp.stack((jnp.arange(8, dtype=jnp.int32), jnp.zeros(8, jnp.int32), jnp.zeros(8, jnp.int32), ), axis=1, )
    disp = jnp.zeros((8, 3), float_dtype).at[3, 0].set(2).at[5, 0].set(-2)
    vel = jnp.arange(24, dtype=float_dtype).reshape(8, 3)
    valid = jnp.ones((8, ), jnp.uint8)

    cot = jnp.arange(24, dtype=float_dtype).reshape(8, 3)

    def local(pmid, disp, vel, valid, cot):
        axis = jax.lax.axis_index("i")
        start = axis * 4
        end = start + 4
        x_mod = (pmid[:, 0].astype(float_dtype) + disp[:, 0]) % 8
        records, count, classes = cuda_routing.route_pack(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 1, 1), owned_start=start, owned_end=end,
            slice_width=4, direction=-1, num_devices=2, capacity=4,
        )
        incoming = jax.lax.ppermute(records, axis_name="i", perm=perm)
        incoming_count = jax.lax.ppermute(count, axis_name="i", perm=perm)
        merged = cuda_routing.route_merge(
            pmid, disp, vel, classes == 1, incoming, incoming_count, mesh_shape=(8, 1, 1), capacity=4, auxiliary=True,
        )
        stay_pos = jnp.compress(
            classes == 1, jnp.arange(4, dtype=jnp.int32), axis=0, size=4, fill_value=jnp.int32(-1),
        )
        stay_valid = jnp.arange(4) < jnp.sum(classes == 1)
        send_left_pos = jnp.compress(
            classes == 2, jnp.arange(4, dtype=jnp.int32), axis=0, size=4, fill_value=jnp.int32(-1),
        )
        send_left_valid = jnp.arange(4) < jnp.sum(classes == 2)
        stay_cot, incoming_left_cot, incoming_right_cot = cuda_routing.route_transpose_split(
            cot, jnp.where(merged[-2] == 1, jnp.uint8(2), merged[-2]), merged[-1], auth_size=4, share_capacity=4,
        )
        scattered = cuda_routing.route_transpose_scatter(
            stay_cot, jax.lax.ppermute(incoming_right_cot, axis_name="i", perm=perm),
            jnp.zeros_like(incoming_right_cot), stay_pos, stay_valid, send_left_pos, send_left_valid,
            jnp.full((4, ), -1, jnp.int32), jnp.zeros((4, ), jnp.uint8), auth_size=4, share_capacity=4,
        )
        return records, count.reshape((1, )), classes, *merged, scattered

    fn = jax.jit(
        shard_map(
            local, mesh=mesh, in_specs=(P("i"), P("i"), P("i"), P("i"), P("i")), out_specs=(P("i"), ) * 10,
            check_rep=False,
        )
    )
    result = fn(pmid, disp, vel, valid, cot)
    assert result[0].shape[-1] == record_words
    np.testing.assert_array_equal(np.asarray(result[1]), np.array([1, 1], np.int32))
    np.testing.assert_array_equal(np.asarray(result[3])[:, 0], np.array([0, 1, 2, 5, 3, 4, 6, 7], np.int32))
    # Auxiliary tags identify the virtual stay stream (0) versus the received
    # record (1), while source indices remain local to their input stream.
    np.testing.assert_array_equal(np.asarray(result[7]), np.array([0, 0, 0, 1, 1, 0, 0, 0], np.uint8), )
    np.testing.assert_array_equal(np.asarray(result[8]), np.array([0, 1, 2, 0, 0, 0, 1, 2], np.int32), )
    expected_transpose = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [12, 13, 14], [15, 16, 17], [9, 10, 11], [18, 19, 20], [21, 22, 23], ],
        dtype=np.dtype(float_dtype),
    )
    np.testing.assert_array_equal(np.asarray(result[9]), expected_transpose)

    # Padded auxiliary-merge rows also enter the route transpose. Their
    # provenance must be -1, never a valid compact-stay source index.
    def aux_padding_indices(pmid, disp, vel):
        return cuda_routing.route_merge(
            pmid, disp, vel, jnp.array([1, 0, 0, 0], dtype=jnp.uint8), jnp.zeros((4, record_words), dtype=jnp.uint32),
            jnp.array(0, dtype=jnp.int32), mesh_shape=(8, 1, 1), capacity=4, auxiliary=True,
        )

    padded = jax.jit(aux_padding_indices)(pmid[:4], disp[:4], vel[:4])
    source_index = np.asarray(padded[-1])
    out_valid = np.asarray(padded[-3])
    assert np.all(source_index[out_valid == 0] == -1)

    def overflow_local(pmid, disp, vel, valid):
        axis = jax.lax.axis_index("i")
        start = axis * 4
        records, count, classes = cuda_routing.route_pack(
            pmid, disp, vel, valid, (pmid[:, 0].astype(float_dtype) + 4) % 8, global_nmesh=8, mesh_shape=(8, 1, 1),
            owned_start=start, owned_end=start + 4, slice_width=4, direction=-1, num_devices=2, capacity=2,
        )
        return records, count.reshape((1, )), classes

    overflow = jax.jit(
        shard_map(
            overflow_local, mesh=mesh, in_specs=(P("i"), P("i"), P("i"), P("i")), out_specs=(P("i"), P("i"), P("i")),
            check_rep=False,
        )
    )(pmid, disp, vel, valid)
    # Every valid row is routed in this synthetic case.  The device count is
    # intentionally larger than the output capacity, while the fixed record
    # buffer remains bounded and readable.
    np.testing.assert_array_equal(np.asarray(overflow[1]), np.array([4, 4], np.int32))


@pytest.mark.skipif(not _gpu_extension_ready(), reason="requires the optional CUDA FFI library")
def test_float64_classification_preserves_sub_float32_boundary_offsets():
    if not cuda_routing.extension_status().get("float64_registered"):
        pytest.skip("loaded CUDA routing library has no float64 ABI")

    boundary = jnp.float64(4.0)
    x_mod = jnp.asarray(
        [jnp.nextafter(boundary, jnp.float64(0.0)), boundary,
         jnp.nextafter(boundary, jnp.float64(8.0)), ], dtype=jnp.float64,
    )
    pmid = jnp.zeros((3, 3), dtype=jnp.int32)
    disp = jnp.zeros((3, 3), dtype=jnp.float64)
    vel = jnp.zeros((3, 3), dtype=jnp.float64)
    valid = jnp.ones((3, ), dtype=jnp.uint8)

    def classify_current(pmid, disp, vel, valid, x_mod):
        return cuda_routing.route_pack(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 1, 1), owned_start=0, owned_end=4,
            slice_width=4, direction=-1, num_devices=2, capacity=3,
        )[-1]

    def classify_bidir(pmid, disp, vel, valid, x_mod):
        return cuda_routing.route_pack_bidir_cuda(
            pmid, disp, vel, valid, x_mod, global_nmesh=8, mesh_shape=(8, 1, 1), owned_start=0, owned_end=4,
            slice_width=4, num_devices=2, capacity=3,
        )[4]

    expected = np.asarray([1, 2, 2], dtype=np.uint8)
    np.testing.assert_array_equal(np.asarray(jax.jit(classify_current)(pmid, disp, vel, valid, x_mod)), expected, )
    np.testing.assert_array_equal(np.asarray(jax.jit(classify_bidir)(pmid, disp, vel, valid, x_mod)), expected, )
