import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

import pmpp.distributed.routing as routing
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.distributed.routing import (
    _halo_capacity, _mask_compact_prefix, _pack_authoritative_only, _pack_authoritative_only_no_acc,
    _reverse_build_owned_only_cot, _scatter_compact_to_dense, compute_halo_mask_shard_map,
)
from pmpp.nbody import Particles


def _periodic_interval(value, start, end):
    return jnp.where(start > end, (value >= start) | (value < end), (value >= start) & (value < end))


def _fake_route_classes(x_mod, valid, global_nmesh, owned_start, owned_end, slice_width, num_devices):
    left_start = (owned_start - slice_width) % global_nmesh
    right_end = (owned_end + slice_width) % global_nmesh
    stay = valid & _periodic_interval(x_mod, owned_start, owned_end)
    send_left = valid & _periodic_interval(x_mod, left_start, owned_start)
    send_right = valid & _periodic_interval(x_mod, owned_end, right_end)
    send_right &= num_devices != 2
    return jnp.where(
        stay, jnp.uint8(1),
        jnp.where(
            send_left, jnp.uint8(2), jnp.where(send_right, jnp.uint8(3), jnp.where(valid, jnp.uint8(4), jnp.uint8(0)))
        ),
    )


def _raveled_key(pmid, mesh_shape):
    wrapped = jnp.mod(pmid.astype(jnp.int32), jnp.asarray(mesh_shape, dtype=jnp.int32))
    return ((wrapped[:, 0] * mesh_shape[1] + wrapped[:, 1]) * mesh_shape[2] + wrapped[:, 2]).astype(jnp.uint32)


def _encode_records(pmid, disp, vel, mask, capacity, mesh_shape):
    count = jnp.sum(mask, dtype=jnp.int32)
    indices = jnp.nonzero(mask, size=capacity, fill_value=0)[0]
    packed_valid = jnp.arange(capacity, dtype=jnp.int32) < count
    key = _raveled_key(pmid[indices], mesh_shape)
    records = jnp.concatenate((
        key[:, None], jnp.zeros_like(key)[:, None], jax.lax.bitcast_convert_type(disp[indices], jnp.uint32),
        jax.lax.bitcast_convert_type(vel[indices], jnp.uint32),
    ), axis=1)
    return jnp.where(packed_valid[:, None], records, jnp.zeros_like(records)), count


def _decode_records(records, mesh_shape):
    key = records[:, 0]
    yz = key % np.uint32(mesh_shape[1] * mesh_shape[2])
    pmid = jnp.stack((
        key // np.uint32(mesh_shape[1] * mesh_shape[2]), yz // np.uint32(mesh_shape[2]), yz % np.uint32(mesh_shape[2]),
    ), axis=1).astype(jnp.int32)
    disp = jax.lax.bitcast_convert_type(records[:, 2:5], jnp.float32)
    vel = jax.lax.bitcast_convert_type(records[:, 5:8], jnp.float32)
    return key, pmid, disp, vel


def _stable_merge_streams(streams, capacity):
    keys = jnp.concatenate([stream[0] for stream in streams])
    pmid = jnp.concatenate([stream[1] for stream in streams])
    disp = jnp.concatenate([stream[2] for stream in streams])
    vel = jnp.concatenate([stream[3] for stream in streams])
    valid = jnp.concatenate([stream[4] for stream in streams])
    tag = jnp.concatenate([jnp.full(stream[0].shape, index, dtype=jnp.uint8) for index, stream in enumerate(streams)])
    source_idx = jnp.concatenate([jnp.arange(stream[0].shape[0], dtype=jnp.int32) for stream in streams])
    filled_key = jnp.where(valid, keys, jnp.iinfo(jnp.uint32).max)
    order = jnp.argsort(filled_key, stable=True)[:capacity]
    out_valid = valid[order]
    row_mask = out_valid[:, None]
    return (
        jnp.where(row_mask, pmid[order], 0), jnp.where(row_mask, disp[order], 0), jnp.where(row_mask, vel[order], 0),
        out_valid.astype(jnp.uint8), jnp.where(out_valid, tag[order],
                                               jnp.uint8(3)), jnp.where(out_valid, source_idx[order], jnp.int32(-1)),
    )


def _fake_route_pack(
    pmid, disp, vel, valid, x_mod, *, global_nmesh, mesh_shape, owned_start, owned_end, slice_width, direction,
    num_devices, capacity
):
    classes = _fake_route_classes(x_mod, valid, global_nmesh, owned_start, owned_end, slice_width, num_devices)
    records, count = _encode_records(pmid, disp, vel, classes == (2 if direction < 0 else 3), capacity, mesh_shape)
    return records, count, classes


def _fake_route_pack_bidir(
    pmid, disp, vel, valid, x_mod, *, global_nmesh, mesh_shape, owned_start, owned_end, slice_width, num_devices,
    capacity, stay_capacity
):
    classes = _fake_route_classes(x_mod, valid, global_nmesh, owned_start, owned_end, slice_width, num_devices)
    left, left_count = _encode_records(pmid, disp, vel, classes == 2, capacity, mesh_shape)
    right, right_count = _encode_records(pmid, disp, vel, classes == 3, capacity, mesh_shape)
    stay_mask = classes == 1
    stay_count = jnp.sum(stay_mask, dtype=jnp.int32)
    stay_indices = jnp.nonzero(stay_mask, size=stay_capacity, fill_value=0)[0]
    stay_valid = jnp.arange(stay_capacity, dtype=jnp.int32) < stay_count
    stay_indices = jnp.where(stay_valid, stay_indices, jnp.int32(-1))
    return left, right, left_count, right_count, classes, stay_indices, stay_count


def _fake_route_merge(
    pmid, disp, vel, stay_mask, incoming_records, incoming_count, *, mesh_shape, capacity, auxiliary=False
):
    stay_count = jnp.sum(stay_mask, dtype=jnp.int32)
    stay_indices = jnp.nonzero(stay_mask, size=capacity, fill_value=0)[0]
    stay_valid = jnp.arange(capacity, dtype=jnp.int32) < stay_count
    local = (
        _raveled_key(pmid[stay_indices],
                     mesh_shape), pmid[stay_indices], disp[stay_indices], vel[stay_indices], stay_valid
    )
    incoming_key, incoming_pmid, incoming_disp, incoming_vel = _decode_records(incoming_records, mesh_shape)
    incoming_valid = jnp.arange(incoming_records.shape[0], dtype=jnp.int32) < incoming_count
    incoming = (incoming_key, incoming_pmid, incoming_disp, incoming_vel, incoming_valid)
    merged = _stable_merge_streams((local, incoming), capacity)
    if auxiliary:
        return merged
    return merged[:4]


def _fake_route_merge_bidir(
    pmid, disp, vel, stay_indices, stay_count, left_records, left_count, right_records, right_count, *, mesh_shape,
    capacity
):
    stay_valid = jnp.arange(stay_indices.shape[0], dtype=jnp.int32) < stay_count
    safe_stay = jnp.clip(stay_indices, 0, pmid.shape[0] - 1)
    local = (_raveled_key(pmid[safe_stay], mesh_shape), pmid[safe_stay], disp[safe_stay], vel[safe_stay], stay_valid)
    left_key, left_pmid, left_disp, left_vel = _decode_records(left_records, mesh_shape)
    right_key, right_pmid, right_disp, right_vel = _decode_records(right_records, mesh_shape)
    left = (left_key, left_pmid, left_disp, left_vel, jnp.arange(left_records.shape[0], dtype=jnp.int32) < left_count)
    right = (
        right_key, right_pmid, right_disp, right_vel, jnp.arange(right_records.shape[0], dtype=jnp.int32) < right_count
    )
    merged = _stable_merge_streams((local, left, right), capacity)
    merged_key_lo = jnp.where(merged[3] != 0, _raveled_key(merged[0], mesh_shape), jnp.iinfo(jnp.uint32).max)
    merged_key_hi = jnp.where(merged[3] != 0, jnp.uint32(0), jnp.iinfo(jnp.uint32).max)
    merged_key = jnp.stack((merged_key_lo, merged_key_hi), axis=-1)
    return (*merged, merged_key, jnp.sum(merged[3] != 0, dtype=jnp.int32))


def _fake_route_merge_bidir_primal(*args, **kwargs):
    merged = _fake_route_merge_bidir(*args, **kwargs)
    return (*merged[:4], merged[-1])


def _fake_route_transpose_split(merged_cot, source_tag, source_idx, *, auth_size, share_capacity):
    outputs = []
    for tag, size in ((0, auth_size), (1, share_capacity), (2, share_capacity)):
        selected = source_tag == tag
        scale = selected.reshape(selected.shape + (1, ) * (merged_cot.ndim - 1)).astype(merged_cot.dtype)
        output = jnp.zeros((size, ) + merged_cot.shape[1:], dtype=merged_cot.dtype)
        outputs.append(output.at[jnp.where(selected, source_idx, 0)].add(merged_cot * scale))
    return tuple(outputs)


def _fake_route_transpose_scatter(
    stay_cot, send_left_cot, send_right_cot, stay_pos, stay_valid, send_left_pos, send_left_valid, send_right_pos,
    send_right_valid, *, auth_size, share_capacity
):
    del share_capacity
    output = jnp.zeros((auth_size, ) + stay_cot.shape[1:], dtype=stay_cot.dtype)
    for cot, pos, valid in ((stay_cot, stay_pos, stay_valid), (send_left_cot, send_left_pos, send_left_valid),
                            (send_right_cot, send_right_pos, send_right_valid)):
        scale = valid.reshape(valid.shape + (1, ) * (cot.ndim - 1)).astype(cot.dtype)
        output = output.at[jnp.where(valid, pos, 0)].add(cot * scale)
    return output


def _install_fake_cuda_routing(monkeypatch, *, bidirectional):
    monkeypatch.setattr(routing, "cuda_routing_enabled", lambda conf: True)
    monkeypatch.setattr(routing, "_use_bidir_cuda_routing", lambda conf: bidirectional)
    monkeypatch.setattr(routing, "cuda_route_pack", _fake_route_pack)
    monkeypatch.setattr(routing, "route_pack_bidir_cuda", _fake_route_pack_bidir)
    monkeypatch.setattr(routing, "cuda_route_merge", _fake_route_merge)
    monkeypatch.setattr(routing, "route_merge_bidir_cuda", _fake_route_merge_bidir)
    monkeypatch.setattr(routing, "route_merge_bidir_primal_i16", _fake_route_merge_bidir_primal)
    monkeypatch.setattr(routing, "cuda_route_transpose_split", _fake_route_transpose_split)
    monkeypatch.setattr(routing, "cuda_route_transpose_scatter", _fake_route_transpose_scatter)


def test_wide_routing_keys_cross_u32_boundaries_without_x64():
    conf = types.SimpleNamespace(mesh_shape=(2048, 2048, 2048), mesh_size=2048**3)
    pmid = jnp.asarray([[1023, 2047, 2047], [1024, 0, 0], [2047, 2047, 2047]], dtype=jnp.int16)

    keys = routing._routing_keys_from_pmid(pmid, conf)

    assert keys.dtype == jnp.uint32
    np.testing.assert_array_equal(
        np.asarray(keys), np.asarray([[0xffffffff, 0], [0, 1], [0xffffffff, 1]], dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        np.asarray(routing._key_fill_value(conf)), np.asarray([0xffffffff, 0xffffffff], dtype=np.uint32)
    )


def test_routing_scalar_count_arithmetic_saturates_without_large_allocations():
    limit = np.iinfo(np.int32).max
    assert int(routing._saturating_add_nonnegative_int32(jnp.int32(limit - 2), jnp.int32(10))) == limit
    assert int(
        routing._saturating_add_nonnegative_int32(jnp.int32(limit // 2), jnp.int32(limit // 2), jnp.int32(100),
                                                  )
    ) == limit
    gathered = jnp.asarray([limit - 4, 2, 8, 0, 0, 0, 0, 0], dtype=jnp.int32)
    assert int(routing._saturating_sum_nonnegative_int32(gathered)) == limit
    assert routing._saturating_add_nonnegative_int32(jnp.int32(7), jnp.int32(11)).shape == ()


def test_wide_routing_key_search_is_lexicographic_high_then_low():
    sorted_keys = jnp.asarray([[0xffffffff, 0], [0, 1], [0xffffffff, 1], [0xffffffff, 0xffffffff]], dtype=jnp.uint32)
    values = sorted_keys[:3]

    np.testing.assert_array_equal(
        np.asarray(routing._key_searchsorted(sorted_keys, values, side="left")), np.asarray([0, 1, 2])
    )
    np.testing.assert_array_equal(
        np.asarray(routing._key_searchsorted(sorted_keys, values, side="right")), np.asarray([1, 2, 3])
    )


def test_portable_wide_key_merge_preserves_global_order():
    sentinel = jnp.asarray([0xffffffff, 0xffffffff], dtype=jnp.uint32)
    keys_a = jnp.asarray([[0xffffffff, 0], [0xffffffff, 1], sentinel], dtype=jnp.uint32)
    valid_a = jnp.asarray([True, True, False])
    keys_b = jnp.asarray([[0, 1], sentinel], dtype=jnp.uint32)
    valid_b = jnp.asarray([True, False])

    source, a_idx, b_idx, valid, _ = routing._linear_merge_plan_two(keys_a, valid_a, keys_b, valid_b, 3, sentinel)
    merged = routing._linear_take_two(keys_a, keys_b, source, a_idx, b_idx)
    merged = routing._fill_invalid_keys(merged, valid, sentinel)

    np.testing.assert_array_equal(
        np.asarray(merged), np.asarray([[0xffffffff, 0], [0, 1], [0xffffffff, 1]], dtype=np.uint32)
    )


def test_native_no_acc_route_returns_zero_padded_merge_without_repacking(monkeypatch):
    conf = types.SimpleNamespace(mesh_shape=(4, 1, 1), mesh_size=4)
    pmid = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=jnp.int16)
    disp = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    vel = disp + jnp.asarray([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
    unused = jnp.asarray([False, False, True])

    monkeypatch.setattr(routing, "cuda_routing_enabled", lambda _: True)

    def fake_route(keys, route_pmid, route_disp, route_vel, valid, *args):
        del args
        assert keys is None
        return (keys, route_pmid, route_disp, route_vel, valid), jnp.int32(0)

    monkeypatch.setattr(routing, "_canonical_route_authoritative_no_acc", fake_route)
    monkeypatch.setattr(
        routing, "_pack_authoritative_only_no_acc", lambda *args:
        (_ for _ in ()).throw(AssertionError("native output must not be repacked")),
    )

    result = routing.move_particles_mesh_halo_no_acc_shard_map(
        pmid, disp, disp, vel, None, None, unused, 4, 2, 0, 3, (), (), 2, 1.0, (), conf,
    )

    np.testing.assert_array_equal(np.asarray(result[0]), np.asarray(pmid))
    np.testing.assert_array_equal(np.asarray(result[1]), np.asarray(disp))
    np.testing.assert_array_equal(np.asarray(result[2]), np.asarray(vel))
    np.testing.assert_array_equal(np.asarray(result[4]), np.asarray(unused))


def test_fused_low_memory_route_fails_closed_on_uncapped_counts(monkeypatch):
    conf = types.SimpleNamespace(mesh_shape=(8, 1, 1))
    pmid = jnp.zeros((4, 3), dtype=jnp.int16)
    disp = jnp.zeros((4, 3), dtype=jnp.float32)
    vel = jnp.zeros_like(disp)
    unused = jnp.zeros((4, ), dtype=jnp.bool_)
    records = jnp.zeros((4, 8), dtype=jnp.uint32)
    monkeypatch.setattr(
        routing, "_routing_keys_from_pmid", lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError("fused route must not allocate keys")),
    )
    monkeypatch.setattr(
        routing, "_x_mod_from_disp", lambda *args, **kwargs:
        (_ for _ in ()).throw(AssertionError("fused route must not allocate x_mod")),
    )
    monkeypatch.setattr(routing.jax.lax, "axis_index", lambda _: jnp.int32(0))
    monkeypatch.setattr(routing.jax.lax, "ppermute", lambda value, **kwargs: value)
    monkeypatch.setattr(routing.jax.lax, "pmax", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(
        routing.jax.lax, "all_gather",
        lambda value, *args, **kwargs: jnp.asarray([np.iinfo(np.int32).max - 1, 10], dtype=jnp.int32),
    )
    monkeypatch.setattr(
        routing, "route_pack_bidir_drift_primal_i16", lambda *args, **kwargs:
        (records, records, jnp.int32(5), jnp.int32(0), jnp.asarray([2], jnp.uint32), jnp.int32(2), jnp.int32(1)),
    )
    monkeypatch.setattr(
        routing, "route_merge_bidir_drift_primal_i16", lambda *args, **kwargs:
        (pmid, disp, vel, jnp.ones((4, ), jnp.bool_), jnp.int32(7)),
    )

    result = routing.move_particles_mesh_halo_fused_drift_low_memory_shard_map(
        pmid, disp, vel, jnp.float32(1), unused, 8, 4, ((0, 1), (1, 0)), ((0, 1), (1, 0)), 2, 1.0,
        jnp.asarray([0, 4], dtype=jnp.int32), conf,
    )
    assert bool(result[5])
    assert int(result[6]) == 5
    assert int(result[7]) == np.iinfo(np.int32).max


def test_migration_domain_failure_reports_slab_and_displacement(monkeypatch):
    monkeypatch.setattr(routing.jax.lax, "pmax", lambda value, *args, **kwargs: value)
    disp = jnp.asarray([[-3.0, 4.0, 0.0], [1.5, 0.0, 0.0]], dtype=jnp.float32)
    valid = jnp.asarray([True, True])

    with pytest.raises(Exception) as error:
        routing._synchronized_migration_domain_check(jnp.int32(1), disp, valid, 8, 2.0)
        jax.effects_barrier()

    message = str(error.value)
    assert "particles_outside_neighbor_range=1" in message
    assert "slab_width_mesh_cells=8" in message
    assert "slab_width_simulation_units=4" in message
    assert "max_abs_x_displacement_mesh_cells=6" in message
    assert "max_abs_x_displacement_simulation_units=3" in message


def _conf(mode, devices=2):
    available = jax.devices("gpu")
    if len(available) < devices:
        pytest.skip(f"this routing test requires {devices} GPUs")
    return Configuration(
        1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(available[:devices]), mode=mode),
        max_ptcl_per_slice=64, max_share_ptcl=24, max_halo_share_ptcl=40, max_share_gather_ptcl=24,
    )


def _physical_states(ptcl):
    active = ~np.asarray(ptcl.unused_index)
    positions = np.asarray(ptcl.pos(dtype=jnp.float32))[active]
    velocities = np.asarray(ptcl.vel)[active]
    accelerations = np.asarray(ptcl.acc)[active]
    by_id = {}
    for pos, vel, acc in zip(positions, velocities, accelerations):
        particle_id = int(round(float(acc[0])))
        state = (pos, vel, acc)
        if particle_id in by_id:
            np.testing.assert_allclose(pos, by_id[particle_id][0], atol=5e-7)
            np.testing.assert_allclose(vel, by_id[particle_id][1], atol=5e-7)
            np.testing.assert_allclose(acc, by_id[particle_id][2], atol=5e-7)
        else:
            by_id[particle_id] = state
    return by_id, active


def test_authoritative_packers_zero_invalid_tails_and_preserve_prefix_order():
    pmid = jnp.asarray([[1, 0, 0], [3, 1, 1], [0, 0, 0]], dtype=jnp.int16)
    disp = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [8, 8, 8]], dtype=jnp.float32)
    vel = disp + 10
    acc = disp + 20
    valid = jnp.asarray([True, True, False])

    packed = _pack_authoritative_only(pmid, disp, vel, acc, valid, 5)
    np.testing.assert_array_equal(np.asarray(packed[0])[:2], np.asarray(pmid)[:2])
    np.testing.assert_allclose(np.asarray(packed[1])[:2], np.asarray(disp)[:2])
    np.testing.assert_allclose(np.asarray(packed[2])[:2], np.asarray(vel)[:2])
    np.testing.assert_allclose(np.asarray(packed[3])[:2], np.asarray(acc)[:2])
    for value in packed[:4]:
        np.testing.assert_allclose(np.asarray(value)[2:], 0)
    np.testing.assert_array_equal(np.asarray(packed[4]), False)
    np.testing.assert_array_equal(np.asarray(packed[5]), [False, False, True, True, True])

    packed_no_acc = _pack_authoritative_only_no_acc(pmid, disp, vel, valid, 5)
    for actual, expected in zip(packed_no_acc[:3], packed[:3]):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(packed_no_acc[3]), False)
    np.testing.assert_array_equal(np.asarray(packed_no_acc[4]), np.asarray(packed[5]))

    already_full = _pack_authoritative_only(pmid, disp, vel, acc, valid, 3)
    np.testing.assert_allclose(np.asarray(already_full[0])[2], 0)
    np.testing.assert_array_equal(np.asarray(already_full[5]), [False, False, True])


def test_compact_transposes_scatter_only_valid_rows_and_sum_duplicate_slots():
    compact = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]])
    slots = jnp.asarray([3, 1, -1], dtype=jnp.int32)
    valid = jnp.asarray([True, True, False])
    dense = _scatter_compact_to_dense(compact, slots, valid, 5)
    np.testing.assert_array_equal(np.asarray(dense), [[0, 0], [3, 4], [0, 0], [1, 2], [0, 0]])
    np.testing.assert_array_equal(np.asarray(_mask_compact_prefix(compact, valid)), [[1, 2], [3, 4], [0, 0]])
    np.testing.assert_array_equal(
        np.asarray(_reverse_build_owned_only_cot(compact, 99, valid)), [[1, 2], [3, 4], [0, 0]]
    )

    duplicate_slots = jnp.asarray([1, 1, 3], dtype=jnp.int32)
    duplicate_dense = _scatter_compact_to_dense(compact, duplicate_slots, jnp.ones(3, dtype=jnp.bool_), 5)
    np.testing.assert_array_equal(np.asarray(duplicate_dense), [[0, 0], [4, 6], [0, 0], [100, 200], [0, 0]])


def test_halo_mask_shard_helper_uses_displaced_periodic_positions_and_ignores_padding():
    pmid = jnp.asarray([[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 0, 0]], dtype=jnp.int16)
    disp = jnp.asarray([[-0.2, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0, 0, 0]], dtype=jnp.float32)
    unused = jnp.asarray([False, False, False, True])
    actual = compute_halo_mask_shard_map(
        pmid, disp, unused, jnp.asarray([3, 0]), jnp.asarray([1, 2]), global_nMesh=4, disp_size=1.0,
    )
    np.testing.assert_array_equal(np.asarray(actual), [True, True, True, False])


def test_halo_capacity_uses_explicit_value_or_conservative_geometry_bound():
    assert _halo_capacity(types.SimpleNamespace(max_halo_share_ptcl=17)) == 17
    inferred = types.SimpleNamespace(
        max_halo_share_ptcl=None, max_ptcl_per_slice=100, ptcl_halo_width=3, local_mesh_shape=(16, 8, 8),
    )
    assert _halo_capacity(inferred) == 19
    capped = types.SimpleNamespace(
        max_halo_share_ptcl=None, max_ptcl_per_slice=7, ptcl_halo_width=20, local_mesh_shape=(1, 8, 8),
    )
    assert _halo_capacity(capped) == 7


@pytest.mark.parametrize("mode", ["mesh_halo", "particle_halo"])
def test_two_gpu_bidirectional_migration_and_reverse_drift_preserve_every_physical_state(mode):
    conf = _conf(mode)
    single = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    grid = Particles.gen_grid(single)
    particle_ids = jnp.arange(single.ptcl_num, dtype=jnp.float32)
    velocity = jnp.zeros_like(grid.disp)
    velocity = velocity.at[grid.pmid[:, 0] == 0, 0].set(jnp.float32(-0.35))
    velocity = velocity.at[grid.pmid[:, 0] == 2, 0].set(jnp.float32(-0.35))
    acceleration = jnp.stack((particle_ids, 2 * particle_ids + 1, -particle_ids), axis=1)
    before = Particles.from_pmid(conf, grid.pmid, grid.disp, vel=velocity, acc=acceleration)
    after_disp = before.disp + before.vel

    moved = conf.mGPU_halo_moving(
        before.pmid, before.disp, after_disp, before.vel, before.acc, conf.halo_start, conf.halo_end,
        before.unused_index,
    )
    moved_ptcl = Particles(
        conf, moved[0], moved[1], vel=moved[2], acc=moved[3], halo_mask=moved[4], unused_index=moved[5],
    )
    assert not bool(np.asarray(moved[6]))
    assert int(np.asarray(moved[7])) == 16
    moved_states, moved_active = _physical_states(moved_ptcl)
    assert set(moved_states) == set(range(64))
    expected_pos = np.mod(np.asarray(grid.pos(dtype=jnp.float32)) + np.asarray(velocity), 4)
    for particle_id in range(64):
        np.testing.assert_allclose(moved_states[particle_id][0], expected_pos[particle_id], atol=5e-7)
        np.testing.assert_allclose(moved_states[particle_id][1], np.asarray(velocity)[particle_id], atol=5e-7)
        np.testing.assert_allclose(moved_states[particle_id][2], np.asarray(acceleration)[particle_id], atol=5e-7)

    if mode == "mesh_halo":
        assert moved_active.sum() == 64
        assert not np.any(np.asarray(moved_ptcl.halo_mask))
        moved_no_acc = conf.mGPU_halo_moving_no_acc(
            before.pmid, before.disp, after_disp, before.vel, conf.halo_start, conf.halo_end, before.unused_index,
        )
        for actual, expected in zip(moved_no_acc[:3], moved[:3]):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=5e-7)
        np.testing.assert_array_equal(np.asarray(moved_no_acc[3]), np.asarray(moved[4]))
        np.testing.assert_array_equal(np.asarray(moved_no_acc[4]), np.asarray(moved[5]))
    else:
        assert moved_active.sum() > 64
        assert np.any(np.asarray(moved_ptcl.halo_mask))

    reconstructed = conf.mGPU_reconstruct_pre_drift(
        moved[0], moved[1], moved[2], moved[3], conf.halo_start, conf.halo_end, moved[5], jnp.float32(1),
    )
    reconstructed_ptcl = Particles(
        conf, reconstructed[0], reconstructed[1], vel=reconstructed[2], acc=reconstructed[3],
        unused_index=reconstructed[4], halo_mask=reconstructed[5],
    )
    reconstructed_states, _ = _physical_states(reconstructed_ptcl)
    assert set(reconstructed_states) == set(range(64))
    expected_before = np.asarray(grid.pos(dtype=jnp.float32))
    for particle_id in range(64):
        np.testing.assert_allclose(reconstructed_states[particle_id][0], expected_before[particle_id], atol=5e-7)

    cot = jnp.ones_like(reconstructed[1])
    pulled = conf.mGPU_halo_move_pullback(
        reconstructed[0], reconstructed[1], moved[1], reconstructed[2], reconstructed[3], conf.halo_end,
        reconstructed[4], cot, 2 * cot, 3 * cot,
    )
    assert len(pulled) == 3
    for value in pulled:
        assert value.shape == cot.shape
        assert np.all(np.isfinite(np.asarray(value)))


@pytest.mark.parametrize("mode", ["mesh_halo", "particle_halo"])
def test_one_gpu_runtime_noops_return_exact_fields_and_expected_masks(mode):
    conf = _conf(mode, devices=1)
    ptcl = Particles.gen_grid(conf, vel=True, acc=True)
    after = ptcl.disp + jnp.float32(0.25)
    moved = conf.mGPU_halo_moving(
        ptcl.pmid, ptcl.disp, after, ptcl.vel, ptcl.acc, conf.halo_start, conf.halo_end, ptcl.unused_index,
    )
    np.testing.assert_array_equal(np.asarray(moved[0]), np.asarray(ptcl.pmid))
    np.testing.assert_allclose(np.asarray(moved[1]), np.asarray(after))
    np.testing.assert_array_equal(np.asarray(moved[4]), False)
    np.testing.assert_array_equal(np.asarray(moved[5]), np.asarray(ptcl.unused_index))
    assert not bool(np.asarray(moved[6]))
    assert int(np.asarray(moved[7])) == 0

    reconstructed = conf.mGPU_reconstruct_pre_drift(
        moved[0], moved[1], moved[2], moved[3], conf.halo_start, conf.halo_end, moved[5], jnp.float32(123),
    )
    for actual, expected in zip(reconstructed[:5], (moved[0], moved[1], moved[2], moved[3], moved[5])):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(reconstructed[5]), False)

    cot = jnp.arange(ptcl.disp.size, dtype=jnp.float32).reshape(ptcl.disp.shape)
    pulled = conf.mGPU_halo_move_pullback(
        ptcl.pmid, ptcl.disp, after, ptcl.vel, ptcl.acc, conf.halo_end, ptcl.unused_index, cot, cot + 1, cot + 2,
    )
    for actual, expected in zip(pulled, (cot, cot + 1, cot + 2)):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    computed_mask = conf.mGPU_compute_halo_mask(
        ptcl.pmid, ptcl.disp, ptcl.unused_index, conf.halo_start, conf.halo_end,
    )
    np.testing.assert_array_equal(np.asarray(computed_mask), False)
    assert conf.mGPU_halo_moving_no_acc is None
    assert conf.mGPU_reconstruct_pre_drift_pullback is None


@pytest.mark.parametrize("bidirectional", [False, True])
def test_two_gpu_cuda_route_orchestration_matches_canonical_forward_and_pullback(monkeypatch, bidirectional):
    conf = _conf("mesh_halo")
    single = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    grid = Particles.gen_grid(single)
    ids = jnp.arange(single.ptcl_num, dtype=jnp.float32)
    vel = jnp.stack((
        jnp.where((grid.pmid[:, 0] == 0) | (grid.pmid[:, 0] == 2), jnp.float32(-0.35),
                  jnp.float32(0.08)), jnp.sin(ids) * jnp.float32(0.01), jnp.cos(ids) * jnp.float32(0.01),
    ), axis=1)
    acc = jnp.stack((ids, 2 * ids + 1, -ids), axis=1)
    before = Particles.from_pmid(conf, grid.pmid, grid.disp, vel=vel, acc=acc)
    after_disp = before.disp + before.vel
    cot = jnp.arange(before.disp.size, dtype=jnp.float32).reshape(before.disp.shape) / 100

    canonical_forward = conf.mGPU_halo_moving_no_acc(
        before.pmid, before.disp, after_disp, before.vel, conf.halo_start, conf.halo_end, before.unused_index,
    )
    canonical_pullback = conf.mGPU_halo_move_pullback(
        before.pmid, before.disp, after_disp, before.vel, before.acc, conf.halo_end, before.unused_index, cot,
        2 * cot + 1, 3 * cot - 2,
    )

    _install_fake_cuda_routing(monkeypatch, bidirectional=bidirectional)
    if bidirectional:
        monkeypatch.setattr(routing, "_DEBUG_BIDIR_ROUTE", True)
        monkeypatch.setattr(
            routing, "_canonical_route_authoritative_with_aux_cuda",
            routing._canonical_route_authoritative_with_aux_bidir_cuda,
        )
    jax.clear_caches()

    native_orchestration_forward = conf.mGPU_halo_moving_no_acc(
        before.pmid, before.disp, after_disp, before.vel, conf.halo_start, conf.halo_end, before.unused_index,
    )
    for actual, expected in zip(native_orchestration_forward, canonical_forward):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=3e-7)

    native_orchestration_pullback = conf.mGPU_halo_move_pullback(
        before.pmid, before.disp, after_disp, before.vel, before.acc, conf.halo_end, before.unused_index, cot,
        2 * cot + 1, 3 * cot - 2,
    )
    for actual, expected in zip(native_orchestration_pullback, canonical_pullback):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=5e-7)


def test_three_stream_loopback_route_preserves_stable_order_and_exact_transpose(monkeypatch):
    devices = jax.devices("gpu")
    if not devices:
        pytest.skip("this routing test requires a GPU")
    conf = Configuration(1.0, (6, 2, 2), mesh_shape=1, float_dtype=jnp.float32)
    mesh = Mesh(np.asarray(devices[:1]), (routing.AXIS_NAME, ))
    pmid = jnp.asarray([[x, 0, 0] for x in range(6)], dtype=jnp.int32)
    sequence = jnp.arange(6, dtype=jnp.float32)
    disp = jnp.stack((jnp.zeros(6, dtype=jnp.float32), sequence / 100, -sequence / 100), axis=1)
    vel = disp + jnp.asarray([0.25, -0.5, 0.75], dtype=jnp.float32)
    acc = 2 * vel - 1
    valid = jnp.ones(6, dtype=jnp.bool_)
    keys = _raveled_key(pmid, conf.mesh_shape).astype(jnp.int32)
    perms = ((0, 0), )
    offsets = jnp.asarray([0], dtype=jnp.int32)
    common = dict(
        global_nMesh=6, max_values_to_share=2, left_perm=perms, right_perm=perms, num_gpus=3, disp_size=1.0,
        offsets=offsets, conf=conf,
    )
    particle_specs = (
        P(routing.AXIS_NAME), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME,
                                                                                        None), P(routing.AXIS_NAME)
    )
    full_out = ((
        P(routing.AXIS_NAME), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME, None),
        P(routing.AXIS_NAME, None), P(routing.AXIS_NAME)
    ), P())

    canonical_full = jax.shard_map(
        lambda key, particle, displacement, velocity, acceleration, is_valid: routing.
        _canonical_route_authoritative(key, particle, displacement, velocity, acceleration, is_valid, **common),
        mesh=mesh, in_specs=(*particle_specs[:-1], P(routing.AXIS_NAME,
                                                     None), particle_specs[-1]), out_specs=full_out, check_vma=False,
    )(keys, pmid, disp, vel, acc, valid)
    for actual, expected in zip(canonical_full[0][:5], (keys, pmid, disp, vel, acc)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(canonical_full[0][5]), True)
    assert int(canonical_full[1]) == 2

    no_acc_out = ((
        P(routing.AXIS_NAME), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME,
                                                                                        None), P(routing.AXIS_NAME)
    ), P())
    canonical_no_acc = jax.shard_map(
        lambda key, particle, displacement, velocity, is_valid: routing.
        _canonical_route_authoritative_no_acc(key, particle, displacement, velocity, is_valid, **common), mesh=mesh,
        in_specs=particle_specs, out_specs=no_acc_out, check_vma=False,
    )(keys, pmid, disp, vel, valid)
    for actual, expected in zip(canonical_no_acc[0][:4], (keys, pmid, disp, vel)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-7)

    aux_out = ((
        P(routing.AXIS_NAME), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME, None), P(routing.AXIS_NAME, None),
        P(routing.AXIS_NAME, None), P(routing.AXIS_NAME)
    ), tuple(P(routing.AXIS_NAME) for _ in range(8)),
               )
    canonical_aux = jax.shard_map(
        lambda key, particle, displacement, velocity, acceleration, is_valid: routing.
        _canonical_route_authoritative_with_aux(
            key, particle, displacement, velocity, acceleration, is_valid, **common
        ), mesh=mesh, in_specs=(*particle_specs[:-1], P(routing.AXIS_NAME,
                                                        None), particle_specs[-1]), out_specs=aux_out, check_vma=False,
    )(keys, pmid, disp, vel, acc, valid)
    for actual, expected in zip(canonical_aux[0][:5], (keys, pmid, disp, vel, acc)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-7)

    _install_fake_cuda_routing(monkeypatch, bidirectional=False)
    merged_cot = jnp.arange(54, dtype=jnp.float32).reshape(6, 3, 3) / 10

    def native_and_transpose(key, particle, displacement, velocity, acceleration, is_valid, cotangent):
        merged, aux = routing._canonical_route_authoritative_with_aux_cuda(
            key, particle, displacement, velocity, acceleration, is_valid, **common
        )
        native_transpose = routing._reverse_route_cot(
            cotangent, *aux, auth_size=6, max_values_to_share=2, left_perm=perms, right_perm=perms, conf=conf,
        )
        reference_transpose = routing._reverse_route_cot(
            cotangent, *aux, auth_size=6, max_values_to_share=2, left_perm=perms, right_perm=perms, conf=None,
        )
        return merged, native_transpose, reference_transpose

    native_aux = jax.shard_map(
        native_and_transpose, mesh=mesh, in_specs=(
            *particle_specs[:-1], P(routing.AXIS_NAME, None), particle_specs[-1], P(routing.AXIS_NAME, None, None)
        ), out_specs=(aux_out[0], P(routing.AXIS_NAME, None, None), P(routing.AXIS_NAME, None, None)), check_vma=False,
    )(keys, pmid, disp, vel, acc, valid, merged_cot)
    for actual, expected in zip(native_aux[0][:5], (keys, pmid, disp, vel, acc)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(native_aux[0][5]), True)
    np.testing.assert_allclose(np.asarray(native_aux[1]), np.asarray(native_aux[2]), rtol=0, atol=0)

    native_no_acc = jax.shard_map(
        lambda key, particle, displacement, velocity, is_valid: routing.
        _canonical_route_authoritative_no_acc_cuda(key, particle, displacement, velocity, is_valid, **common),
        mesh=mesh, in_specs=particle_specs, out_specs=no_acc_out, check_vma=False,
    )(keys, pmid, disp, vel, valid)
    assert native_no_acc[0][0] is None
    for actual, expected in zip(native_no_acc[0][1:4], (pmid, disp, vel)):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(native_no_acc[0][4]), True)
    assert int(native_no_acc[1]) == 2
