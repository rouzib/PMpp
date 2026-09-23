"""Small independent checks for direct exceptional particle exchange.

Run in a fresh CPU process with XLA_FLAGS=--xla_force_host_platform_device_count=4
and JAX_PLATFORMS=cpu, or on four physical devices.
"""

import numpy as np
import pytest
from types import SimpleNamespace
import os

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from src.pmpp.core.utils import AXIS_NAME
from src.pmpp.distributed.routing import (
    _exchange_far_drift_records, _hybrid_route_authoritative_fields,
    reconstruct_pre_drift_and_pullback_mesh_halo_shard_map,
)
from src.pmpp.core.utils import build_ring_permutations


def _test_devices():
    platform = os.environ.get("PMPP_HYBRID_TEST_PLATFORM", "cpu")
    return jax.devices(platform)


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_far_exchange_matches_independent_destination_oracle(chunk_size):
    devices = _test_devices()
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    capacity = 5
    pmid = np.zeros((4, capacity, 3), np.int16)
    disp = np.zeros((4, capacity, 3), np.float32)
    vel = np.zeros_like(disp)
    valid = np.zeros((4, capacity), bool)
    # The first two rows on each source cross exactly two of four slabs.
    # The third row crosses only to a neighbor and must not enter far packets.
    for source in range(4):
        for row in range(3):
            pmid[source, row] = (source * 4 + row, source, row)
            valid[source, row] = True
        disp[source, 0, 0] = 8
        disp[source, 1, 0] = -8
        disp[source, 2, 0] = 4
        disp[source, 0, 1] = source + 0.25
        disp[source, 1, 1] = source + 0.75

    def route(local_pmid, local_disp, local_vel, local_valid):
        records, count, far, bad, failed = _exchange_far_drift_records(
            local_pmid, local_disp, local_vel, local_valid, jnp.float32(0),
            global_nmesh=16, mesh_shape=(16, 16, 16), disp_size=1,
            num_devices=4, send_capacity=3, recv_capacity=3,
            chunk_size=chunk_size,
        )
        return records, count[None], far[None], bad[None], failed

    mapped = shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None),
                  P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P(AXIS_NAME), P(AXIS_NAME),
                   P(AXIS_NAME), P()),
    )
    records, counts, far_counts, bad_counts, failed = jax.jit(mapped)(
        jnp.asarray(pmid.reshape(4 * capacity, 3)),
        jnp.asarray(disp.reshape(4 * capacity, 3)),
        jnp.asarray(vel.reshape(4 * capacity, 3)),
        jnp.asarray(valid.reshape(4 * capacity)),
    )
    records = np.asarray(records).reshape(4, 3, 8)
    counts = np.asarray(counts).reshape(4)
    assert not bool(failed)
    np.testing.assert_array_equal(counts, 2)
    np.testing.assert_array_equal(np.asarray(far_counts), 2)
    np.testing.assert_array_equal(np.asarray(bad_counts), 0)
    for destination in range(4):
        source = (destination - 2) % 4
        expected = []
        for row in (0, 1):
            x, y, z = pmid[source, row]
            key = (int(x) * 16 + int(y)) * 16 + int(z)
            payload = np.concatenate((disp[source, row], vel[source, row]))
            expected.append(np.concatenate((np.array((key, 0), np.uint32), payload.view(np.uint32))))
        np.testing.assert_array_equal(records[destination, :2], np.asarray(expected, np.uint32))
        np.testing.assert_array_equal(records[destination, 2], 0)


def test_far_exchange_capacity_failure_is_global():
    devices = _test_devices()
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    pmid = jnp.zeros((4 * 2, 3), jnp.int16)
    disp = jnp.zeros((4 * 2, 3), jnp.float32).at[:2, 0].set(8)
    valid = jnp.arange(8) < 2

    def route(local_pmid, local_disp, local_valid):
        *_, failed = _exchange_far_drift_records(
            local_pmid, local_disp, jnp.zeros_like(local_disp), local_valid,
            jnp.float32(0), global_nmesh=16, mesh_shape=(16, 16, 16),
            disp_size=1, num_devices=4, send_capacity=1,
            recv_capacity=1, chunk_size=1,
        )
        return failed

    mapped = shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=P(),
    )
    assert bool(jax.jit(mapped)(pmid, disp, valid))


def test_far_exchange_rejects_nonfinite_sparse_payload():
    devices = _test_devices()
    if len(devices) < 4:
        pytest.skip("needs four logical devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    pmid = jnp.zeros((4 * 2, 3), jnp.int16)
    disp = jnp.zeros((4 * 2, 3), jnp.float32).at[0, 0].set(8).at[0, 1].set(jnp.nan)
    valid = jnp.arange(8) == 0

    def route(local_pmid, local_disp, local_valid):
        records, _, _, _, failed = _exchange_far_drift_records(
            local_pmid, local_disp, jnp.zeros_like(local_disp), local_valid,
            jnp.float32(0), global_nmesh=16, mesh_shape=(16, 16, 16),
            disp_size=1, num_devices=4, send_capacity=2, recv_capacity=2,
            chunk_size=1,
        )
        return records, failed

    mapped = shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P()),
    )
    records, failed = jax.jit(mapped)(pmid, disp, valid)
    assert bool(failed)
    np.testing.assert_array_equal(np.asarray(records), 0)


@pytest.mark.parametrize("far", [False, True])
def test_generic_hybrid_route_matches_global_oracle_and_transposes(far):
    devices = _test_devices()
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    left_perm, right_perm = build_ring_permutations(4)
    conf = SimpleNamespace(
        mesh_shape=(16, 16, 16), mesh_size=16**3,
        multigpu=SimpleNamespace(far_send_capacity=3, far_recv_capacity=3, far_chunk_size=2),
    )
    capacity = 5
    pmid = np.zeros((4, capacity, 3), np.int16)
    disp = np.zeros((4, capacity, 3), np.float32)
    vel = np.zeros_like(disp)
    acc = np.zeros_like(disp)
    valid = np.zeros((4, capacity), bool)
    for source in range(4):
        for row in range(4):
            pmid[source, row] = (source * 4 + row, source, row)
            disp[source, row, 1] = source + row / 8
            vel[source, row, 0] = source * 4 + row + 0.5
            acc[source, row, 0] = source * 4 + row
            valid[source, row] = True
        if far:
            disp[source, 0, 0] = 8
            disp[source, 1, 0] = -8
        disp[source, 2, 0] = 4
    flat_pmid = jnp.asarray(pmid.reshape(4 * capacity, 3))
    flat_valid = jnp.asarray(valid.reshape(4 * capacity))

    def route(local_pmid, local_disp, local_vel, local_acc, local_valid):
        result, _ = _hybrid_route_authoritative_fields(
            None, local_pmid, (local_disp, local_vel, local_acc), local_valid,
            16, 3, left_perm, right_perm, 4, 1, conf,
        )
        return result[1], result[2], result[3], result[4], result[5]

    mapped = jax.jit(shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME, None),
                  P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME, None),
                   P(AXIS_NAME, None), P(AXIS_NAME)),
    ))
    input_disp = jnp.asarray(disp.reshape(4 * capacity, 3))
    input_vel = jnp.asarray(vel.reshape(4 * capacity, 3))
    input_acc = jnp.asarray(acc.reshape(4 * capacity, 3))
    out = mapped(flat_pmid, input_disp, input_vel, input_acc, flat_valid)
    actual = [np.asarray(item).reshape((4, capacity, -1)) if item.ndim == 2
              else np.asarray(item).reshape((4, capacity)) for item in out]
    for destination in range(4):
        expected = []
        for source in range(4):
            for row in range(4):
                x = (int(pmid[source, row, 0]) + float(disp[source, row, 0])) % 16
                if int(x // 4) == destination:
                    key = tuple(int(value) for value in pmid[source, row])
                    expected.append((key, disp[source, row], vel[source, row], acc[source, row]))
        expected.sort(key=lambda item: item[0])
        assert len(expected) == 4
        np.testing.assert_array_equal(actual[4][destination], [True] * 4 + [False])
        for row, (key, expected_disp, expected_vel, expected_acc) in enumerate(expected):
            np.testing.assert_array_equal(actual[0][destination, row], key)
            np.testing.assert_array_equal(actual[1][destination, row], expected_disp)
            np.testing.assert_array_equal(actual[2][destination, row], expected_vel)
            np.testing.assert_array_equal(actual[3][destination, row], expected_acc)

    def float_outputs(d, v):
        _, routed_disp, routed_vel, _, _ = mapped(flat_pmid, d, v, input_acc, flat_valid)
        return routed_disp, routed_vel

    tangent_d = jnp.asarray(np.arange(4 * capacity * 3, dtype=np.float32).reshape(-1, 3) / 17)
    tangent_v = tangent_d / 3
    cot_d = jnp.asarray(np.arange(4 * capacity * 3, dtype=np.float32).reshape(-1, 3) / 19)
    cot_v = cot_d / 5
    _, push = jax.jvp(float_outputs, (input_disp, input_vel), (tangent_d, tangent_v))
    _, pullback = jax.vjp(float_outputs, input_disp, input_vel)
    pull = pullback((cot_d, cot_v))
    lhs = jnp.vdot(push[0], cot_d) + jnp.vdot(push[1], cot_v)
    rhs = jnp.vdot(tangent_d, pull[0]) + jnp.vdot(tangent_v, pull[1])
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-5)


def test_reverse_reconstruction_carries_particle_specific_cotangents():
    devices = _test_devices()
    if len(devices) < 4:
        pytest.skip("needs four logical devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    left_perm, right_perm = build_ring_permutations(4)
    conf = SimpleNamespace(
        mesh_shape=(16, 16, 16), mesh_size=16**3,
        multigpu=SimpleNamespace(migration_policy="hybrid", far_send_capacity=3,
                                 far_recv_capacity=3, far_chunk_size=2),
    )
    capacity = 5
    pmid = np.zeros((4, capacity, 3), np.int16)
    disp = np.zeros((4, capacity, 3), np.float32)
    vel = np.zeros_like(disp)
    acc = np.zeros_like(disp)
    unused = np.ones((4, capacity), bool)
    for source in range(4):
        for row in range(4):
            pmid[source, row] = (source * 4 + row, source, row)
            acc[source, row, 0] = source * 4 + row + 1
            unused[source, row] = False
        vel[source, 0, 0] = 8
        vel[source, 1, 0] = -8
        vel[source, 2, 0] = 4
    flat_pmid = jnp.asarray(pmid.reshape(-1, 3))
    flat_disp = jnp.asarray(disp.reshape(-1, 3))
    flat_vel = jnp.asarray(vel.reshape(-1, 3))
    flat_acc = jnp.asarray(acc.reshape(-1, 3))
    flat_unused = jnp.asarray(unused.reshape(-1))

    def forward(local_pmid, local_disp, local_vel, local_acc, local_unused):
        state, _ = _hybrid_route_authoritative_fields(
            None, local_pmid, (local_disp + local_vel, local_vel, local_acc),
            ~local_unused, 16, 3, left_perm, right_perm, 4, 1, conf,
        )
        return state[1], state[2], state[3], state[4], ~state[5]

    specs = (P(AXIS_NAME, None),) * 4 + (P(AXIS_NAME),)
    mapped_forward = jax.jit(shard_map(
        forward, mesh=mesh, in_specs=specs, out_specs=specs,
    ))
    moved = mapped_forward(flat_pmid, flat_disp, flat_vel, flat_acc, flat_unused)
    cot = jnp.broadcast_to(moved[3][:, :1], moved[1].shape)

    def reverse(local_pmid, local_disp, local_vel, local_acc, local_unused, local_cot):
        return reconstruct_pre_drift_and_pullback_mesh_halo_shard_map(
            local_pmid, local_disp, local_vel, local_acc, local_unused,
            jnp.float32(1), local_cot, 2 * local_cot, 3 * local_cot,
            16, 3, 0, capacity, left_perm, right_perm, 4, 1, None, conf,
        )

    mapped_reverse = jax.jit(shard_map(
        reverse, mesh=mesh,
        in_specs=specs + (P(AXIS_NAME, None),),
        out_specs=(P(AXIS_NAME, None),) * 4 + (P(AXIS_NAME), P(AXIS_NAME))
                  + (P(AXIS_NAME, None),) * 3,
    ))
    recovered = mapped_reverse(*moved, cot)
    np.testing.assert_array_equal(np.asarray(recovered[0]), np.asarray(flat_pmid))
    np.testing.assert_array_equal(np.asarray(recovered[1]), np.asarray(flat_disp))
    np.testing.assert_array_equal(np.asarray(recovered[2]), np.asarray(flat_vel))
    np.testing.assert_array_equal(np.asarray(recovered[4]), np.asarray(flat_unused))
    expected = np.broadcast_to(acc.reshape(-1, 3)[:, :1], acc.reshape(-1, 3).shape)
    np.testing.assert_array_equal(np.asarray(recovered[6]), expected)
    np.testing.assert_array_equal(np.asarray(recovered[7]), 2 * expected)
    np.testing.assert_array_equal(np.asarray(recovered[8]), 3 * expected)


def test_eight_device_sparse_offsets_and_idle_participants():
    devices = _test_devices()
    if len(devices) < 8:
        pytest.skip("needs eight logical devices")
    mesh = Mesh(np.asarray(devices[:8]), (AXIS_NAME,))
    pmid = np.zeros((8, 2, 3), np.int16)
    disp = np.zeros((8, 2, 3), np.float32)
    valid = np.zeros((8, 2), bool)
    for source, delta in ((0, 2), (1, 3), (2, 2)):
        pmid[source, 0] = (source * 4, source, 0)
        disp[source, 0, 0] = delta * 4
        valid[source, 0] = True

    def route(local_pmid, local_disp, local_valid):
        records, count, _, _, failed = _exchange_far_drift_records(
            local_pmid, local_disp, jnp.zeros_like(local_disp), local_valid,
            jnp.float32(0), global_nmesh=32, mesh_shape=(32, 16, 16),
            disp_size=1, num_devices=8, send_capacity=2, recv_capacity=3,
            chunk_size=1,
        )
        return records, count[None], failed

    mapped = jax.jit(shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P(AXIS_NAME), P()),
    ))
    records, counts, failed = mapped(jnp.asarray(pmid.reshape(-1, 3)),
                                     jnp.asarray(disp.reshape(-1, 3)),
                                     jnp.asarray(valid.reshape(-1)))
    assert not bool(failed)
    np.testing.assert_array_equal(np.asarray(counts), [0, 0, 1, 0, 2, 0, 0, 0])
    records = np.asarray(records).reshape(8, 3, 8)
    np.testing.assert_array_equal(records[2, 0, 0], 0)
    np.testing.assert_array_equal(records[4, :2, 0], [((4 * 16 + 1) * 16), ((8 * 16 + 2) * 16)])


def test_equal_keys_preserve_stream_and_source_slot_order():
    devices = _test_devices()
    if len(devices) < 4:
        pytest.skip("needs four logical devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    left_perm, right_perm = build_ring_permutations(4)
    conf = SimpleNamespace(
        mesh_shape=(16, 16, 16), mesh_size=16**3,
        multigpu=SimpleNamespace(far_send_capacity=2, far_recv_capacity=2, far_chunk_size=1),
    )
    pmid = jnp.zeros((4 * 5, 3), jnp.int16)
    disp = np.zeros((4, 5, 3), np.float32)
    valid = np.zeros((4, 5), bool)
    for source in range(4):
        valid[source, 0] = True
        disp[source, 0, 1] = source + 1
    valid[2, 1] = True
    disp[2, 1, 1] = 22

    def route(local_pmid, local_disp, local_valid):
        state, _ = _hybrid_route_authoritative_fields(
            None, local_pmid, (local_disp,), local_valid, 16, 2,
            left_perm, right_perm, 4, 1, conf,
        )
        return state[2], state[3]

    mapped = jax.jit(shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P(AXIS_NAME)),
    ))
    out_disp, out_valid = mapped(pmid, jnp.asarray(disp.reshape(-1, 3)),
                                 jnp.asarray(valid.reshape(-1)))
    np.testing.assert_array_equal(np.asarray(out_valid).reshape(4, 5)[0], [True] * 5)
    np.testing.assert_array_equal(np.asarray(out_disp).reshape(4, 5, 3)[0, :, 1], [1, 4, 2, 3, 22])
