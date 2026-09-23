"""CPU mock of fused pack/merge to check hybrid branch orchestration.

The native CUDA kernels have separate physical-GPU tests. This mock checks
the JAX collective schedule, conditional skipping, and full mover interface.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from src.pmpp.core.utils import AXIS_NAME, build_ring_permutations
from src.pmpp.distributed import routing
from src.pmpp.distributed import cuda


def _fake_pack(pmid, disp, vel, valid, factor, *, global_nmesh, mesh_shape,
               disp_size, num_devices, capacity, **_):
    offset, far, bad = routing._far_destination(
        pmid, disp, vel, valid, factor, global_nmesh, disp_size, num_devices,
    )
    left = valid & (offset == num_devices - 1) & ~bad
    right = valid & (offset == 1) & ~bad
    stay = valid & (offset == 0) & ~bad

    def packet(mask):
        count = jnp.sum(mask, dtype=jnp.int32)
        slots = jnp.nonzero(mask, size=capacity, fill_value=0)[0]
        records = routing._far_packet_from_slots(
            pmid, disp, vel, factor, slots, jnp.arange(capacity) < count, mesh_shape,
        )
        return records, count

    left_records, left_count = packet(left)
    right_records, right_count = packet(right)
    return (left_records, right_records, left_count, right_count,
            jnp.asarray([jnp.sum(stay)], jnp.uint32),
            jnp.sum(stay, dtype=jnp.int32),
            jnp.sum(far | bad, dtype=jnp.int32))


def _fake_merge(pmid, disp, vel, valid, factor, _stay_blocks, _stay_count,
                left_records, left_count, right_records, right_count, *,
                global_nmesh, mesh_shape, disp_size, num_devices, capacity, **_):
    offset, _, bad = routing._far_destination(
        pmid, disp, vel, valid, factor, global_nmesh, disp_size, num_devices,
    )
    stay = valid & (offset == 0) & ~bad
    stay_slots = jnp.nonzero(stay, size=capacity, fill_value=0)[0]
    stay_valid = jnp.arange(capacity) < jnp.sum(stay)
    stay_pmid = pmid[stay_slots]
    stay_disp = disp[stay_slots] + vel[stay_slots] * factor
    stay_vel = vel[stay_slots]

    def decode(records, count):
        key = records[:, 0].astype(jnp.int32)
        x = key // (mesh_shape[1] * mesh_shape[2])
        y = (key // mesh_shape[2]) % mesh_shape[1]
        z = key % mesh_shape[2]
        decoded = jnp.stack((x, y, z), axis=1).astype(pmid.dtype)
        values = records[:, 2:].view(jnp.float32)
        return decoded, values[:, :3], values[:, 3:], jnp.arange(records.shape[0]) < count

    left = decode(left_records, left_count)
    right = decode(right_records, right_count)
    joined_pmid = jnp.concatenate((stay_pmid, left[0], right[0]))
    joined_disp = jnp.concatenate((stay_disp, left[1], right[1]))
    joined_vel = jnp.concatenate((stay_vel, left[2], right[2]))
    joined_valid = jnp.concatenate((stay_valid, left[3], right[3]))
    key = (joined_pmid[:, 0].astype(jnp.int32) * mesh_shape[1]
           + joined_pmid[:, 1].astype(jnp.int32)) * mesh_shape[2]
    key += joined_pmid[:, 2].astype(jnp.int32)
    key = jnp.where(joined_valid, key, jnp.iinfo(jnp.int32).max)
    order = jnp.lexsort((jnp.arange(key.shape[0]), key))[:capacity]
    output_valid = joined_valid[order]
    return (jnp.where(output_valid[:, None], joined_pmid[order], 0),
            jnp.where(output_valid[:, None], joined_disp[order], 0),
            jnp.where(output_valid[:, None], joined_vel[order], 0),
            output_valid, jnp.sum(joined_valid, dtype=jnp.int32))


def test_fused_dispatch_skips_sparse_transport_until_globally_needed(monkeypatch):
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    left_perm, right_perm = build_ring_permutations(4)
    conf = SimpleNamespace(
        mesh_shape=(16, 16, 16),
        multigpu=SimpleNamespace(migration_policy="hybrid", far_send_capacity=3,
                                 far_recv_capacity=3, far_chunk_size=2),
    )
    monkeypatch.setattr(routing, "route_pack_bidir_drift_primal_i16", _fake_pack)
    monkeypatch.setattr(routing, "route_merge_bidir_drift_primal_i16", _fake_merge)
    exchange = routing._exchange_far_drift_records
    events = []

    def counted_exchange(*args, **kwargs):
        jax.debug.callback(lambda: events.append("far"))
        return exchange(*args, **kwargs)

    monkeypatch.setattr(routing, "_exchange_far_drift_records", counted_exchange)
    pmid = np.zeros((4, 5, 3), np.int16)
    unused = np.ones((4, 5), bool)
    near = np.zeros((4, 5, 3), np.float32)
    far = near.copy()
    vel = np.zeros_like(near)
    for source in range(4):
        for row in range(4):
            pmid[source, row] = (source * 4 + row, source, row)
            unused[source, row] = False
        near[source, 2, 0] = 4
        far[source, 2, 0] = 4
        far[source, 0, 0] = 8
        far[source, 1, 0] = -8

    def route(local_pmid, local_disp, local_vel, local_unused):
        return routing.move_particles_mesh_halo_fused_drift_low_memory_shard_map(
            local_pmid, local_disp, local_vel, jnp.float32(0), local_unused,
            16, 3, left_perm, right_perm, 4, 1,
            jnp.asarray([0, 4, 8, 12], jnp.int32), conf,
        )

    mapped = jax.jit(shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None),
                  P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME, None),
                   P(AXIS_NAME), P(AXIS_NAME), P(), P(), P()),
    ))
    arguments = (jnp.asarray(pmid.reshape(-1, 3)),
                 jnp.asarray(vel.reshape(-1, 3)), jnp.asarray(unused.reshape(-1)))
    for case, expected_events in ((near, 0), (far, 4), (near, 4)):
        result = mapped(arguments[0], jnp.asarray(case.reshape(-1, 3)), arguments[1], arguments[2])
        jax.block_until_ready(result)
        assert not bool(result[5]) and int(result[7]) == 0
        assert len(events) == expected_events
        assert int(np.sum(~np.asarray(result[4]))) == 16
    assert mapped._cache_size() == 1


@pytest.mark.parametrize("augmented", [False, True])
def test_native_ffi_merge_outputs_match_varying_failure_branch(monkeypatch, augmented):
    """FFI outputs need an explicit manual-axis type before lax.cond."""
    devices = jax.devices("cpu")
    if len(devices) < 4:
        pytest.skip("needs four logical CPU devices")
    mesh = Mesh(np.asarray(devices[:4]), (AXIS_NAME,))
    monkeypatch.setattr(cuda, "_FUSED_PRIMAL_REGISTERED", True)
    monkeypatch.setattr(cuda, "_HYBRID_REGISTERED", True)

    def fake_ffi_call(_target, outputs):
        return lambda *args, **kwargs: tuple(jnp.zeros(spec.shape, spec.dtype) for spec in outputs)

    monkeypatch.setattr(cuda.jax.ffi, "ffi_call", fake_ffi_call)

    def route(pmid, disp, vel, valid):
        result = cuda.route_merge_bidir_drift_primal_i16(
            pmid, disp, vel, valid, jnp.float32(0),
            jnp.zeros((1,), jnp.uint32), jnp.int32(1),
            jnp.zeros((1, 8), jnp.uint32), jnp.int32(0),
            jnp.zeros((2, 8), jnp.uint32), jnp.int32(0),
            disp_size=1, global_nmesh=16, mesh_shape=(16, 16, 16),
            owned_start=jnp.int32(0), owned_end=jnp.int32(4), slice_width=4,
            num_devices=4, record_capacity=1, capacity=pmid.shape[0],
            augmented=augmented, manual_axis_name=AXIS_NAME,
        )
        return jax.lax.cond(
            jnp.bool_(False),
            lambda _: (jnp.zeros_like(pmid), jnp.zeros_like(disp),
                       jnp.zeros_like(vel), jnp.zeros_like(valid)),
            lambda _: result[:4],
            operand=None,
        )

    mapped = jax.jit(shard_map(
        route, mesh=mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None),
                  P(AXIS_NAME, None), P(AXIS_NAME)),
        out_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None),
                   P(AXIS_NAME, None), P(AXIS_NAME)),
    ))
    result = mapped(jnp.zeros((8, 3), jnp.int16),
                    jnp.zeros((8, 3), jnp.float32),
                    jnp.zeros((8, 3), jnp.float32),
                    jnp.ones((8,), jnp.bool_))
    assert all(not np.any(np.asarray(value)) for value in result)
