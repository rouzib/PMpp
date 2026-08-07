import os
from pathlib import Path
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.nbody import Particles

REPO_ROOT = Path(__file__).resolve().parents[1]


def _single_conf(shape=(3, 2, 2), mesh_shape=2):
    return Configuration(0.75, shape, mesh_shape=mesh_shape, float_dtype=jnp.float32)


def _multi_conf(mode="mesh_halo", capacity=40):
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("this ownership test requires two GPUs")
    return Configuration(
        1.0, (4, 4, 4), mesh_shape=1,
        multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices[:2]), mode=mode),
        max_ptcl_per_slice=capacity, max_share_ptcl=16, max_halo_share_ptcl=32, max_share_gather_ptcl=16,
    )


def _lexsort_rows(values):
    values = np.asarray(values)
    return values[np.lexsort(tuple(values[:, axis] for axis in range(values.shape[1] - 1, -1, -1)))]


def test_particle_coordinate_conversions_preserve_subcell_information_and_periodicity():
    conf = _single_conf()
    box = np.asarray(conf.box_size, dtype=np.float32)
    pos = jnp.asarray([[-0.03, 0.11, 0.73], [0.38, 0.77, 1.51], [2.26, -0.01, 0.05], ], dtype=conf.float_dtype)

    unwrapped = Particles.from_pos(conf, pos, wrap=False)
    wrapped = Particles.from_pos_sharded(conf, pos, wrap=True)
    expected_wrapped = np.mod(np.asarray(pos), box)

    np.testing.assert_allclose(np.asarray(unwrapped.pos(dtype=conf.float_dtype, wrap=False)), pos, atol=2e-7)
    np.testing.assert_allclose(np.asarray(wrapped.pos(dtype=conf.float_dtype)), expected_wrapped, atol=2e-7)
    assert np.any(np.asarray(unwrapped.pmid) != np.asarray(wrapped.pmid))
    assert np.all(np.asarray(wrapped.pmid) >= 0)

    pmid, disp = Particles.pos_to_pmid(pos, conf)
    reconstructed = Particles.pmid_to_pos(pmid, disp, conf)
    np.testing.assert_allclose(np.asarray(reconstructed), expected_wrapped, atol=2e-7)
    np.testing.assert_array_equal(np.asarray(pmid), np.asarray(wrapped.pmid))
    np.testing.assert_allclose(np.asarray(disp), np.asarray(wrapped.disp), atol=1e-7)

    jacobian = jax.jacfwd(lambda x: Particles.from_pos(conf, x, wrap=False).pos(wrap=False))(pos)
    np.testing.assert_allclose(np.asarray(jacobian).reshape(pos.size, pos.size), np.eye(pos.size), atol=2e-6)


def test_grid_ids_indexing_attributes_and_ordered_initialization_are_exact():
    conf = _single_conf(shape=(2, 2, 2), mesh_shape=2)
    grid = Particles.gen_grid(conf, vel=True, acc=True)

    assert len(grid) == conf.ptcl_num
    assert grid.pmid.dtype == conf.pmid_dtype
    assert grid.disp.dtype == conf.float_dtype
    np.testing.assert_allclose(np.asarray(grid.vel), 0)
    np.testing.assert_allclose(np.asarray(grid.acc), 0)
    np.testing.assert_array_equal(np.asarray(grid.unused_index), False)
    np.testing.assert_array_equal(np.asarray(grid.halo_mask), False)
    expected_ids = np.ravel_multi_index(tuple(np.asarray(grid.pmid).T), conf.mesh_shape).astype(np.uint64)
    np.testing.assert_array_equal(np.asarray(grid.raveled_id()), expected_ids)

    first = grid[0]
    assert first.pmid.shape == (3, )
    np.testing.assert_array_equal(np.asarray(first.pmid), np.asarray(grid.pmid[0]))

    attr = {"mass": np.arange(conf.ptcl_num, dtype=np.float64), "tag": np.ones((conf.ptcl_num, 2))}
    attributed = Particles(conf, grid.pmid, grid.disp, attr=attr)
    assert attributed.attr["mass"].dtype == conf.float_dtype
    assert attributed.attr["tag"].dtype == conf.float_dtype
    np.testing.assert_array_equal(np.asarray(attributed[:2].attr["mass"]), [0, 1])

    shifted = grid.pos(dtype=conf.float_dtype).at[:, 0].add(jnp.float32(0.37))
    ordered = Particles.from_ordered_pos(conf, shifted, vel=jnp.ones_like(shifted), acc=2 * jnp.ones_like(shifted))
    np.testing.assert_array_equal(np.asarray(ordered.pmid), np.asarray(grid.pmid))
    np.testing.assert_allclose(np.asarray(ordered.pos(dtype=conf.float_dtype)), np.asarray(shifted), atol=2e-7)
    with pytest.raises(ValueError, match="full particle-grid ordered"):
        Particles.from_ordered_pos(conf, shifted[:-1])


def test_raveled_ids_and_particle_copy_wrap_only_when_requested():
    conf = _single_conf(shape=(2, 2, 2), mesh_shape=1)
    pmid = jnp.asarray([[-1, 0, 0], [0, 1, 1], [2, 0, 0]], dtype=conf.pmid_dtype)
    disp = jnp.zeros((3, 3), dtype=conf.float_dtype)
    original = Particles.from_pmid(conf, pmid, disp, vel=jnp.ones_like(disp), acc=2 * jnp.ones_like(disp))

    np.testing.assert_array_equal(np.asarray(original.raveled_id(dtype=jnp.int32, wrap=True)), [4, 3, 0])
    np.testing.assert_array_equal(np.asarray(original.raveled_id(dtype=jnp.int32, wrap=False)), [-4, 3, 8])

    copied = Particles.from_ptcl(original)
    explicit_unwrapped = Particles.from_ptcl(original, conf=conf, wrap=False)
    np.testing.assert_array_equal(np.asarray(copied.pmid), [[1, 0, 0], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(np.asarray(explicit_unwrapped.pmid), np.asarray(pmid))
    np.testing.assert_allclose(np.asarray(copied.vel), 1)
    np.testing.assert_allclose(np.asarray(copied.acc), 2)


def test_periodic_slice_and_halo_masks_match_hand_computed_boundaries():
    x = jnp.asarray([0.0, 0.9, 1.0, 2.9, 3.0, 3.9])
    non_wrapped = Particles.particles_in_slice_mask(x, 1, 3)
    wrapped = Particles.particles_in_slice_mask(x, 3, 1)
    np.testing.assert_array_equal(np.asarray(non_wrapped), [False, False, True, True, False, False])
    np.testing.assert_array_equal(np.asarray(wrapped), [True, True, False, False, True, True])

    unused = jnp.asarray([False, True, False, False, False, False])
    halo = Particles.compute_halo_mask(x, jnp.asarray([3, 1]), jnp.asarray([2, 3]), unused)
    np.testing.assert_array_equal(np.asarray(halo), [True, False, False, True, True, True])

    host_wrapped = Particles._host_particles_in_slice_mask(np.asarray(x), 3, 1)
    host_halo = Particles._host_compute_halo_mask(np.asarray(x), (3, 1), (2, 3), np.asarray(unused))
    np.testing.assert_array_equal(host_wrapped, np.asarray(wrapped))
    np.testing.assert_array_equal(host_halo, np.asarray(halo))


def test_distribute_particle_positions_handles_real_empty_and_default_vector_fields():
    conf = types.SimpleNamespace(
        multigpu=None, slice_start=jnp.asarray([0, 2]), slice_end=jnp.asarray([2, 3]), max_ptcl_per_slice=3,
        disp_size=1.0, nMesh=4, halo_start=jnp.asarray([[3, 0], [1, 2]]), halo_end=jnp.asarray([[1, 2], [3, 0]]),
    )
    pmid = jnp.asarray([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=jnp.int32)
    disp = jnp.asarray([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]], dtype=jnp.float32)

    left = Particles.distribute_ptcl_pos(pmid, disp, None, None, conf, 0)
    np.testing.assert_array_equal(np.asarray(left[0])[:2], np.asarray(pmid)[:2])
    np.testing.assert_allclose(np.asarray(left[2]), 0)
    np.testing.assert_allclose(np.asarray(left[3]), 0)
    np.testing.assert_array_equal(np.asarray(left[4]), [False, False, True])

    empty = Particles.distribute_ptcl_pos(pmid, disp, None, None, conf, 1)
    assert np.count_nonzero(np.asarray(empty[4]) == 0) == 1
    np.testing.assert_allclose(np.asarray(empty[0]), 0)
    np.testing.assert_allclose(np.asarray(empty[1]), 0)


def test_remove_and_add_particles_preserve_fields_and_stable_slot_order():
    pmid = jnp.asarray([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=jnp.int32)
    disp = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    vel = disp + 20
    acc = disp + 40
    unused = jnp.asarray([False, False, False, False])
    mask = jnp.asarray([False, True, False, True])

    removed = Particles.remove_particles(pmid, disp, vel, acc, mask, unused)
    for field in removed[:4]:
        np.testing.assert_allclose(np.asarray(field)[mask], 0)
    np.testing.assert_array_equal(np.asarray(removed[4]), [False, True, False, True])

    new_pmid = jnp.asarray([[9, 1, 1], [8, 2, 2], [7, 3, 3]], dtype=jnp.int32)
    new_disp = jnp.asarray([[0.9, 0, 0], [0.8, 0, 0], [0.7, 0, 0]], dtype=jnp.float32)
    new_vel = new_disp + 10
    new_acc = new_disp + 20
    added = Particles.add_particles(
        *removed[:4], removed[4], new_pmid, new_disp, new_vel, new_acc, jnp.asarray([True, False, True]), 3,
    )
    np.testing.assert_array_equal(np.asarray(added[0]), [[1, 0, 0], [9, 1, 1], [3, 0, 0], [7, 3, 3]])
    np.testing.assert_allclose(np.asarray(added[1])[[1, 3]], np.asarray(new_disp)[[0, 2]])
    np.testing.assert_allclose(np.asarray(added[2])[[1, 3]], np.asarray(new_vel)[[0, 2]])
    np.testing.assert_allclose(np.asarray(added[3])[[1, 3]], np.asarray(new_acc)[[0, 2]])
    np.testing.assert_array_equal(np.asarray(added[4]), False)


def _run_add_particles_overflow_worker():
    shape = (2, 3)
    pmid = jnp.zeros(shape, dtype=jnp.int32)
    vector = jnp.zeros(shape, dtype=jnp.float32)
    unused = jnp.asarray([False, True])
    new_pmid = jnp.asarray([[1, 0, 0], [2, 0, 0]], dtype=jnp.int32)
    new_vector = jnp.ones(shape, dtype=jnp.float32)

    try:
        Particles.add_particles(
            pmid, vector, vector, vector, unused, new_pmid, new_vector, new_vector, new_vector,
            jnp.asarray([True, True]), 2,
        )[0].block_until_ready()
    except Exception as exc:
        if "Exceeded max_amount_particles_per_slice" in str(exc):
            os._exit(0)
        raise
    raise AssertionError("expected particle insertion overflow to raise")


def test_add_particles_capacity_overflow_fails_closed():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    paths = [str(REPO_ROOT / "src"), str(REPO_ROOT), str(REPO_ROOT / "tests")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--add-overflow-worker"], cwd=REPO_ROOT, env=env,
                   check=True)


@pytest.mark.parametrize("mode, capacity", [("mesh_halo", 64), ("particle_halo", 64)])
def test_two_gpu_host_partition_preserves_authoritative_particle_state(mode, capacity):
    conf = _multi_conf(mode=mode, capacity=capacity)
    grid = Particles.gen_grid(Configuration(1.0, (4, 4, 4), mesh_shape=1))
    positions = grid.pos(dtype=jnp.float32).at[:, 0].add(
        jnp.linspace(jnp.float32(-0.45), jnp.float32(0.45), grid.pmid.shape[0], dtype=jnp.float32)
    )
    velocities = jnp.stack((jnp.arange(64), -jnp.arange(64), 2 * jnp.arange(64)), axis=1).astype(jnp.float32)
    accelerations = velocities * jnp.float32(0.125)

    partitioned = Particles.from_pos(conf, positions, vel=velocities, acc=accelerations)
    active = ~np.asarray(partitioned.unused_index)
    actual_pos = np.asarray(partitioned.pos(dtype=jnp.float32))[active]
    actual_vel = np.asarray(partitioned.vel)[active]
    actual_acc = np.asarray(partitioned.acc)[active]
    wrapped_positions = np.mod(np.asarray(positions), np.asarray(conf.box_size, dtype=np.float32))

    if mode == "mesh_halo":
        assert active.sum() == 64
        assert not np.any(np.asarray(partitioned.halo_mask))
        np.testing.assert_allclose(_lexsort_rows(actual_pos), _lexsort_rows(wrapped_positions), atol=5e-7)
    else:
        assert active.sum() == 96
        assert np.asarray(partitioned.halo_mask).sum() == 64
        assert np.any(np.asarray(partitioned.halo_mask))
        source_keys = {tuple(np.round(pos, 6)) for pos in wrapped_positions}
        assert {tuple(np.round(pos, 6)) for pos in actual_pos} == source_keys

    expected_by_x = {
        tuple(np.round(np.asarray(pos), 6)): (vel, acc)
        for pos, vel, acc in zip(wrapped_positions, np.asarray(velocities), np.asarray(accelerations))
    }
    for pos, vel, acc in zip(actual_pos, actual_vel, actual_acc):
        expected_vel, expected_acc = expected_by_x[tuple(np.round(pos, 6))]
        np.testing.assert_allclose(vel, expected_vel)
        np.testing.assert_allclose(acc, expected_acc)

    per_device_active = []
    for device_id in range(2):
        fields = partitioned.values_on_device(device_id)
        local_active = ~np.asarray(fields[4])
        local_x = np.asarray(Particles.pmid_to_pos(fields[0], fields[1], conf))[local_active, 0]
        if mode == "mesh_halo":
            start = int(np.asarray(conf.owned_slice_start)[device_id])
            end = int(np.asarray(conf.owned_slice_end)[device_id])
            assert np.all(np.asarray(Particles._host_particles_in_slice_mask(local_x, start, end)))
        else:
            start = int(np.asarray(conf.slice_start)[device_id])
            end = int(np.asarray(conf.slice_end)[device_id])
            assert np.all(np.asarray(Particles._host_particles_in_slice_mask(local_x, start, end)))
        per_device_active.append(local_active.sum())
    assert sum(per_device_active) == (64 if mode == "mesh_halo" else 96)


def test_multi_gpu_particle_constructors_share_one_exact_ownership_contract():
    conf = _multi_conf()
    base = Particles.gen_grid(Configuration(1.0, (4, 4, 4), mesh_shape=1), vel=True, acc=True)
    shifted_pmid = base.pmid.at[0, 0].set(-1).at[-1, 0].set(4)
    disp = base.disp.at[:, 0].add(jnp.float32(0.2))
    source = Particles(base.conf, shifted_pmid, disp, vel=base.vel + 3, acc=base.acc + 5)

    via_pmid = Particles.from_pmid(conf, shifted_pmid % 4, disp, vel=source.vel, acc=source.acc)
    via_particle = Particles.from_ptcl(source, conf=conf, wrap=True)
    via_ordered = Particles.from_ordered_pos(conf, source.pos(dtype=jnp.float32), vel=source.vel, acc=source.acc)

    for candidate in (via_pmid, via_particle, via_ordered):
        authoritative = ~np.asarray(candidate.unused_index) & ~np.asarray(candidate.halo_mask)
        assert authoritative.sum() == 64
        assert not np.any(np.asarray(candidate.halo_mask))
        np.testing.assert_allclose(np.asarray(candidate.vel)[authoritative], 3)
        np.testing.assert_allclose(np.asarray(candidate.acc)[authoritative], 5)


def test_host_partition_rejects_uninitialized_runtime_and_real_capacity_overflow():
    conf = _single_conf(shape=(2, 2, 2), mesh_shape=1)
    grid = Particles.gen_grid(conf)
    with pytest.raises(ValueError, match="initialized multi-GPU runtime"):
        Particles._partition_and_shard_particle_fields(conf, grid.pmid, grid.disp, None, None)

    multi = _multi_conf(capacity=20)
    positions = Particles.gen_grid(Configuration(1.0, (4, 4, 4), mesh_shape=1)).pos(dtype=jnp.float32)
    with pytest.raises(ValueError, match="Exceeded max_ptcl_per_slice"):
        Particles.from_pos(multi, positions)


if __name__ == "__main__":
    if sys.argv[1:] != ["--add-overflow-worker"]:
        raise SystemExit("expected --add-overflow-worker")
    _run_add_particles_overflow_worker()
