import numpy as np
import pytest

from pmpp.extras.quijote import (build_quijote_canonicalization, canonicalize_quijote_arrays, )


def _synthetic_shuffled_quijote(grid_size=4, seed=23):
    rng = np.random.default_rng(seed)
    count = grid_size**3
    box_size = 80.0
    a_start = 1.0 / 128.0
    expansion_rate = 11.5
    cell_size = box_size / grid_size

    grid = np.indices((grid_size, ) * 3, dtype=np.int64)
    q = np.moveaxis(grid, 0, -1).reshape((-1, 3))
    ids_by_q = rng.permutation(np.arange(1001, 1001 + count, dtype=np.int64))

    displacement = rng.uniform(-0.2, 0.2, size=(count, 3)) * cell_size
    residual = rng.uniform(-0.01, 0.01, size=(count, 3))
    velocity_factor = a_start * a_start * expansion_rate
    ic_vel_by_q = displacement * velocity_factor
    ic_pos_by_q = np.mod(q * cell_size + displacement + residual * cell_size, box_size)

    ic_shuffle = rng.permutation(count)
    target_shuffle = rng.permutation(count)
    target_pos_by_q = np.mod(q * cell_size + 0.37 * cell_size, box_size)
    target_vel_by_q = np.column_stack((q[:, 0] + 10.0, q[:, 1] + 20.0, q[:, 2] + 30.0))
    target_scalar_by_q = np.arange(count, dtype=np.float32)**2

    return {
        "box_size": box_size,
        "a_start": a_start,
        "expansion_rate": expansion_rate,
        "q": q,
        "ids_by_q": ids_by_q,
        "ic_ids": ids_by_q[ic_shuffle],
        "ic_pos": ic_pos_by_q[ic_shuffle],
        "ic_vel": ic_vel_by_q[ic_shuffle],
        "ic_pos_by_q": ic_pos_by_q,
        "ic_vel_by_q": ic_vel_by_q,
        "target_ids": ids_by_q[target_shuffle],
        "target_pos": target_pos_by_q[target_shuffle],
        "target_vel": target_vel_by_q[target_shuffle],
        "target_scalar": target_scalar_by_q[target_shuffle],
        "target_pos_by_q": target_pos_by_q,
        "target_vel_by_q": target_vel_by_q,
        "target_scalar_by_q": target_scalar_by_q,
    }


def test_canonicalizes_shuffled_ic_and_multiple_target_arrays():
    data = _synthetic_shuffled_quijote()
    canonicalization = build_quijote_canonicalization(
        data["ic_ids"], data["ic_pos"], data["ic_vel"], box_size=data["box_size"], a_start=data["a_start"],
        expansion_rate=data["expansion_rate"], q_tolerance=0.02,
    )

    np.testing.assert_array_equal(canonicalization.canonical_q_coordinates(dtype=np.int64), data["q"])
    assert canonicalization.grid_size == 4
    assert canonicalization.particle_count == 4**3
    assert canonicalization.max_q_residual <= 0.01 + 1e-12

    canonical_ic_pos, canonical_ic_vel = canonicalization.apply(data["ic_ids"], data["ic_pos"], data["ic_vel"])
    np.testing.assert_allclose(canonical_ic_pos, data["ic_pos_by_q"])
    np.testing.assert_allclose(canonical_ic_vel, data["ic_vel_by_q"])

    canonical_target_pos, canonical_target_vel, canonical_target_scalar = (
        canonicalize_quijote_arrays(
            canonicalization, data["target_ids"], data["target_pos"], data["target_vel"], data["target_scalar"],
        )
    )
    np.testing.assert_allclose(canonical_target_pos, data["target_pos_by_q"])
    np.testing.assert_allclose(canonical_target_vel, data["target_vel_by_q"])
    np.testing.assert_array_equal(canonical_target_scalar, data["target_scalar_by_q"])
    assert canonical_target_scalar.dtype == data["target_scalar"].dtype


def test_each_target_snapshot_may_have_a_different_raw_order():
    data = _synthetic_shuffled_quijote(seed=29)
    canonicalization = build_quijote_canonicalization(
        data["ic_ids"], data["ic_pos"], data["ic_vel"], box_size=data["box_size"], a_start=data["a_start"],
        expansion_rate=data["expansion_rate"],
    )
    second_shuffle = np.random.default_rng(31).permutation(4**3)
    second_ids = data["ids_by_q"][second_shuffle]
    second_target = data["target_pos_by_q"][second_shuffle]

    (first, ) = canonicalization.apply(data["target_ids"], data["target_pos"])
    (second, ) = canonicalization.apply(second_ids, second_target)

    np.testing.assert_array_equal(first, data["target_pos_by_q"])
    np.testing.assert_array_equal(second, data["target_pos_by_q"])


def test_rejects_duplicate_or_mismatched_particle_ids():
    data = _synthetic_shuffled_quijote()
    duplicate_ic_ids = data["ic_ids"].copy()
    duplicate_ic_ids[0] = duplicate_ic_ids[1]
    with pytest.raises(ValueError, match="must be unique"):
        build_quijote_canonicalization(
            duplicate_ic_ids, data["ic_pos"], data["ic_vel"], box_size=data["box_size"], a_start=data["a_start"],
            expansion_rate=data["expansion_rate"],
        )

    canonicalization = build_quijote_canonicalization(
        data["ic_ids"], data["ic_pos"], data["ic_vel"], box_size=data["box_size"], a_start=data["a_start"],
        expansion_rate=data["expansion_rate"],
    )
    mismatched_ids = data["target_ids"].copy()
    mismatched_ids[0] = 999_999
    with pytest.raises(ValueError, match="ID set does not match"):
        canonicalization.apply(mismatched_ids, data["target_pos"])


def test_rejects_q_reconstruction_outside_tolerance():
    grid_size = 2
    count = grid_size**3
    q = np.moveaxis(np.indices((grid_size, ) * 3, dtype=np.float64), 0, -1).reshape((-1, 3))
    pos = q.copy()
    pos[3, 1] += 0.08

    with pytest.raises(ValueError, match="exceeds q_tolerance"):
        build_quijote_canonicalization(
            np.arange(count, dtype=np.int64), pos, np.zeros_like(pos), box_size=float(grid_size), a_start=1.0,
            expansion_rate=1.0, q_tolerance=0.05,
        )


def test_rejects_non_bijective_reconstructed_q_grid():
    grid_size = 2
    count = grid_size**3
    q = np.moveaxis(np.indices((grid_size, ) * 3, dtype=np.float64), 0, -1).reshape((-1, 3))
    pos = q.copy()
    pos[1] = pos[0]

    with pytest.raises(ValueError, match="not a bijection"):
        build_quijote_canonicalization(
            np.arange(count, dtype=np.int64), pos, np.zeros_like(pos), box_size=float(grid_size), a_start=1.0,
            expansion_rate=1.0, q_tolerance=0.01,
        )
