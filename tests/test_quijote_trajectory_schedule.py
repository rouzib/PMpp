from pathlib import Path

import numpy as np
import pytest

import scripts.train_quijote_trajectory_corrector as trajectory_trainer

from scripts.train_quijote_trajectory_corrector import (
    DEFAULT_CHECKPOINT_REDSHIFTS, DEFAULT_MESH_SHAPES, CheckpointScore, CURRICULUM_STAGES, GateThresholds,
    RESOLUTION_CURRICULUM, _run_nbody_segments, _run_multiple_shooting_segments, _validate_training_args,
    aggregate_checkpoint_scores, alternating_mesh_shape, allocate_interval_steps, build_checkpoint_metadata,
    build_arg_parser, build_checkpoint_schedule, build_source_content_manifest, build_split_manifest,
    build_shared_growth_grid, checkpoint_score, collect_code_provenance, correction_differentiation_metadata,
    curriculum_stage_for_update, differentiable_upper_tail_cvar, gate_flags, initialize_lexicographic_selection,
    load_curriculum_transfer, load_warm_start, measure_position_quantization_floor, normalize_checkpoint_weights,
    resolution_curriculum_orchestration, resolve_mesh_shapes, validate_checkpoint_compatibility,
    validate_curriculum_transfer_compatibility, worst_shell_cross_objective,
)


def test_constrained_step_allocation_matches_known_quijote_case():
    checkpoint_a = (0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)

    allocation = allocate_interval_steps(1.0 / 128.0, checkpoint_a, 20)

    assert allocation == (5, 2, 3, 3, 7)
    assert sum(allocation) == 20
    assert all(count >= 1 for count in allocation)


def test_default_schedule_has_exact_checkpoint_knots_and_total_steps():
    schedule = build_checkpoint_schedule(1.0 / 128.0, DEFAULT_CHECKPOINT_REDSHIFTS, total_steps=63, )
    expected_checkpoints = np.asarray((0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0))

    assert schedule.interval_steps == (15, 5, 11, 11, 21)
    assert schedule.total_steps == 63
    assert schedule.global_scale_factors.shape == (64, )
    assert np.all(np.diff(schedule.global_scale_factors) > 0)
    np.testing.assert_array_equal(
        schedule.global_scale_factors[np.asarray(schedule.checkpoint_indices)], expected_checkpoints,
    )

    reconstructed = np.concatenate([
        segment if index == 0 else segment[1:] for index, segment in enumerate(schedule.segment_scale_factors)
    ])
    np.testing.assert_array_equal(reconstructed, schedule.global_scale_factors)
    for index, segment in enumerate(schedule.segment_scale_factors):
        assert segment.size == schedule.interval_steps[index] + 1
        assert segment[-1] == schedule.scale_factors[index]
        if index:
            assert segment[0] == schedule.scale_factors[index - 1]


def test_shared_growth_grid_preserves_lpt_prefix_and_global_schedule():
    schedule = build_checkpoint_schedule(0.01, (3.0, 1.0, 0.0), total_steps=9)
    lpt = np.asarray((0.0, 0.005, 0.01))

    growth = build_shared_growth_grid(0.01, schedule.global_scale_factors, lpt, )

    np.testing.assert_array_equal(growth[:3], lpt)
    np.testing.assert_array_equal(growth[2:], schedule.global_scale_factors)
    assert np.all(np.diff(growth) > 0)

    early_schedule = build_checkpoint_schedule(0.01, (3.0, 1.0), total_steps=4)
    early_growth = build_shared_growth_grid(0.01, early_schedule.global_scale_factors, )
    assert early_growth[-1] == 1.0


def test_checkpoint_weights_are_normalized_and_validated():
    assert normalize_checkpoint_weights(None, 2) == (0.5, 0.5)
    np.testing.assert_allclose(normalize_checkpoint_weights((1.0, 2.0, 1.0), 3), (0.25, 0.5, 0.25), )
    with pytest.raises(ValueError, match="Expected 3"):
        normalize_checkpoint_weights((1.0, 2.0), 3)
    with pytest.raises(ValueError, match="positive"):
        normalize_checkpoint_weights((0.0, 0.0), 2)


def test_schedule_rejects_impossible_or_out_of_order_inputs():
    with pytest.raises(ValueError, match="at least one step"):
        allocate_interval_steps(0.01, (0.2, 0.5, 1.0), 2)
    with pytest.raises(ValueError, match="strictly decreasing"):
        build_checkpoint_schedule(0.01, (1.0, 2.0), total_steps=4)
    with pytest.raises(ValueError, match="strictly increasing"):
        allocate_interval_steps(0.25, (0.25, 1.0), 4)


def test_segment_runner_reuses_same_correction_and_preserves_checkpoints():
    calls = []
    correction = object()

    def fake_nbody(state, cosmo, conf, correction=None):
        calls.append((cosmo, conf, correction))
        return state + conf

    states = _run_nbody_segments(fake_nbody, 0, "cosmology", (1, 2, 3), correction, require_correction=True, )

    assert states == (1, 3, 6)
    assert [call[1] for call in calls] == [1, 2, 3]
    assert all(call[2] is correction for call in calls)
    with pytest.raises(ValueError, match="correction is required"):
        _run_nbody_segments(fake_nbody, 0, "cosmology", (1, ), None, require_correction=True, )


def test_phase_segment_runner_can_build_checkpoint_without_global_jax_binding():
    from collections import namedtuple

    PhaseCorrection = namedtuple("PhaseCorrection", ("phase_space", ))
    correction = PhaseCorrection(phase_space=1.0)

    def fake_nbody(state, cosmo, conf, correction=None):
        del cosmo
        assert correction is not None
        return state + conf

    states = _run_nbody_segments(fake_nbody, 0.0, 1.0, (1.0, 2.0), correction, require_correction=True, )

    np.testing.assert_allclose(np.asarray(states), (1.0, 3.0))


def test_snapshot_multiple_shooting_stages_use_true_starts_then_full_rollout():
    correction = object()

    def fake_nbody(state, cosmo, conf, correction=None):
        assert cosmo == "cosmology"
        assert correction is not None
        return state + conf

    common = dict(
        nbody=fake_nbody, initial_particles=0, teacher_starts=(0, 10, 20, 30, 40), cosmo="cosmology",
        segment_confs=(1, 2, 3, 4, 5), correction=correction,
    )
    assert _run_multiple_shooting_segments(**common, stage=CURRICULUM_STAGES[0]) == (1, 12, 23, 34, 45)
    assert _run_multiple_shooting_segments(**common, stage=CURRICULUM_STAGES[1]) == (1, 3, 23, 27, 45)
    assert _run_multiple_shooting_segments(**common, stage=CURRICULUM_STAGES[2]) == (1, 3, 6, 10, 15)

    assert [curriculum_stage_for_update(step, 9) for step in range(9)
            ] == [*([CURRICULUM_STAGES[0]] * 3), *([CURRICULUM_STAGES[1]] * 3), *([CURRICULUM_STAGES[2]] * 3), ]


def test_parser_defaults_to_safe_multi_checkpoint_phase_aware_training():
    args = build_arg_parser().parse_args([])

    assert args.checkpoint_redshifts == DEFAULT_CHECKPOINT_REDSHIFTS
    assert args.total_pm_steps == 63
    assert args.correction_model == "hybrid_quijote"
    assert args.mesh_shape is None
    assert args.mesh_shapes == DEFAULT_MESH_SHAPES
    assert args.analysis_grid == 256
    assert args.cross_gate == 0.999
    assert args.position_rmse_gate_cells == 0.01
    assert args.position_p99_gate_cells == 0.05
    assert args.curriculum_transfer is None
    assert args.source_hash_cache.name == "quijote_source_hash_cache.json"
    assert args.refresh_source_hashes is False
    assert args.local_channels == 32
    assert args.local_cutoff_cells == 2.5
    assert args.phase_max_displacement_cells == 0.25
    assert args.particle_huber_weight > 0
    assert args.density_weight > 0
    assert args.power_weight > 0
    assert args.velocity_weight > 0
    assert args.projection_weight > 0
    assert args.cross_weight > 0


def test_curriculum_transfer_cli_cannot_weaken_or_mix_warm_start_modes():
    parser = build_arg_parser()
    both = parser.parse_args(["--initial-correction", "strict.pkl", "--curriculum-transfer", "parent.pkl"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_training_args(both)

    weakened = parser.parse_args(["--curriculum-transfer", "parent.pkl", "--allow-legacy-warm-start"])
    with pytest.raises(ValueError, match="Legacy warm starts are forbidden"):
        _validate_training_args(weakened)


def test_mesh_shape_resolution_preserves_legacy_override_and_alternates():
    assert resolve_mesh_shapes(None, None) == (1.0, 2.0)
    assert resolve_mesh_shapes(2.0, (1.0, 2.0)) == (2.0, )
    assert [alternating_mesh_shape((1.0, 2.0), step) for step in range(5)] == [1.0, 2.0, 1.0, 2.0, 1.0, ]
    with pytest.raises(ValueError, match="supports mesh ratios 1 and 2"):
        resolve_mesh_shapes(None, (1.0, 3.0))


def test_split_manifest_reserves_blind_realization_without_overlap(tmp_path):
    manifest = build_split_manifest("0", tmp_path / "0", "1", tmp_path / "1")

    assert manifest["training"][0]["accessed"] is True
    assert manifest["blind"]["accessed"] is False
    assert manifest["blind"]["status"] == "reserved_unseen"
    with pytest.raises(ValueError, match="IDs must be different"):
        build_split_manifest("0", tmp_path / "0", "0", tmp_path / "1")
    with pytest.raises(ValueError, match="paths must be different"):
        build_split_manifest("0", tmp_path / "same", "1", tmp_path / "same")


def test_worst_shell_cross_and_position_tail_objectives_are_gate_oriented():
    phase, hinge, minimum = worst_shell_cross_objective(
        np, np.asarray([0.9995, 0.9980, 0.2]), np.asarray([True, True, False]), threshold=0.999,
    )

    assert minimum == pytest.approx(0.998)
    assert phase == pytest.approx(0.002)
    assert hinge == pytest.approx(0.001)
    values = np.arange(1.0, 101.0)
    assert differentiable_upper_tail_cvar(np, values, 0.02) == pytest.approx(99.5)


def test_checkpoint_ranking_is_cross_then_rmse_then_p99():
    strong_cross = CheckpointScore(0.9995, 0.02, 0.08)
    weak_cross = CheckpointScore(0.9994, 0.001, 0.002)
    tied_cross_better_rmse = CheckpointScore(0.9995, 0.01, 0.09)
    tied_cross_rmse_better_p99 = CheckpointScore(0.9995, 0.01, 0.05)

    assert strong_cross.rank_key < weak_cross.rank_key
    assert tied_cross_better_rmse.rank_key < strong_cross.rank_key
    assert tied_cross_rmse_better_p99.rank_key < tied_cross_better_rmse.rank_key
    aggregate = aggregate_checkpoint_scores([strong_cross, CheckpointScore(0.9991, 0.03, 0.04)])
    assert aggregate == CheckpointScore(0.9991, 0.03, 0.08)
    assert checkpoint_score({
        "selection/min_cross_correlation": 0.999,
        "selection/max_position_rmse_cells": 0.01,
        "selection/max_position_p99_cells": 0.05,
    }) == CheckpointScore(0.999, 0.01, 0.05)


def test_checkpoint_metadata_rejects_any_compatibility_mismatch(tmp_path):
    manifest = build_split_manifest("0", tmp_path / "0", "1", tmp_path / "1")
    metadata = build_checkpoint_metadata(
        architecture={
            "model": "hybrid_quijote",
            "channels": 32
        }, pytree={
            "class": "test.Correction",
            "leaves": []
        }, normalization={"position_units": "particle_cells"}, mesh_shapes=(1.0, 2.0), data_manifest=manifest,
        code_provenance={
            "critical_source_sha256": "abc",
            "git_revision": "123"
        }, cosmology={
            "sigma8": 0.834,
            "Omega_m": 0.3175
        }, solver_protocol={"total_pm_steps": 63},
    )
    checkpoint = {"correction": {"weight": 1.0}, "metadata": metadata}

    validate_checkpoint_compatibility(checkpoint, metadata)
    path = tmp_path / "checkpoint.pkl"
    import pickle

    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    assert load_warm_start(path, metadata) == {"weight": 1.0}

    incompatible = dict(metadata)
    incompatible["compatibility"] = dict(metadata["compatibility"])
    incompatible["compatibility"]["mesh_shapes"] = [1.0]
    with pytest.raises(ValueError, match="mesh_shapes"):
        validate_checkpoint_compatibility(checkpoint, incompatible)

    incompatible_cosmology = dict(metadata)
    incompatible_cosmology["compatibility"] = dict(metadata["compatibility"])
    incompatible_cosmology["compatibility"]["cosmology"] = {"sigma8": 0.9, "Omega_m": 0.3175, }
    with pytest.raises(ValueError, match="cosmology"):
        validate_checkpoint_compatibility(checkpoint, incompatible_cosmology)

    bare = tmp_path / "bare.pkl"
    with bare.open("wb") as handle:
        pickle.dump({"weight": 1.0}, handle)
    with pytest.raises(ValueError, match="Legacy bare correction"):
        load_warm_start(bare, metadata)


def test_machine_readable_gate_flags_require_cross_positions_mass_and_capacity():
    thresholds = GateThresholds()
    passing = gate_flags(
        min_cross_correlation=0.999, position_rmse_cells=0.01, position_p99_cells=0.05, mass_relative_error=1e-6,
        capacity_overflow_detected=False, thresholds=thresholds,
    )
    assert passing["cross_plus_positions_pass"] is True
    assert passing["run_valid"] is True

    overflow = gate_flags(
        min_cross_correlation=1.0, position_rmse_cells=0.0, position_p99_cells=0.0, mass_relative_error=0.0,
        capacity_overflow_detected=True, thresholds=thresholds,
    )
    assert overflow["cross_plus_positions_pass"] is True
    assert overflow["capacity_overflow_pass"] is False
    assert overflow["run_valid"] is False


def _curriculum_metadata(tmp_path, resolution, *, source_hash="shared-source"):
    manifest = build_split_manifest("0", tmp_path / "0", "1", tmp_path / "1")
    manifest["canonical_particle_ordering_version"] = ("quijote-gadget-id-to-lagrangian-grid-v1")
    manifest["source_content"] = {"aggregate_sha256": source_hash}
    manifest["training"][0].update({
        "ic_prefix": str(tmp_path / "0/ICs/ics"),
        "particle_resolution": resolution,
        "source_particle_resolution": 256,
        "downsample_factor": 256 // resolution,
        "box_size_mpc_h": 1000.0,
        "initial_redshift": 127.0,
        "hubble_parameter": 0.6711,
        "snapshot_redshifts": [3.0, 2.0, 1.0, 0.5, 0.0],
        "snapshot_sources": {
            "z3": "snap_001",
            "z0": "snap_004"
        },
    })
    metadata = build_checkpoint_metadata(
        architecture={
            "model": "hybrid_quijote",
            "channels": 32
        }, pytree={
            "class": "test.Correction",
            "leaves": [{
                "shape": [2]
            }]
        }, normalization={
            "position_units": "particle_cells",
            "particle_spacing_mpc_h": 1000.0 / resolution,
            "cross_band": {
                "k_min_h_mpc": 0.0,
                "k_max_h_mpc": resolution / 1000
            },
            "gate_thresholds": {
                "cross": 0.999
            },
        }, mesh_shapes=(1.0, 2.0), data_manifest=manifest, code_provenance={
            "critical_source_sha256": "code-hash",
            "git_revision": "abc"
        }, cosmology={
            "sigma8": 0.834,
            "n_s": 0.9624,
            "Omega_m": 0.3175,
            "Omega_b": 0.049,
            "h": 0.6711,
        }, solver_protocol={
            "total_pm_steps": 63,
            "checkpoint_redshifts": [3.0, 2.0, 1.0, 0.5, 0.0],
        },
    )
    if resolution >= 128:
        metadata["lineage"] = {
            "from_resolution": 64,
            "to_resolution": 128,
            "parent_best_checkpoint_certified": True,
            "parent_training_run_complete": True,
            "ancestor_lineage": None,
        }
    return metadata


def test_resolution_curriculum_transfer_is_explicit_adjacent_and_strict(tmp_path):
    assert RESOLUTION_CURRICULUM == (64, 128, 256)
    metadata_64 = _curriculum_metadata(tmp_path, 64)
    metadata_128 = _curriculum_metadata(tmp_path, 128)
    metadata_256 = _curriculum_metadata(tmp_path, 256)
    checkpoint = {
        "correction": {
            "weight": np.asarray([1.0, 2.0])
        },
        "metadata": metadata_64,
        "completed_updates": 9,
        "training_run_completed_updates": 9,
        "training_run_complete": True,
        "best_checkpoint_certified": True,
        "multiple_shooting_stage_update_counts": {
            "teacher_forced_single_intervals": 3,
            "teacher_forced_consecutive_pairs": 3,
            "full_ic_to_z0_rollout": 3,
        },
        "checkpoint_kind": "post_update_lexicographic_evaluation",
        "selection_curriculum_stage": "full_ic_to_z0_rollout",
        "selection_score": {
            "min_cross_correlation": 0.99,
            "max_position_rmse_cells": 0.1,
            "max_position_p99_cells": 0.2,
        },
    }

    transition = validate_curriculum_transfer_compatibility(checkpoint, metadata_128)
    assert transition["from_resolution"] == 64
    assert transition["to_resolution"] == 128
    with pytest.raises(ValueError, match="Warm-start checkpoint is incompatible"):
        validate_checkpoint_compatibility(checkpoint, metadata_128)
    with pytest.raises(ValueError, match="next declared resolution stage"):
        validate_curriculum_transfer_compatibility(checkpoint, metadata_256)

    different_source = _curriculum_metadata(tmp_path, 128, source_hash="other-source")
    with pytest.raises(ValueError, match="source_content_sha256"):
        validate_curriculum_transfer_compatibility(checkpoint, different_source)


def test_curriculum_loader_records_parent_provenance(tmp_path):
    import pickle

    parent = tmp_path / "parent.pkl"
    checkpoint = {
        "correction": {
            "weight": np.asarray([3.0])
        },
        "metadata": _curriculum_metadata(tmp_path, 128),
        "completed_updates": 9,
        "training_run_completed_updates": 9,
        "training_run_complete": True,
        "best_checkpoint_certified": True,
        "multiple_shooting_stage_update_counts": {
            "teacher_forced_single_intervals": 3,
            "teacher_forced_consecutive_pairs": 3,
            "full_ic_to_z0_rollout": 3,
        },
        "checkpoint_kind": "post_update_lexicographic_evaluation",
        "selection_curriculum_stage": "full_ic_to_z0_rollout",
        "selection_score": {
            "min_cross_correlation": 0.99,
            "max_position_rmse_cells": 0.1,
            "max_position_p99_cells": 0.2,
        },
    }
    with parent.open("wb") as handle:
        pickle.dump(checkpoint, handle)

    correction, lineage = load_curriculum_transfer(parent, _curriculum_metadata(tmp_path, 256))
    assert np.array_equal(correction["weight"], np.asarray([3.0]))
    assert lineage["from_resolution"] == 128
    assert lineage["to_resolution"] == 256
    assert lineage["parent_completed_updates"] == 9
    assert len(lineage["parent_checkpoint_sha256"]) == 64

    handoff = resolution_curriculum_orchestration(
        128, checkpoint_dir=tmp_path / "stage128/checkpoints", source_hash_cache=tmp_path / "source-hashes.json",
        lineage=lineage,
    )
    assert handoff["next_stage"]["resolution"] == 256
    cli = handoff["next_stage"]["required_cli_arguments"]
    assert cli[:2] == ["--downsample-res", "256"]
    assert "--curriculum-transfer" in cli
    assert "--source-hash-cache" in cli


def test_source_content_hash_cache_streams_each_unchanged_file_once(tmp_path, monkeypatch):
    ic = tmp_path / "ics.0"
    snapshot = tmp_path / "snap_001.0"
    ic.write_bytes(b"initial conditions")
    snapshot.write_bytes(b"snapshot")
    trajectory = {
        "ic_prefix": str(tmp_path / "ics"),
        "snapshots": {
            3.0: {
                "snapshot_prefix": str(tmp_path / "snap_001")
            }
        },
    }
    calls = []
    original = trajectory_trainer._sha256_file

    def counted(path, **kwargs):
        calls.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(trajectory_trainer, "_sha256_file", counted)
    cache = tmp_path / "hash-cache.json"
    first = build_source_content_manifest(trajectory, (3.0, ), cache_path=cache)
    second = build_source_content_manifest(trajectory, (3.0, ), cache_path=cache)
    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    assert len(calls) == 2

    snapshot.write_bytes(b"changed snapshot contents")
    third = build_source_content_manifest(trajectory, (3.0, ), cache_path=cache)
    assert third["aggregate_sha256"] != first["aggregate_sha256"]
    assert len(calls) == 3


def test_position_quantization_floor_and_differentiation_metadata_are_explicit():
    positions = np.asarray([[0.25, 1.0, 10.0], [100.0, 500.0, 999.0]], dtype=np.float32)
    floor = measure_position_quantization_floor(positions, particle_cell_size=4.0)
    assert floor["loaded_dtype"] == "float32"
    assert floor["coordinate_ulp_mpc_h"]["maximum"] > 0
    assert floor["rounding_floor_particle_cells"]["rmse"] > 0

    class ForceOnly:
        phase_space = None

    class DirectPhase:
        phase_space = object()

    assert correction_differentiation_metadata(ForceOnly())["all_segments_use_public_nbody_custom_vjp"] is True
    phase = correction_differentiation_metadata(DirectPhase())
    assert phase["all_segments_use_public_nbody_custom_vjp"] is False
    assert "rematerialization" in phase["differentiation_path"]


def test_initial_model_is_a_lexicographic_checkpoint_candidate():
    calls = []

    def evaluator(correction):
        calls.append(correction)
        return CheckpointScore(0.9995, 0.02, 0.04), {"mesh_1": {"ok": True}}

    selection = initialize_lexicographic_selection({"weight": 1.0}, evaluator, lambda value: dict(value))
    score, step, kind, parameters, metrics = selection
    assert len(calls) == 1
    assert score == CheckpointScore(0.9995, 0.02, 0.04)
    assert step == 0
    assert "initial" in kind
    assert parameters == {"weight": 1.0}
    assert metrics["mesh_1"]["ok"] is True


def test_code_provenance_covers_solver_correction_and_data_loader_sources():
    provenance = collect_code_provenance()
    files = set(provenance["critical_source_files"])
    assert "scripts/train_quijote_potential_corrector.py" in files
    assert "src/pmpp/nbody/integrator.py" in files
    assert "src/pmpp/cic/scatter.py" in files
    assert "src/pmpp/cic/gather.py" in files
    assert "src/pmpp/corrections/window.py" in files
    assert len(provenance["critical_source_sha256"]) == 64
