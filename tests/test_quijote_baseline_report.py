"""Lightweight tests for the held-out QUIJOTE baseline/report harness."""

from __future__ import annotations

import copy
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_quijote_baselines import (
    CapacityConfiguration, GateThresholds, ProtocolConfiguration, _run_checkpoint_trajectory, _timed_full_jit,
    _pmpp_failure_record, _stable_mapping_hash, add_correction_runtime_overhead, build_evaluation_contract,
    build_checkpoint_schedule, finalize_report_summary, fastpm_input_content_hashes, invoke_official_fastpm,
    load_official_fastpm_outputs, parse_ratio_path, plot_position_cdf, plot_spectra, plot_xyz_projections,
    summarize_case, summarize_family, validate_learned_checkpoint_for_blind_evaluation,
    verify_official_fastpm_provenance, write_json,
)


def _snapshot_record(
    *, cross_pass: bool = True, positions_pass: bool = True, power_pass: bool = False, projection_pass: bool = False,
):
    return {
        "status": "evaluated",
        "acceptance": {
            "min_r": 0.9995 if cross_pass else 0.998,
            "position_rmse_cells": 0.009 if positions_pass else 0.02,
            "position_p99_cells": 0.04 if positions_pass else 0.08,
            "cross_correlation_pass": cross_pass,
            "particle_positions_pass": positions_pass,
            "passed": cross_pass and positions_pass,
        },
        "auxiliary_non_gating": {
            "power": {
                "target_pass": power_pass,
                "gating": False
            },
            "projection_pearson": {
                "target_pass": projection_pass,
                "gating": False,
            },
        },
    }


def _all_snapshots(**kwargs):
    return {key: _snapshot_record(**kwargs) for key in ("3", "2", "1", "0.5", "0")}


def _healthy_case(**kwargs):
    return {"snapshots": _all_snapshots(**kwargs), "health": {"run_valid": True}, }


def _learned_checkpoint(tmp_path):
    training_path = (tmp_path / "0").resolve()
    blind_path = (tmp_path / "1").resolve()
    data_manifest = {
        "canonical_particle_ordering_version": ("quijote-gadget-id-to-lagrangian-grid-v1"),
        "training": [{
            "realization_id": "0",
            "path": str(training_path),
            "role": "training",
            "source_particle_resolution": 256,
            "particle_resolution": 256,
            "downsample_factor": 1,
        }],
        "blind": {
            "realization_id": "1",
            "path": str(blind_path),
            "role": "blind_test",
        },
        "source_content": {
            "aggregate_sha256": "a" * 64
        },
    }
    spacing = 1000.0 / 256.0
    schedule = build_checkpoint_schedule(1.0 / 128.0, ProtocolConfiguration())
    cosmology = {"sigma8": 0.834, "n_s": 0.9624, "Omega_m": 0.3175, "Omega_b": 0.049, "h": 0.6711, }
    metadata = {
        "schema_version": 2,
        "architecture": {
            "correction_model": "hybrid_quijote",
            "components": {
                "long_range": {},
                "local_pair": {},
                "phase_space": {},
            },
        },
        "pytree": {
            "class": "pmpp.corrections.nbody.NBodyCorrection"
        },
        "normalization": {
            "position_units": "particle_cells",
            "periodic_distance": "minimum_image",
            "particle_spacing_mpc_h": spacing,
            "gate_thresholds": {
                "min_cross_correlation": 0.999,
                "position_rmse_cells": 0.01,
                "position_p99_cells": 0.05,
            },
            "cross_band": {
                "k_min_h_mpc": 0.0,
                "k_max_h_mpc": 0.5 * np.pi / spacing,
            },
        },
        "mesh_shapes": [1.0, 2.0],
        "solver_protocol": {
            "checkpoint_redshifts": [3.0, 2.0, 1.0, 0.5, 0.0],
            "checkpoint_scale_factors": list(schedule.scale_factors),
            "interval_steps": list(schedule.interval_steps),
            "total_pm_steps": 63,
            "global_scale_factors": schedule.global_scale_factors.tolist(),
            "a_start": float(schedule.global_scale_factors[0]),
            "analysis_grid": 256,
        },
        "cosmology": cosmology,
        "lineage": {
            "from_resolution": 128,
            "to_resolution": 256,
            "parent_best_checkpoint_certified": True,
            "parent_training_run_complete": True,
            "ancestor_lineage": {
                "from_resolution": 64,
                "to_resolution": 128,
                "parent_best_checkpoint_certified": True,
                "parent_training_run_complete": True,
            },
        },
        "data_manifest": data_manifest,
        "compatibility": {
            "data_manifest_sha256": _stable_mapping_hash(data_manifest),
            "critical_source_sha256": "source-hash",
        },
    }
    per_mesh_metrics = {
        f"z{redshift:g}".replace(".", "p"): {
            "min_cross_correlation": 0.9991,
            "position_rmse_cells": 0.009,
            "position_p99_cells": 0.04,
        }
        for redshift in (3.0, 2.0, 1.0, 0.5, 0.0)
    }
    serialized_metrics = {
        mesh: {
            f"{slug}/{name}": value
            for slug, values in per_mesh_metrics.items()
            for name, value in values.items()
        }
        for mesh in ("mesh_1", "mesh_2")
    }
    return {
        "schema_version": 2,
        "completed_updates": 10,
        "training_run_completed_updates": 10,
        "training_run_complete": True,
        "best_checkpoint_certified": True,
        "multiple_shooting_stage_update_counts": {
            "teacher_forced_single_intervals": 4,
            "teacher_forced_consecutive_pairs": 3,
            "full_ic_to_z0_rollout": 3,
        },
        "checkpoint_kind": "post_update_lexicographic_evaluation",
        "selection_curriculum_stage": "full_ic_to_z0_rollout",
        "selection_score": {
            "min_cross_correlation": 0.9991,
            "max_position_rmse_cells": 0.009,
            "max_position_p99_cells": 0.04,
        },
        "selection_rank_key": [-0.9991, 0.009, 0.04],
        "metrics": serialized_metrics,
        "correction": object(),
        "metadata": metadata,
    }, spacing, training_path, blind_path


def _blind_validation_kwargs(spacing, blind_path):
    return {
        "evaluation_realization_id": "1",
        "evaluation_path": blind_path,
        "particle_spacing": spacing,
        "evaluation_source_content_sha256": "b" * 64,
        "evaluation_schedule": build_checkpoint_schedule(1.0 / 128.0, ProtocolConfiguration()),
        "evaluation_cosmology": {
            "sigma8": 0.834,
            "n_s": 0.9624,
            "Omega_m": 0.3175,
            "Omega_b": 0.049,
            "h": 0.6711,
        },
    }


def test_blind_evaluation_accepts_update_zero_selected_after_completed_training(tmp_path, ):
    checkpoint, spacing, _, blind_path = _learned_checkpoint(tmp_path)
    checkpoint["completed_updates"] = 0
    checkpoint["checkpoint_kind"] = "initial_lexicographic_evaluation"
    checkpoint["training_run_completed_updates"] = 10
    checkpoint["training_run_complete"] = True

    provenance = validate_learned_checkpoint_for_blind_evaluation(
        checkpoint, **_blind_validation_kwargs(spacing, blind_path),
    )

    assert provenance["completed_updates"] == 0
    assert provenance["training_run_completed_updates"] == 10

    del checkpoint["training_run_completed_updates"]
    del checkpoint["training_run_complete"]
    with pytest.raises(ValueError, match="completed training run"):
        validate_learned_checkpoint_for_blind_evaluation(checkpoint, **_blind_validation_kwargs(spacing, blind_path), )


def test_blind_evaluation_rejects_uncertified_or_nonprotocol_checkpoints(tmp_path):
    checkpoint, spacing, _, blind_path = _learned_checkpoint(tmp_path)
    kwargs = _blind_validation_kwargs(spacing, blind_path)

    periodic = copy.deepcopy(checkpoint)
    periodic["checkpoint_kind"] = "periodic_post_update"
    with pytest.raises(ValueError, match="periodic and final"):
        validate_learned_checkpoint_for_blind_evaluation(periodic, **kwargs)

    incomplete_score = copy.deepcopy(checkpoint)
    incomplete_score["selection_score"] = {}
    with pytest.raises(ValueError, match="exact lexicographic"):
        validate_learned_checkpoint_for_blind_evaluation(incomplete_score, **kwargs)

    wrong_protocol = copy.deepcopy(checkpoint)
    wrong_protocol["metadata"]["solver_protocol"]["total_pm_steps"] = 5
    with pytest.raises(ValueError, match="63-step protocol"):
        validate_learned_checkpoint_for_blind_evaluation(wrong_protocol, **kwargs)

    copied_training_data = dict(kwargs)
    copied_training_data["evaluation_source_content_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="identical"):
        validate_learned_checkpoint_for_blind_evaluation(checkpoint, **copied_training_data)


def test_protocol_is_fixed_and_schedule_contains_all_endpoints():
    protocol = ProtocolConfiguration()
    schedule = build_checkpoint_schedule(1.0 / 128.0, protocol)

    assert schedule.total_steps == 63
    assert schedule.global_scale_factors.shape == (64, )
    assert all(step_count >= 1 for step_count in schedule.interval_steps)
    assert np.array_equal(
        schedule.global_scale_factors[np.asarray(schedule.checkpoint_indices)],
        np.asarray((0.25, 1 / 3, 0.5, 2 / 3, 1.0)),
    )

    with pytest.raises(ValueError, match="63"):
        ProtocolConfiguration(total_steps=62)
    with pytest.raises(ValueError, match="mesh ratios"):
        ProtocolConfiguration(mesh_ratios=(1, ))
    with pytest.raises(ValueError, match=r"256\^3"):
        ProtocolConfiguration(analysis_grid=64)
    with pytest.raises(ValueError, match="exact thresholds"):
        ProtocolConfiguration(thresholds=GateThresholds(0.998, 0.01, 0.05))


def test_timed_checkpoint_path_uses_one_trajectory_and_one_initialization():
    jax = pytest.importorskip("jax")

    class Conf:
        float_dtype = jax.numpy.float32

    calls = {"trajectory": 0, "initializations": 0}

    def fake_collect(particles, cosmology, conf, collector, collector_state, correction=None, ):
        del cosmology, conf, correction
        calls["trajectory"] += 1
        calls["initializations"] += 1
        state = collector_state
        current = particles
        one_third = float(np.nextafter(np.float32(1.0 / 3.0), np.float32(1.0)))
        two_thirds = float(np.nextafter(np.float32(2.0 / 3.0), np.float32(0.0)))
        for a_prev, a_next in ((0.0, one_third), (one_third, two_thirds), (two_thirds, 1.0), ):
            current = current + 1
            state = collector(state, a_prev, a_next, current, None, None)
        return state

    states, seen = _run_checkpoint_trajectory(
        fake_collect, jax, jax.numpy.asarray(0), None, Conf(), (1.0 / 3.0, 2.0 / 3.0, 1.0), None,
    )
    assert calls == {"trajectory": 1, "initializations": 1}
    assert np.array_equal(np.asarray(seen), np.ones(3, dtype=bool))
    assert tuple(int(value) for value in states) == (1, 2, 3)


def test_jitted_checkpoint_collector_handles_particles_pytree():
    jax = pytest.importorskip("jax")
    from pmpp.core import Configuration
    from pmpp.nbody import Particles

    conf = Configuration(1.0, (2, 2, 2), a_start=0.25, a_stop=1.0)
    particles = Particles(conf, pmid=np.zeros((8, 3), dtype=np.int16), disp=np.zeros((8, 3), dtype=np.float32), )

    def fake_collect(initial_particles, cosmology, active_conf, collector, collector_state, correction=None, ):
        del cosmology, active_conf, correction

        def body(carry, pair):
            current, state = carry
            a_prev, a_next = pair
            current = current.replace(disp=current.disp + 1.0)
            state = collector(state, a_prev, a_next, current, None, conf)
            return (current, state), None

        pairs = (
            jax.numpy.asarray((0.0, 0.25, 0.5),
                              dtype=jax.numpy.float32), jax.numpy.asarray((0.25, 0.5, 1.0), dtype=jax.numpy.float32),
        )
        initialized = initial_particles.replace(acc=jax.numpy.zeros_like(initial_particles.disp))
        (_, state), _ = jax.lax.scan(body, (initialized, collector_state), pairs)
        return state

    compiled = jax.jit(
        lambda initial: _run_checkpoint_trajectory(fake_collect, jax, initial, None, conf, (0.25, 0.5, 1.0), None,
                                                   )
    )
    states, seen = compiled(particles)
    assert np.array_equal(np.asarray(seen), np.ones(3, dtype=bool))
    assert tuple(float(np.asarray(state.disp[0, 0])) for state in states) == (1.0, 2.0, 3.0, )


def test_memory_report_does_not_claim_cumulative_counters_as_case_peak():
    jax = pytest.importorskip("jax")
    value, timing = _timed_full_jit(
        jax, lambda x, y: x + y, jax.numpy.asarray(1.0), jax.numpy.asarray(2.0), repeats=1, devices=(),
    )
    assert float(value) == pytest.approx(3.0)
    memory = timing["memory_measurement"]
    assert memory["isolated_case_peak_available"] is False
    assert "not an isolated per-case peak" in memory["scope"]
    assert "process lifetime" in memory["attribution_caveat"]
    assert "process_lifetime_peak_rss_bytes" in memory["process_before"]


def test_capacity_callback_failure_becomes_persistable_invalid_case():
    failure = _pmpp_failure_record(
        RuntimeError("Exceeded max_ptcl_per_slice: capacity overflow"), mesh_ratio=2, acceptance_eligible=True,
    )
    assert failure["status"] == "capacity_or_particle_invariant_failure"
    assert failure["health"]["capacity_overflow_detected"] is True
    assert failure["health"]["capacity_overflow_pass"] is False
    assert failure["health"]["run_valid"] is False
    assert failure["summary"]["hard_gate_pass"] is False


def test_acceptance_contract_requires_native_data_and_smoke_can_never_pass():
    native = {"res": 256, "source_res": 256, "downsample_factor": 1}
    contract = build_evaluation_contract("acceptance", native)
    assert contract["acceptance_eligible"] is True
    assert contract["overall_pass_allowed"] is True

    single_gpu = build_evaluation_contract("acceptance", native, num_devices=1, multigpu_mode=None)
    assert single_gpu["acceptance_eligible"] is False
    assert "exactly two GPUs" in single_gpu["reason"]

    small = {"res": 64, "source_res": 256, "downsample_factor": 4}
    smoke = build_evaluation_contract("smoke", small, requested_downsample_res=64)
    assert smoke["status"] == "non_acceptance"
    assert smoke["acceptance_eligible"] is False
    assert smoke["overall_pass_allowed"] is False
    summary = summarize_family({"mesh_1": _healthy_case(), "mesh_2": _healthy_case(), }, acceptance_eligible=False, )
    assert summary["measurement_status"] == "complete"
    assert summary["status"] == "non_acceptance"
    assert summary["hard_gate_pass_all_meshes_and_snapshots"] is False


def test_learned_checkpoint_is_blind_evaluation_only_and_records_both_manifests(tmp_path, ):
    checkpoint, spacing, training_path, blind_path = _learned_checkpoint(tmp_path)
    provenance = validate_learned_checkpoint_for_blind_evaluation(
        checkpoint, current_critical_source_sha256="source-hash", **_blind_validation_kwargs(spacing, blind_path),
    )
    assert provenance["purpose"] == "blind_evaluation_only"
    assert provenance["warm_start"] is False
    assert provenance["optimizer_state_loaded"] is False
    assert provenance["evaluation_manifest"]["matches_reserved_blind"] is True
    assert provenance["training_manifest"]["training"][0]["realization_id"] == "0"

    with pytest.raises(ValueError, match="different from training"):
        training_kwargs = _blind_validation_kwargs(spacing, blind_path)
        training_kwargs.update(evaluation_realization_id="0", evaluation_path=training_path)
        validate_learned_checkpoint_for_blind_evaluation(checkpoint, **training_kwargs, )


def test_non_native_learned_checkpoint_is_allowed_only_for_held_out_smoke(tmp_path):
    checkpoint, _, _, blind_path = _learned_checkpoint(tmp_path)
    checkpoint = copy.deepcopy(checkpoint)
    spacing = 1000.0 / 64.0
    training = checkpoint["metadata"]["data_manifest"]["training"][0]
    training["particle_resolution"] = 64
    training["downsample_factor"] = 4
    checkpoint["metadata"]["normalization"]["particle_spacing_mpc_h"] = spacing
    checkpoint["metadata"]["normalization"]["cross_band"]["k_max_h_mpc"] = 0.5 * np.pi / spacing
    checkpoint["metadata"].pop("lineage")
    checkpoint["metadata"]["compatibility"]["data_manifest_sha256"] = (
        _stable_mapping_hash(checkpoint["metadata"]["data_manifest"])
    )

    kwargs = _blind_validation_kwargs(spacing, blind_path)
    with pytest.raises(ValueError, match="native 256-resolution"):
        validate_learned_checkpoint_for_blind_evaluation(checkpoint, **kwargs)

    provenance = validate_learned_checkpoint_for_blind_evaluation(
        checkpoint, allow_non_native_smoke_checkpoint=True, **kwargs,
    )
    assert provenance["purpose"] == "held_out_smoke_evaluation_only"
    assert provenance["acceptance_eligible"] is False


def test_smoke_report_never_emits_overall_or_family_gate_pass():
    cases = {"mesh_1": _healthy_case(), "mesh_2": _healthy_case(), }
    report = {
        "evaluation_contract": {
            "mode": "smoke",
            "acceptance_eligible": False,
        },
        "baselines": {
            "pmpp_uncorrected": {
                "cases": cases
            },
            "pmpp_fastpm_3_4": {
                "cases": cases
            },
            "pmpp_learned_correction": {
                "cases": cases
            },
            "official_fastpm": {
                "availability": "unavailable",
                "cases": {}
            },
        },
    }
    finalize_report_summary(report)
    assert report["summary"]["measurement_matrix_complete"] is True
    assert report["summary"]["pmpp_baseline_matrix_complete"] is False
    assert report["summary"]["learned_acceptance_protocol_complete"] is False
    assert report["summary"]["overall_acceptance_pass"] is False
    assert report["summary"]["overall_acceptance_status"] == "not_eligible"
    assert (
        report["summary"]["families"]["pmpp_learned_correction"]["hard_gate_pass_all_meshes_and_snapshots"] is False
    )

    missing_contract = {"baselines": report["baselines"]}
    finalize_report_summary(missing_contract)
    assert missing_contract["summary"]["evaluation_mode"] == "unspecified"
    assert missing_contract["summary"]["acceptance_eligible"] is False
    assert missing_contract["summary"]["overall_acceptance_pass"] is False


def test_capacity_configuration_records_effective_static_capacities():
    automatic = CapacityConfiguration(max_ptcl_factor=1.05).resolve(101)
    assert automatic["max_ptcl_per_slice"] == 107
    assert automatic["max_share_ptcl"] == 50_000
    assert automatic["max_halo_share_ptcl"] == 400_000
    assert automatic["max_share_gather_ptcl"] == 1_200_000
    assert automatic["overflow_policy"].startswith("any capacity overflow")

    explicit = CapacityConfiguration(max_ptcl_per_slice=123).resolve(101)
    assert explicit["max_ptcl_per_slice"] == 123
    with pytest.raises(ValueError, match="below observed"):
        CapacityConfiguration(max_ptcl_per_slice=100).resolve(101)


def test_case_gate_is_cross_and_positions_only():
    # Deliberately fail both auxiliary targets: they must not change the hard gate.
    snapshots = _all_snapshots(power_pass=False, projection_pass=False)
    summary = summarize_case(snapshots)
    assert summary["status"] == "complete"
    assert summary["hard_gate_pass"] is True
    assert summary["power_and_projections_are_non_gating"] is True

    snapshots["1"] = _snapshot_record(cross_pass=False)
    summary = summarize_case(snapshots)
    assert summary["hard_gate_pass"] is False
    assert summary["worst_min_cross_correlation"] == pytest.approx(0.998)

    snapshots = _all_snapshots()
    snapshots["0.5"] = _snapshot_record(positions_pass=False)
    assert summarize_case(snapshots)["hard_gate_pass"] is False


def test_family_requires_both_meshes_and_every_snapshot():
    cases = {"mesh_1": _healthy_case(), "mesh_2": _healthy_case(), }
    complete = summarize_family(cases)
    assert complete["status"] == "complete"
    assert complete["hard_gate_pass_all_meshes_and_snapshots"] is True

    no_health = {"mesh_1": {"snapshots": _all_snapshots()}, "mesh_2": {"snapshots": _all_snapshots()}, }
    assert summarize_family(no_health)["run_health_pass_all_meshes"] is False
    assert (summarize_family(no_health)["hard_gate_pass_all_meshes_and_snapshots"] is False)

    del cases["mesh_2"]
    incomplete = summarize_family(cases)
    assert incomplete["status"] == "incomplete"
    assert incomplete["missing_mesh_cases"] == ["mesh_2"]
    assert incomplete["hard_gate_pass_all_meshes_and_snapshots"] is False


def test_report_summary_and_runtime_overhead_are_explicit():
    cases = {
        "mesh_1": {
            "snapshots": _all_snapshots(),
            "runtime": {
                "timed_median_seconds": 2.0
            },
            "health": {
                "run_valid": True
            },
        },
        "mesh_2": {
            "snapshots": _all_snapshots(),
            "runtime": {
                "timed_median_seconds": 4.0
            },
            "health": {
                "run_valid": True
            },
        },
    }
    corrected = {
        "mesh_1": {
            "snapshots": _all_snapshots(),
            "runtime": {
                "timed_median_seconds": 2.5
            },
            "health": {
                "run_valid": True
            },
        },
        "mesh_2": {
            "snapshots": _all_snapshots(),
            "runtime": {
                "timed_median_seconds": 5.0
            },
            "health": {
                "run_valid": True
            },
        },
    }
    report = {
        "evaluation_contract": {
            "mode": "acceptance",
            "acceptance_eligible": True,
        },
        "baselines": {
            "pmpp_uncorrected": {
                "cases": cases
            },
            "pmpp_fastpm_3_4": {
                "cases": corrected
            },
            "official_fastpm": {
                "availability": "unavailable",
                "cases": {}
            },
        }
    }
    finalize_report_summary(report)

    assert report["summary"]["pmpp_baseline_matrix_complete"] is True
    assert report["summary"]["official_fastpm_status"] == "unavailable"
    assert report["pmpp_fastpm_3_4_runtime_overhead"]["mesh_1"] == {
        "seconds": pytest.approx(0.5),
        "fraction": pytest.approx(0.25),
    }

    # Calling the helper directly is intentionally idempotent.
    add_correction_runtime_overhead(report)
    assert report["pmpp_fastpm_3_4_runtime_overhead"]["mesh_2"]["fraction"] == pytest.approx(0.25)


def test_overall_acceptance_requires_every_healthy_baseline_family():
    healthy = {"mesh_1": _healthy_case(), "mesh_2": _healthy_case()}
    report = {
        "status": "complete",
        "evaluation_contract": {
            "mode": "acceptance",
            "acceptance_eligible": True
        },
        "baselines": {
            "pmpp_uncorrected": {
                "cases": copy.deepcopy(healthy)
            },
            "pmpp_fastpm_3_4": {
                "cases": copy.deepcopy(healthy)
            },
            "pmpp_learned_correction": {
                "cases": copy.deepcopy(healthy)
            },
            "official_fastpm": {
                "availability": "available",
                "cases": copy.deepcopy(healthy),
            },
        },
    }
    finalize_report_summary(report)
    assert report["summary"]["overall_acceptance_pass"] is True

    report["baselines"]["pmpp_uncorrected"]["cases"]["mesh_1"]["health"] = {"run_valid": False}
    finalize_report_summary(report)
    assert report["summary"]["pmpp_baseline_matrix_complete"] is False
    assert report["summary"]["overall_acceptance_pass"] is False


def test_report_json_is_strict_and_sanitizes_non_finite_values(tmp_path):
    output = tmp_path / "report.json"
    write_json(output, {"finite": np.float32(1.5), "array": np.asarray([0.0, np.nan, np.inf]), }, )
    text = output.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"array": [0.0, None, None], "finite": 1.5}


def test_ratio_path_parser_rejects_ambiguous_configuration(tmp_path):
    one = tmp_path / "one.lua"
    two = tmp_path / "two.lua"
    parsed = parse_ratio_path([f"1={one}", f"2={two}"], "--parameter")
    assert parsed == {1: one, 2: two}

    with pytest.raises(ValueError, match="RATIO=PATH"):
        parse_ratio_path([str(one)], "--parameter")
    with pytest.raises(ValueError, match="repeats"):
        parse_ratio_path([f"1={one}", f"1={two}"], "--parameter")
    with pytest.raises(ValueError, match="1 or 2"):
        parse_ratio_path([f"3={one}"], "--parameter")


def test_official_fastpm_unavailable_and_supplied_invocation(tmp_path):
    protocol_manifest = tmp_path / "protocol.json"
    protocol_manifest.write_text("{}", encoding="utf-8")
    unavailable = invoke_official_fastpm(
        executable=None, mesh_ratio=1, parameter_file=None, protocol_manifest=protocol_manifest,
        output_dir=tmp_path / "unavailable",
    )
    assert unavailable == {
        "status": "unavailable",
        "reason": "--fastpm-executable was not supplied",
        "invoked": False,
    }

    parameter_file = tmp_path / "fake_parameter.py"
    parameter_file.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path('seen.txt').write_text(os.environ['PMPP_QUIJOTE_MESH_RATIO'])\n", encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    completed = invoke_official_fastpm(
        executable=sys.executable, mesh_ratio=2, parameter_file=parameter_file, protocol_manifest=protocol_manifest,
        output_dir=run_dir,
    )
    assert completed["status"] == "completed"
    assert completed["invoked"] is True
    assert completed["returncode"] == 0
    assert (run_dir / "seen.txt").read_text(encoding="utf-8") == "2"
    assert (run_dir / "stdout.log").is_file()
    assert (run_dir / "stderr.log").is_file()


def test_official_fastpm_manifest_requires_protocol_and_canonical_order(tmp_path):
    protocol = ProtocolConfiguration()
    schedule = build_checkpoint_schedule(1.0 / 128.0, protocol)
    snapshots = {}
    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    for key in ("3", "2", "1", "0.5", "0"):
        path = tmp_path / f"z_{key.replace('.', 'p')}.npy"
        np.save(path, positions)
        snapshots[key] = path.name
    manifest = {
        "schema_version": 1,
        "implementation": "official_fastpm",
        "mesh_ratio": 1,
        "total_steps": 63,
        "redshifts": [3.0, 2.0, 1.0, 0.5, 0.0],
        "global_scale_factors": schedule.global_scale_factors.tolist(),
        "particle_order": "canonical_lagrangian_grid",
        "snapshots": snapshots,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_official_fastpm_outputs(
        manifest_path, mesh_ratio=1, expected_particles=2, protocol=protocol, schedule=schedule,
    )
    assert set(loaded) == {3.0, 2.0, 1.0, 0.5, 0.0}
    assert np.array_equal(loaded[0.5], positions)

    manifest["particle_order"] = "gadget_file_order"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="particle_order"):
        load_official_fastpm_outputs(
            manifest_path, mesh_ratio=1, expected_particles=2, protocol=protocol, schedule=schedule,
        )


def test_official_fastpm_label_requires_linked_observed_provenance(tmp_path):
    protocol = ProtocolConfiguration()
    schedule = build_checkpoint_schedule(1.0 / 128.0, protocol)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    parameter_file = tmp_path / "parameter.py"
    parameter_file.write_text("pass\n", encoding="utf-8")
    invocation = invoke_official_fastpm(
        executable=sys.executable, mesh_ratio=1, parameter_file=parameter_file, protocol_manifest=protocol_path,
        output_dir=tmp_path / "run",
    )
    assert invocation["status"] == "completed"

    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    snapshots = {}
    output_hashes = {}
    for key in ("3", "2", "1", "0.5", "0"):
        snapshot_path = tmp_path / f"z_{key.replace('.', 'p')}.npy"
        np.save(snapshot_path, positions)
        snapshots[key] = snapshot_path.name
        output_hashes[key] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    output_manifest = tmp_path / "outputs.json"
    output_manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "implementation": "official_fastpm",
            "mesh_ratio": 1,
            "total_steps": 63,
            "redshifts": [3.0, 2.0, 1.0, 0.5, 0.0],
            "global_scale_factors": schedule.global_scale_factors.tolist(),
            "particle_order": "canonical_lagrangian_grid",
            "snapshots": snapshots,
        }), encoding="utf-8",
    )
    trajectory = {
        "ic_pos": positions,
        "ic_vel": positions * np.float32(0.1),
        "particle_order": "canonical_lagrangian_grid",
        "canonical_particle_ids_sha256": "c" * 64,
        "res": 256,
        "box_size": 1000.0,
    }
    hashes = fastpm_input_content_hashes(trajectory)
    unlinked = verify_official_fastpm_provenance(
        None, invocation=invocation, output_manifest=output_manifest, trajectory=trajectory, protocol=protocol,
        schedule=schedule, mesh_ratio=1,
    )
    assert unlinked["status"] == "external_unverified"
    assert unlinked["official_verified"] is False

    provenance = {
        "schema_version": 1,
        "source_repo_url": "https://github.com/fastpm/fastpm",
        "source_commit": "a" * 40,
        "build": {
            "compiler": "gcc 12",
            "command": "make"
        },
        "binary_sha256": invocation["executable_sha256"],
        "parameter_file_sha256": invocation["parameter_file_sha256"],
        "protocol_manifest_sha256": invocation["protocol_manifest_sha256"],
        "invocation_sha256": invocation["invocation_sha256"],
        "output_manifest_sha256": hashlib.sha256(output_manifest.read_bytes()).hexdigest(),
        "output_files_sha256": output_hashes,
        **hashes, "position_units": "Mpc/h",
        "box_size_mpc_h": 1000.0,
        "mesh_ratio": 1,
        "total_steps": 63,
        "redshifts": [3.0, 2.0, 1.0, 0.5, 0.0],
        "global_scale_factors": schedule.global_scale_factors.tolist(),
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    verified = verify_official_fastpm_provenance(
        provenance_path, invocation=invocation, output_manifest=output_manifest, trajectory=trajectory,
        protocol=protocol, schedule=schedule, mesh_ratio=1,
    )
    assert verified["status"] == "external_unverified"
    assert verified["official_verified"] is False
    assert any(
        "locked official FastPM protocol commit" in reason or "source_checkout" in reason
        for reason in verified["reasons"]
    )

    provenance["binary_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    unverified = verify_official_fastpm_provenance(
        provenance_path, invocation=invocation, output_manifest=output_manifest, trajectory=trajectory,
        protocol=protocol, schedule=schedule, mesh_ratio=1,
    )
    assert unverified["status"] == "external_unverified"
    assert unverified["official_verified"] is False
    assert any("binary_sha256" in reason for reason in unverified["reasons"])


def test_plot_bundle_is_created_from_small_cpu_arrays(tmp_path):
    pytest.importorskip("matplotlib")
    thresholds = GateThresholds()
    k = np.asarray([0.1, 0.2, 0.3])
    r = np.asarray([0.9999, 0.9997, 0.9995])
    ratio = np.asarray([1.0, 1.005, 0.995])
    plot_spectra(tmp_path / "spectra.png", k, r, ratio, 0.25, thresholds, "small report", )
    x, cdf = plot_position_cdf(
        tmp_path / "cdf.png", np.asarray([0.0, 0.01, 0.02, 0.03]), thresholds, 0.009, 0.04, "small report",
    )
    reference = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    candidate = reference + np.float32(0.1)
    pearson = plot_xyz_projections(tmp_path / "projections.png", reference, candidate, "small report")

    assert (tmp_path / "spectra.png").stat().st_size > 0
    assert (tmp_path / "cdf.png").stat().st_size > 0
    assert (tmp_path / "projections.png").stat().st_size > 0
    assert x.shape == cdf.shape
    assert set(pearson) == {"x", "y", "z"}
    assert min(value for value in pearson.values() if value is not None) > 0.999
