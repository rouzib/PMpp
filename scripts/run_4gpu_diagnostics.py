#!/usr/bin/env python
"""Run forward-only 4-GPU diagnostics and collect structured logs.

The script is intended for cluster runs where we want reproducible, pasteable
artifacts rather than ad hoc notebook output. It exercises:

1. PMPP vs PMWD forward comparisons on small cases,
2. step-by-step PMPP vs PMWD traces to localize the first bad N-body step, and
3. larger PMPP-only forward stress cases.

Each case is executed in a separate subprocess so JAX/XLA state is isolated and
GPU memory is released between runs. The parent process writes:

- ``env.json``: environment and device inventory
- ``summary.json``: aggregate status across all cases
- ``<case>/console.log``: raw stdout/stderr for one case
- ``<case>/result.json``: structured metrics for one case
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import socket
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PM
from pmwd.lpt import lpt as lpt_pmwd
from pmwd.modes import linear_modes as linear_modes_pmwd
from pmwd.modes import white_noise as white_noise_pmwd
from pmwd.nbody import nbody_init as nbody_init_pmwd
from pmwd.nbody import nbody_step as nbody_step_pmwd
from pmwd.scatter import scatter as scatter_pmwd

from pmpp.boltzmann import boltzmann as boltzmann_pmpp
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM as SimpleLCDM_PP
from pmpp.gravity import duplicate_slot_counts
from pmpp.lpt import lpt as lpt_pmpp
from pmpp.modes import linear_modes as linear_modes_pmpp
from pmpp.modes import white_noise as white_noise_pmpp
from pmpp.nbody import nbody_init as nbody_init_pmpp
from pmpp.nbody import nbody_step as nbody_step_pmpp
from pmpp.scatter import scatter as scatter_pmpp
from pmpp.utils import create_compute_mesh


DEFAULT_A_START = 1 / 64
DEFAULT_A_NBODY_MAXSTEP = 1 / 64
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Case:
    name: str
    num_ptcl: int
    mesh_shape: int
    a_stop: float
    compare_pmwd: bool
    trace_steps: bool
    use_float64: bool = True
    max_ptcl_factor: float = 3.0
    max_share_ptcl: int = 50_000
    max_share_gather_ptcl: int = 150_000


def _case_name(prefix: str, num_ptcl: int, mesh_shape: int, a_stop: float) -> str:
    token = str(a_stop).replace(".", "p")
    return f"{prefix}_np{num_ptcl}_mesh{mesh_shape}_a{token}"


def build_case_catalog() -> dict[str, Case]:
    cases: list[Case] = []

    for num_ptcl in (8, 16):
        for mesh_shape in (1, 2, 4):
            a_stop = 1 / 8
            cases.append(
                Case(
                    name=_case_name("compare", num_ptcl, mesh_shape, a_stop),
                    num_ptcl=num_ptcl,
                    mesh_shape=mesh_shape,
                    a_stop=a_stop,
                    compare_pmwd=True,
                    trace_steps=False,
                    use_float64=True,
                )
            )

    for mesh_shape in (1, 2, 4):
        a_stop = 1 / 4
        cases.append(
            Case(
                name=_case_name("trace", 16, mesh_shape, a_stop),
                num_ptcl=16,
                mesh_shape=mesh_shape,
                a_stop=a_stop,
                compare_pmwd=True,
                trace_steps=True,
                use_float64=True,
            )
        )

    for num_ptcl, mesh_shape, a_stop in (
        (32, 1, 1 / 4),
        (32, 2, 1 / 4),
        (32, 4, 1 / 4),
        (64, 1, 1 / 8),
    ):
        cases.append(
            Case(
                name=_case_name("stress", num_ptcl, mesh_shape, a_stop),
                num_ptcl=num_ptcl,
                mesh_shape=mesh_shape,
                a_stop=a_stop,
                compare_pmwd=False,
                trace_steps=False,
                use_float64=False,
                max_ptcl_factor=2.5,
            )
        )

    for num_ptcl, mesh_shape, a_stop in (
        (64, 2, 1 / 8),
        (64, 4, 1 / 8),
    ):
        cases.append(
            Case(
                name=_case_name("stress", num_ptcl, mesh_shape, a_stop),
                num_ptcl=num_ptcl,
                mesh_shape=mesh_shape,
                a_stop=a_stop,
                compare_pmwd=False,
                trace_steps=False,
                use_float64=False,
                max_ptcl_factor=2.5,
            )
        )

    return {case.name: case for case in cases}


CASE_CATALOG = build_case_catalog()
SUITES = {
    "core": [
        case.name
        for case in CASE_CATALOG.values()
        if case.name.startswith("compare_") or case.name.startswith("trace_")
    ],
    "extended": [
        case.name
        for case in CASE_CATALOG.values()
        if not (case.num_ptcl == 64 and case.mesh_shape in (2, 4))
    ],
    "full": list(CASE_CATALOG.keys()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="extended")
    parser.add_argument("--case", choices=tuple(CASE_CATALOG), default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_numpy(array):
    if array is None:
        return None
    return np.asarray(jax.device_get(array))


def maybe_float(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def resolve_gpu_devices(num_devices: int) -> list[jax.Device]:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpu_devices) < num_devices:
        raise RuntimeError(
            f"Requested {num_devices} GPUs but only found {len(gpu_devices)}: {gpu_devices}"
        )
    return gpu_devices[:num_devices]


def run_cmd(*args: str) -> dict[str, object]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def collect_env_report(num_devices: int) -> dict[str, object]:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    selected = gpu_devices[: min(num_devices, len(gpu_devices))]
    report = {
        "timestamp_unix": int(time()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "repo_root": str(REPO_ROOT),
        "git_head": run_cmd("git", "rev-parse", "HEAD"),
        "git_status_short": run_cmd("git", "status", "--short"),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "jaxlib_version": getattr(getattr(jax, "lib", None), "__version__", "unknown"),
        "device_count_total": len(jax.devices()),
        "gpu_count": len(gpu_devices),
        "selected_gpu_count": len(selected),
        "devices": [str(device) for device in jax.devices()],
        "selected_devices": [str(device) for device in selected],
        "env_subset": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_GPUS",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_PLATFORMS",
            )
            if key in os.environ
        },
        "nvidia_smi_l": run_cmd("nvidia-smi", "-L"),
    }
    return report


def build_pmpp_conf(case: Case, gpu_devices: list[jax.Device]) -> Configuration:
    ptcl_grid_shape = (case.num_ptcl,) * 3
    ptcl_spacing = 100.0 / case.num_ptcl
    compute_mesh = create_compute_mesh(gpu_devices)
    dtype_kwargs = {}
    if case.use_float64:
        dtype_kwargs = {"cosmo_dtype": jnp.float64, "float_dtype": jnp.float64}

    return Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=case.mesh_shape,
        compute_mesh=compute_mesh,
        a_start=DEFAULT_A_START,
        a_stop=case.a_stop,
        a_nbody_maxstep=DEFAULT_A_NBODY_MAXSTEP,
        max_ptcl_per_slice=int(math.ceil(case.num_ptcl**3 / len(gpu_devices) * case.max_ptcl_factor)),
        max_share_ptcl=case.max_share_ptcl,
        max_share_gather_ptcl=case.max_share_gather_ptcl,
        **dtype_kwargs,
    )


def build_pmwd_conf(conf_pmpp: Configuration, use_float64: bool) -> ConfigurationPMWD:
    dtype_kwargs = {}
    if use_float64:
        dtype_kwargs = {"cosmo_dtype": jnp.float64, "float_dtype": jnp.float64}
    return ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_stop=conf_pmpp.a_stop,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        **dtype_kwargs,
    )


def density_stats(density: np.ndarray, expected_sum: float) -> dict[str, object]:
    finite_mask = np.isfinite(density)
    density_sum = float(density.sum())
    return {
        "shape": list(density.shape),
        "sum": density_sum,
        "mean": float(density.mean()),
        "min": float(density.min()),
        "max": float(density.max()),
        "expected_sum": float(expected_sum),
        "abs_sum_error": float(abs(density_sum - expected_sum)),
        "rel_sum_error": float(abs(density_sum - expected_sum) / expected_sum) if expected_sum else 0.0,
        "non_finite_count": int(density.size - int(finite_mask.sum())),
        "all_finite": bool(finite_mask.all()),
    }


def density_diff_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    diff = candidate - reference
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "sum_diff": float(diff.sum()),
    }


def particle_accounting(ptcl, conf) -> dict[str, object]:
    report = {
        "acc_is_none": getattr(ptcl, "acc", None) is None,
        "vel_is_none": getattr(ptcl, "vel", None) is None,
    }

    unused = to_numpy(getattr(ptcl, "unused_index", None))
    halo = to_numpy(getattr(ptcl, "halo_mask", None))
    if unused is not None:
        num_devices = int(getattr(conf, "num_devices", 1))
        max_ptcl_per_slice = int(getattr(conf, "max_ptcl_per_slice", unused.size))
        unused = unused.reshape(num_devices, max_ptcl_per_slice)
        report["unused_slots"] = int(unused.sum())
        report["valid_slots"] = int(unused.size - unused.sum())
        report["per_device_valid_slots"] = [int((~device_unused).sum()) for device_unused in unused]
        report["per_device_unused_slots"] = [int(device_unused.sum()) for device_unused in unused]
    if halo is not None:
        num_devices = int(getattr(conf, "num_devices", 1))
        max_ptcl_per_slice = int(getattr(conf, "max_ptcl_per_slice", halo.size))
        halo = halo.reshape(num_devices, max_ptcl_per_slice)
        report["halo_slots"] = int(halo.sum())
        report["per_device_halo_slots"] = [int(device_halo.sum()) for device_halo in halo]

    if bool(getattr(conf, "use_mGPU", False)):
        counts = to_numpy(duplicate_slot_counts(ptcl, conf))
        if counts is not None:
            counts = counts.reshape(-1, counts.shape[-1])[:, 0]
            valid_mask = np.ones_like(counts, dtype=bool)
            if unused is not None:
                valid_mask = ~unused.reshape(-1)
            valid_counts = counts[valid_mask]
            report["duplicate_slot_count"] = {
                "max": float(valid_counts.max()) if valid_counts.size else 0.0,
                "mean": float(valid_counts.mean()) if valid_counts.size else 0.0,
                "slots_gt_1": int(np.count_nonzero(valid_counts > 1.0 + 1e-6)),
                "slots_gt_2": int(np.count_nonzero(valid_counts > 2.0 + 1e-6)),
            }

    return report


def stage_report(stage: str, ptcl, density: np.ndarray, conf) -> dict[str, object]:
    return {
        "stage": stage,
        "density": density_stats(density, expected_sum=float(np.prod(conf.mesh_shape))),
        "particles": particle_accounting(ptcl, conf),
    }


def build_findings(case: Case, result: dict[str, object]) -> list[str]:
    findings: list[str] = []
    if result.get("status") != "ok":
        findings.append("case_failed")
        return findings

    pmpp = result.get("pmpp", {})
    final_density = pmpp.get("final", {}).get("density", {})
    if final_density.get("non_finite_count", 0):
        findings.append("final_density_non_finite")
    if final_density.get("abs_sum_error", 0.0) > 1e-2:
        findings.append("final_mass_not_conserved")

    if case.compare_pmwd:
        comparisons = result.get("comparisons", {})
        lpt_cmp = comparisons.get("lpt_density", {})
        final_cmp = comparisons.get("final_density", {})
        if lpt_cmp.get("max_abs_diff", 0.0) > 1e-2:
            findings.append("lpt_diverges_from_pmwd")
        if final_cmp.get("max_abs_diff", 0.0) > 1e-1:
            findings.append("final_diverges_from_pmwd")

        per_step = comparisons.get("per_step_density", [])
        for entry in per_step:
            if entry.get("diff", {}).get("max_abs_diff", 0.0) > 1e-1:
                findings.append(f"step_{entry['step_index']}_diverges_from_pmwd")
                break

    if case.trace_steps:
        for entry in pmpp.get("step_trace", []):
            if entry.get("density", {}).get("abs_sum_error", 0.0) > 1e-2:
                findings.append(f"step_{entry['step_index']}_mass_not_conserved")
                break

    return findings


def _pmpp_step_functions():
    init_fn = jax.jit(nbody_init_pmpp, static_argnames=("conf",))
    step_fn = jax.jit(nbody_step_pmpp, static_argnames=("conf",))
    lpt_fn = jax.jit(lpt_pmpp, static_argnames=("conf",))
    scatter_fn = jax.jit(scatter_pmpp, static_argnames=("conf",))
    return lpt_fn, init_fn, step_fn, scatter_fn


def run_case(case: Case, num_devices: int, seed: int) -> dict[str, object]:
    started = perf_counter()
    gpu_devices = resolve_gpu_devices(num_devices)
    lpt_fn_pmpp, init_fn_pmpp, step_fn_pmpp, scatter_fn_pmpp = _pmpp_step_functions()

    conf_pmpp = build_pmpp_conf(case, gpu_devices)
    base_cosmo_pmpp = SimpleLCDM_PP(conf_pmpp)
    cosmo_pmpp = boltzmann_pmpp(base_cosmo_pmpp, conf_pmpp)
    modes_pmpp = white_noise_pmpp(seed, conf_pmpp)
    modes_pmpp = linear_modes_pmpp(modes_pmpp, cosmo_pmpp, conf_pmpp)

    lpt_conf = conf_pmpp.replace(max_share_ptcl=conf_pmpp.max_share_ptcl * 2)
    ptcl_lpt_pmpp = lpt_fn_pmpp(modes_pmpp, cosmo_pmpp, lpt_conf)
    dens_lpt_pmpp = to_numpy(scatter_fn_pmpp(ptcl_lpt_pmpp, conf_pmpp))
    pmpp_report: dict[str, object] = {
        "config": {
            "num_devices": int(conf_pmpp.num_devices),
            "ptcl_grid_shape": list(map(int, conf_pmpp.ptcl_grid_shape)),
            "mesh_shape": list(map(int, conf_pmpp.mesh_shape)),
            "local_mesh_shape": list(map(int, conf_pmpp.local_mesh_shape)),
            "max_ptcl_per_slice": int(conf_pmpp.max_ptcl_per_slice),
            "max_share_ptcl": int(conf_pmpp.max_share_ptcl),
            "max_share_gather_ptcl": int(conf_pmpp.max_share_gather_ptcl),
            "a_start": float(conf_pmpp.a_start),
            "a_stop": float(conf_pmpp.a_stop),
            "a_nbody_maxstep": float(conf_pmpp.a_nbody_maxstep),
            "num_steps": int(len(conf_pmpp.a_nbody) - 1),
        },
        "lpt": stage_report("lpt", ptcl_lpt_pmpp, dens_lpt_pmpp, conf_pmpp),
    }

    ptcl_init_pmpp = init_fn_pmpp(conf_pmpp.a_nbody[0], ptcl_lpt_pmpp, cosmo_pmpp, conf_pmpp)
    dens_init_pmpp = to_numpy(scatter_fn_pmpp(ptcl_init_pmpp, conf_pmpp))
    pmpp_report["init_force"] = stage_report("init_force", ptcl_init_pmpp, dens_init_pmpp, conf_pmpp)

    step_trace_pmpp = []
    ptcl_pmpp = ptcl_init_pmpp

    pmwd_report = None
    comparisons: dict[str, object] = {}
    per_step_comparisons = []

    if case.compare_pmwd:
        conf_pmwd = build_pmwd_conf(conf_pmpp, case.use_float64)
        base_cosmo_pmwd = SimpleLCDM_PM(conf_pmwd)
        cosmo_pmwd = boltzmann_pmwd(base_cosmo_pmwd, conf_pmwd)
        modes_pmwd = white_noise_pmwd(seed, conf_pmwd)
        modes_pmwd = linear_modes_pmwd(modes_pmwd, cosmo_pmwd, conf_pmwd)

        ptcl_lpt_pmwd, _ = lpt_pmwd(modes_pmwd, cosmo_pmwd, conf_pmwd)
        dens_lpt_pmwd = to_numpy(scatter_pmwd(ptcl_lpt_pmwd, conf_pmwd))
        pmwd_report = {"lpt": stage_report("lpt", ptcl_lpt_pmwd, dens_lpt_pmwd, conf_pmwd)}
        comparisons["lpt_density"] = density_diff_metrics(dens_lpt_pmwd, dens_lpt_pmpp)

        ptcl_init_pmwd, _ = nbody_init_pmwd(conf_pmwd.a_nbody[0], ptcl_lpt_pmwd, None, cosmo_pmwd, conf_pmwd)
        dens_init_pmwd = to_numpy(scatter_pmwd(ptcl_init_pmwd, conf_pmwd))
        pmwd_report["init_force"] = stage_report("init_force", ptcl_init_pmwd, dens_init_pmwd, conf_pmwd)
        comparisons["init_force_density"] = density_diff_metrics(dens_init_pmwd, dens_init_pmpp)
    else:
        conf_pmwd = None
        ptcl_init_pmwd = None

    ptcl_pmwd = ptcl_init_pmwd

    for step_index, (a_prev, a_next) in enumerate(zip(conf_pmpp.a_nbody[:-1], conf_pmpp.a_nbody[1:])):
        ptcl_pmpp = step_fn_pmpp(a_prev, a_next, ptcl_pmpp, cosmo_pmpp, conf_pmpp)
        dens_pmpp = to_numpy(scatter_fn_pmpp(ptcl_pmpp, conf_pmpp))

        step_entry_pmpp = {
            "step_index": int(step_index),
            "a_prev": float(a_prev),
            "a_next": float(a_next),
            **stage_report(f"step_{step_index}", ptcl_pmpp, dens_pmpp, conf_pmpp),
        }
        if case.trace_steps:
            step_trace_pmpp.append(step_entry_pmpp)

        if case.compare_pmwd:
            ptcl_pmwd, _ = nbody_step_pmwd(a_prev, a_next, ptcl_pmwd, None, cosmo_pmwd, conf_pmwd)
            dens_pmwd = to_numpy(scatter_pmwd(ptcl_pmwd, conf_pmwd))

            if case.trace_steps:
                step_report_pmwd = {
                    "step_index": int(step_index),
                    "a_prev": float(a_prev),
                    "a_next": float(a_next),
                    **stage_report(f"step_{step_index}", ptcl_pmwd, dens_pmwd, conf_pmwd),
                }
                pmwd_report.setdefault("step_trace", []).append(step_report_pmwd)

            if case.trace_steps:
                per_step_comparisons.append(
                    {
                        "step_index": int(step_index),
                        "a_prev": float(a_prev),
                        "a_next": float(a_next),
                        "diff": density_diff_metrics(dens_pmwd, dens_pmpp),
                    }
                )

    final_density_pmpp = to_numpy(scatter_fn_pmpp(ptcl_pmpp, conf_pmpp))
    pmpp_report["final"] = stage_report("final", ptcl_pmpp, final_density_pmpp, conf_pmpp)
    if case.trace_steps:
        pmpp_report["step_trace"] = step_trace_pmpp

    if case.compare_pmwd:
        final_density_pmwd = to_numpy(scatter_pmwd(ptcl_pmwd, conf_pmwd))
        pmwd_report["final"] = stage_report("final", ptcl_pmwd, final_density_pmwd, conf_pmwd)
        comparisons["final_density"] = density_diff_metrics(final_density_pmwd, final_density_pmpp)
        if per_step_comparisons:
            comparisons["per_step_density"] = per_step_comparisons

    elapsed = perf_counter() - started

    result = {
        "case": asdict(case),
        "status": "ok",
        "elapsed_seconds": float(elapsed),
        "pmpp": pmpp_report,
        "pmwd": pmwd_report,
        "comparisons": comparisons,
    }
    result["findings"] = build_findings(case, result)
    return result


def run_case_entry(case_name: str, num_devices: int, seed: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"

    case = CASE_CATALOG[case_name]
    print(f"[INFO] Running case {case.name}")
    print(f"[INFO] num_devices={num_devices} seed={seed}")

    try:
        result = run_case(case, num_devices=num_devices, seed=seed)
    except Exception:
        result = {
            "case": asdict(case),
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
        result["findings"] = build_findings(case, result)
        json_dump(result_path, result)
        print("[ERROR] Case failed")
        print(result["traceback"])
        return 1

    json_dump(result_path, result)
    print(json.dumps({"case": case.name, "status": result["status"], "findings": result["findings"]}, indent=2))
    return 0


def run_suite(args: argparse.Namespace, output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    env_report = collect_env_report(args.num_devices)
    json_dump(output_root / "env.json", env_report)

    summary = {
        "suite": args.suite,
        "seed": args.seed,
        "num_devices_requested": args.num_devices,
        "python": args.python,
        "output_dir": str(output_root),
        "cases": [],
    }

    exit_code = 0
    for case_name in SUITES[args.suite]:
        case_dir = output_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        console_path = case_dir / "console.log"
        command = [
            args.python,
            str(Path(__file__).resolve()),
            "--case",
            case_name,
            "--num-devices",
            str(args.num_devices),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(case_dir),
        ]
        print(f"[RUN] {case_name}")
        print(f"[RUN] command={' '.join(command)}")
        started = perf_counter()
        with console_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        elapsed = perf_counter() - started

        result_path = case_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "case": {"name": case_name},
                "status": "failed",
                "findings": ["missing_result_json"],
            }

        summary["cases"].append(
            {
                "name": case_name,
                "returncode": int(proc.returncode),
                "elapsed_seconds": float(elapsed),
                "console_log": str(console_path),
                "result_json": str(result_path),
                "status": result.get("status", "unknown"),
                "findings": result.get("findings", []),
            }
        )

        if proc.returncode != 0 or result.get("status") != "ok":
            exit_code = 1

        print(
            json.dumps(
                {
                    "case": case_name,
                    "returncode": proc.returncode,
                    "status": result.get("status"),
                    "findings": result.get("findings", []),
                },
                indent=2,
            )
        )

    summary["failed_cases"] = [case["name"] for case in summary["cases"] if case["status"] != "ok"]
    summary["cases_with_findings"] = [
        {"name": case["name"], "findings": case["findings"]}
        for case in summary["cases"]
        if case["findings"]
    ]
    json_dump(output_root / "summary.json", summary)
    return exit_code


def default_output_dir() -> Path:
    stamp = time()
    return REPO_ROOT / "notebooks" / "tests" / "output" / f"diagnostics_4gpu_{int(stamp)}"


def main() -> int:
    args = parse_args()

    if args.list_cases:
        for suite_name, case_names in SUITES.items():
            print(f"[{suite_name}]")
            for case_name in case_names:
                print(case_name)
        return 0

    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()

    try:
        if args.case is not None:
            return run_case_entry(args.case, args.num_devices, args.seed, output_dir)
        return run_suite(args, output_dir)
    finally:
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
