#!/usr/bin/env python
"""Run a fixed-geometry 4-GPU validation suite and collect logs.

This suite is intentionally focused on ``mesh_shape = 1`` and exercises:

- distributed FFT parity,
- modes / LPT branch gradients,
- per-step drift / kick / halo-move gradients,
- scatter / gravity gradients,
- end-to-end N-body forward+gradient parity, and
- optional forward-only PMWD/PMPP comparison runs.

Each case runs in its own subprocess with its own output directory. The driver
collects:

- ``env.json``: machine / software / visible GPU summary
- ``suite_manifest.json``: exact commands and case descriptions
- ``summary.json``: status, runtime, and artifact inventory per case
- ``<case>/console.log``: raw stdout/stderr for each case
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SuiteCase:
    name: str
    script: str
    description: str
    args: tuple[str, ...]


COMMON_SEED = ("--seed", "0")
COMMON_4GPU = ("--num-devices", "4")
MESH_SHAPE_ONE = ("--mesh-shape", "1")
CASE_NAMES = (
    "fft_gradients",
    "modes_pipeline_gradients",
    "lpt_gradients",
    "drift_gradients",
    "kick_gradients",
    "halo_moving_gradients",
    "scatter_gradients",
    "gravity_gradients",
    "nbody_gradients",
    "forward_compare_smoke",
    "forward_compare_large",
)
SIZE_PROFILES = {
    "baseline": {
        "fft_mesh": 64,
        "modes_num_ptcl": 8,
        "lpt_num_ptcl": 8,
        "step_num_ptcl": 16,
        "scatter_num_ptcl": 16,
        "gravity_num_ptcl": 16,
        "nbody_num_ptcl": 8,
        "forward_smoke_num_ptcl": 32,
        "forward_large_num_ptcl": 64,
    },
    "large": {
        "fft_mesh": 128,
        "modes_num_ptcl": 16,
        "lpt_num_ptcl": 16,
        "step_num_ptcl": 32,
        "scatter_num_ptcl": 32,
        "gravity_num_ptcl": 32,
        "nbody_num_ptcl": 32,
        "forward_smoke_num_ptcl": 64,
        "forward_large_num_ptcl": 128,
    },
    "h100": {
        "fft_mesh": 512,
        "modes_num_ptcl": 64,
        "lpt_num_ptcl": 64,
        "step_num_ptcl": 128,
        "scatter_num_ptcl": 128,
        "gravity_num_ptcl": 128,
        "nbody_num_ptcl": 64,
        "forward_smoke_num_ptcl": 128,
        "forward_large_num_ptcl": 256,
    },
}


def _cap(base: int, num_ptcl: int, scale: int) -> str:
    return str(max(base, scale * num_ptcl * num_ptcl))


def build_cases(size_profile: str) -> dict[str, SuiteCase]:
    sizes = SIZE_PROFILES[size_profile]
    step_num_ptcl = sizes["step_num_ptcl"]
    scatter_num_ptcl = sizes["scatter_num_ptcl"]
    gravity_num_ptcl = sizes["gravity_num_ptcl"]
    forward_smoke_num_ptcl = sizes["forward_smoke_num_ptcl"]
    forward_large_num_ptcl = sizes["forward_large_num_ptcl"]

    cases = [
        SuiteCase(
            name="fft_gradients",
            script="scripts/compare_fft_gradients.py",
            description="Distributed FFT forward and gradient parity against JAX reference.",
            args=("--mesh-size", str(sizes["fft_mesh"]), *COMMON_4GPU),
        ),
        SuiteCase(
            name="modes_pipeline_gradients",
            script="scripts/compare_modes_pipeline_gradients.py",
            description="from_sigma8 -> boltzmann -> white_noise -> linear_modes forward and gradient comparison.",
            args=("--num-ptcl", str(sizes["modes_num_ptcl"]), *COMMON_SEED, *COMMON_4GPU),
        ),
        SuiteCase(
            name="lpt_gradients",
            script="scripts/compare_lpt_gradients.py",
            description="Linear modes -> LPT forward and gradient comparison with PMWD.",
            args=(
                "--num-ptcl",
                str(sizes["lpt_num_ptcl"]),
                *COMMON_SEED,
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "2.5",
                "--max-share-ptcl",
                _cap(8000, sizes["lpt_num_ptcl"], 24),
                "--max-share-gather-ptcl",
                _cap(16000, sizes["lpt_num_ptcl"], 48),
            ),
        ),
        SuiteCase(
            name="drift_gradients",
            script="scripts/compare_drift_gradients.py",
            description="Drift forward and adjoint parity on a constructed particle-crossing state.",
            args=(
                "--num-ptcl",
                str(step_num_ptcl),
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "2.0",
                "--max-share-ptcl",
                _cap(256, step_num_ptcl, 2),
                "--max-share-gather-ptcl",
                _cap(1024, step_num_ptcl, 8),
            ),
        ),
        SuiteCase(
            name="kick_gradients",
            script="scripts/compare_kick_gradients.py",
            description="Kick forward and adjoint parity against PMWD.",
            args=(
                "--num-ptcl",
                str(step_num_ptcl),
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "2.0",
                "--max-share-ptcl",
                _cap(256, step_num_ptcl, 2),
                "--max-share-gather-ptcl",
                _cap(1024, step_num_ptcl, 8),
            ),
        ),
        SuiteCase(
            name="halo_moving_gradients",
            script="scripts/compare_halo_moving_gradients.py",
            description="True halo-move VJP versus legacy replay on forced crossings.",
            args=(
                "--num-ptcl",
                str(step_num_ptcl),
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "2.5",
                "--max-share-ptcl",
                _cap(256, step_num_ptcl, 2),
                "--max-share-gather-ptcl",
                _cap(1024, step_num_ptcl, 8),
            ),
        ),
        SuiteCase(
            name="scatter_gradients",
            script="scripts/compare_scatter_gradients.py",
            description="Scatter gradient parity with residual projections.",
            args=(
                "--num-ptcl",
                str(scatter_num_ptcl),
                *COMMON_SEED,
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "1.5",
                "--max-share-ptcl",
                _cap(20000, scatter_num_ptcl, 6),
                "--max-share-gather-ptcl",
                _cap(50000, scatter_num_ptcl, 18),
            ),
        ),
        SuiteCase(
            name="gravity_gradients",
            script="scripts/compare_gravity_gradients.py",
            description="Gravity forward and disp-gradient parity. This indirectly exercises gather on 4 GPUs.",
            args=(
                "--num-ptcl",
                str(gravity_num_ptcl),
                *COMMON_SEED,
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "1.5",
                "--max-share-ptcl",
                _cap(20000, gravity_num_ptcl, 6),
                "--max-share-gather-ptcl",
                _cap(50000, gravity_num_ptcl, 18),
            ),
        ),
        SuiteCase(
            name="nbody_gradients",
            script="scripts/compare_nbody_gradients.py",
            description="End-to-end N-body forward and gradient parity against PMWD.",
            args=(
                "--num-ptcl",
                str(sizes["nbody_num_ptcl"]),
                "--target-seed",
                "0",
                "--seed",
                "1",
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--a-start",
                "0.016666666666666666",
                "--a-stop",
                "0.06666666666666667",
                "--a-nbody-maxstep",
                "0.016666666666666666",
                "--max-ptcl-factor",
                "2.5",
                "--max-share-ptcl",
                _cap(12000, sizes["nbody_num_ptcl"], 40),
                "--max-share-gather-ptcl",
                _cap(30000, sizes["nbody_num_ptcl"], 120),
            ),
        ),
        SuiteCase(
            name="forward_compare_smoke",
            script="scripts/compare_pmpp_pmwd_idxless.py",
            description="Forward-only PMWD/PMPP comparison with projection plots and layout overlays.",
            args=(
                "--num-ptcl",
                str(forward_smoke_num_ptcl),
                *COMMON_SEED,
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "1.75",
                "--max-share-ptcl",
                _cap(20000, forward_smoke_num_ptcl, 8),
                "--max-share-gather-ptcl",
                _cap(60000, forward_smoke_num_ptcl, 24),
            ),
        ),
        SuiteCase(
            name="forward_compare_large",
            script="scripts/compare_pmpp_pmwd_idxless.py",
            description="Larger forward-only PMWD/PMPP comparison for 4-GPU stress.",
            args=(
                "--num-ptcl",
                str(forward_large_num_ptcl),
                *COMMON_SEED,
                *MESH_SHAPE_ONE,
                *COMMON_4GPU,
                "--max-ptcl-factor",
                "1.75",
                "--max-share-ptcl",
                _cap(50000, forward_large_num_ptcl, 8),
                "--max-share-gather-ptcl",
                _cap(120000, forward_large_num_ptcl, 24),
            ),
        ),
    ]
    return {case.name: case for case in cases}


CASES = build_cases("baseline")
SUITES = {
    "core": [
        "fft_gradients",
        "modes_pipeline_gradients",
        "lpt_gradients",
        "drift_gradients",
        "kick_gradients",
        "halo_moving_gradients",
        "scatter_gradients",
        "gravity_gradients",
        "nbody_gradients",
    ],
    "full": [
        "fft_gradients",
        "modes_pipeline_gradients",
        "lpt_gradients",
        "drift_gradients",
        "kick_gradients",
        "halo_moving_gradients",
        "scatter_gradients",
        "gravity_gradients",
        "nbody_gradients",
        "forward_compare_smoke",
        "forward_compare_large",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), default="full")
    parser.add_argument("--case", choices=CASE_NAMES, default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--size-profile", choices=tuple(SIZE_PROFILES), default="h100")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir() -> Path:
    return REPO_ROOT / "notebooks" / "tests" / "output" / f"stage_suite_4gpu_{int(time())}"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def collect_env_report(num_devices: int, python_executable: str) -> dict[str, object]:
    try:
        import jax  # local import so the driver still works for syntax/listing modes
    except Exception as exc:
        return {
            "python": sys.version,
            "executable": python_executable,
            "repo_root": str(REPO_ROOT),
            "jax_import_error": str(exc),
        }

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    selected = gpu_devices[: min(num_devices, len(gpu_devices))]
    return {
        "timestamp_unix": int(time()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": python_executable,
        "repo_root": str(REPO_ROOT),
        "git_head": run_cmd("git", "rev-parse", "HEAD"),
        "git_status_short": run_cmd("git", "status", "--short"),
        "gpu_count": len(gpu_devices),
        "selected_gpu_count": len(selected),
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


def list_artifacts(case_dir: Path) -> list[str]:
    artifacts: list[str] = []
    for path in sorted(case_dir.rglob("*")):
        if path.is_file():
            artifacts.append(str(path.relative_to(case_dir)))
    return artifacts


def case_command(case: SuiteCase, python_executable: str, num_devices: int, case_dir: Path) -> list[str]:
    script_path = REPO_ROOT / case.script
    cmd = [python_executable, str(script_path), *case.args]
    if "--num-devices" not in case.args:
        cmd.extend(["--num-devices", str(num_devices)])
    else:
        for idx, token in enumerate(cmd):
            if token == "--num-devices" and idx + 1 < len(cmd):
                cmd[idx + 1] = str(num_devices)
                break
    cmd.extend(["--output-dir", str(case_dir)])
    return cmd


def run_one_case(case: SuiteCase, python_executable: str, num_devices: int, output_root: Path) -> dict[str, object]:
    case_dir = output_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    console_path = case_dir / "console.log"
    command = case_command(case, python_executable, num_devices, case_dir)

    started = perf_counter()
    with console_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    elapsed = perf_counter() - started

    return {
        "name": case.name,
        "description": case.description,
        "script": case.script,
        "command": command,
        "command_shell": " ".join(shlex.quote(part) for part in command),
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "case_dir": str(case_dir),
        "console_log": str(console_path),
        "artifacts": list_artifacts(case_dir),
        "status": "ok" if proc.returncode == 0 else "failed",
    }


def manifest_payload(
    case_names: list[str],
    cases: dict[str, SuiteCase],
    size_profile: str,
    python_executable: str,
    num_devices: int,
    output_root: Path,
) -> dict[str, object]:
    payload = {
        "suite_cases": [],
        "size_profile": size_profile,
        "python": python_executable,
        "num_devices": num_devices,
        "output_dir": str(output_root),
    }
    for name in case_names:
        case = cases[name]
        case_dir = output_root / case.name
        payload["suite_cases"].append(
            {
                **asdict(case),
                "command": case_command(case, python_executable, num_devices, case_dir),
            }
        )
    return payload


def main() -> int:
    args = parse_args()
    cases = build_cases(args.size_profile)

    if args.list_cases:
        for suite_name, case_names in SUITES.items():
            print(f"[{suite_name}]")
            for case_name in case_names:
                print(case_name)
        return 0

    output_root = args.output_dir if args.output_dir is not None else default_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    case_names = [args.case] if args.case else SUITES[args.suite]

    write_json(output_root / "env.json", collect_env_report(args.num_devices, args.python))
    write_json(
        output_root / "suite_manifest.json",
        manifest_payload(case_names, cases, args.size_profile, args.python, args.num_devices, output_root),
    )

    summary = {
        "suite": args.suite if args.case is None else None,
        "case": args.case,
        "python": args.python,
        "num_devices": args.num_devices,
        "size_profile": args.size_profile,
        "output_dir": str(output_root),
        "cases": [],
    }

    exit_code = 0
    for case_name in case_names:
        case = cases[case_name]
        print(f"[RUN] {case.name}")
        result = run_one_case(case, args.python, args.num_devices, output_root)
        summary["cases"].append(result)
        print(json.dumps({"case": case.name, "returncode": result["returncode"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
        if result["returncode"] != 0:
            exit_code = 1

    summary["failed_cases"] = [case["name"] for case in summary["cases"] if case["returncode"] != 0]
    write_json(output_root / "summary.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
