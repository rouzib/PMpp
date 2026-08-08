"""Process-isolated H200 qualification and low-memory forward runner.

The module deliberately imports JAX and PM++ only inside the worker process.
Allocator policy and ``CUDA_VISIBLE_DEVICES`` therefore take effect before the
first CUDA client is initialized.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

GIB = 1024**3
TARGET_RESOLUTION = 2048
TARGET_MEMORY_GIB = 125.0
HEADROOM_GIB = 15.0
MIN_OUTPUT_FREE_GIB = 48.0
INT32_MAX = 2**31 - 1
REQUIRED_PROBES = ("distributed_fft", "native_route", "fused_grid", "pallas_cic", "nbody_step")
DEFAULT_SCIENTIFIC_RESOLUTIONS = (64, 128, 256, 512)
REQUIRED_PRODUCTION_SCIENTIFIC_RESOLUTIONS = frozenset((64, 128, 256, 512))
MAX_POSITION_RMSE_CELLS = 1e-5
MIN_DENSITY_CROSS_CORRELATION = 0.999999
MAX_SHELL_POWER_RELATIVE_ERROR = 1e-4


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    """One physical GPU reported by ``nvidia-smi``."""

    index: int
    name: str
    memory_total_mib: int
    mig_mode: str


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    """One resolution/device-count qualification case."""

    resolution: int
    devices: int

    @property
    def label(self) -> str:
        return f"n{self.resolution}_d{self.devices}"


def _run_text(command: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(command, check=check, capture_output=True, text=True)
    return completed.stdout.strip()


def query_gpu_info() -> list[GPUInfo]:
    """Query physical device identity, memory, and MIG state without JAX."""
    output = _run_text([
        "nvidia-smi", "--query-gpu=index,name,memory.total,mig.mode.current", "--format=csv,noheader,nounits",
    ])
    devices = []
    for raw in output.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi GPU row: {raw!r}")
        devices.append(GPUInfo(int(fields[0]), fields[1], int(fields[2]), fields[3]))
    return devices


def parse_device_ids(value: str) -> tuple[int, ...]:
    """Parse a comma-separated physical GPU list and reject duplicates."""
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return result


def parse_case(value: str) -> CaseSpec:
    """Parse ``RESOLUTION:DEVICE_COUNT``."""
    try:
        resolution_text, devices_text = value.split(":", maxsplit=1)
        case = CaseSpec(int(resolution_text), int(devices_text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("case must have the form RESOLUTION:DEVICES") from error
    if case.resolution < 2 or case.devices < 1:
        raise argparse.ArgumentTypeError("resolution must be >=2 and device count must be positive")
    if case.resolution % case.devices:
        raise argparse.ArgumentTypeError("resolution must be divisible by the device count")
    return case


def parse_resolutions(value: str) -> tuple[int, ...]:
    """Parse a strictly increasing comma-separated resolution sequence."""
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("resolutions must be comma-separated positive integers") from error
    if not result or any(item < 2 for item in result) or tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError("resolutions must be unique, strictly increasing integers >=2")
    return result


def memory_budget_gib(memory_total_mib: int) -> float:
    """Return the hard per-device resident-memory budget."""
    total_gib = memory_total_mib / 1024
    return min(TARGET_MEMORY_GIB, total_gib - HEADROOM_GIB)


def case_memory_gate_gib(case: CaseSpec, budget_gib: float) -> float:
    """Return the conservative per-device peak gate for a qualification case."""
    if case == CaseSpec(1024, 4):
        return budget_gib / 4
    return budget_gib * (case.resolution / TARGET_RESOLUTION)**3


def bfc_memory_fraction(memory_total_mib: int, budget_gib: float) -> float:
    """Reserve a BFC pool below the process budget, leaving context space."""
    total_gib = memory_total_mib / 1024
    pool_gib = max(1.0, budget_gib - 2.0)
    return min(0.99, pool_gib / total_gib)


def recommended_particle_factor(observed_occupancy_ratios: Iterable[float]) -> float:
    """Apply the production occupancy margin to successful calibration runs."""
    ratios = [float(value) for value in observed_occupancy_ratios]
    if any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("occupancy ratios must be finite and non-negative")
    return max(1.05, max(ratios, default=0.0) + 0.02)


def scaled_communication_capacity(observed: int, source_resolution: int) -> int:
    """Scale a communication high-water mark by surface area and 25% margin."""
    if observed < 0 or source_resolution <= 0:
        raise ValueError("observed capacity and source resolution must be non-negative/positive")
    return math.ceil(observed * (TARGET_RESOLUTION / source_resolution)**2 * 1.25)


def _low_memory_high_water(report: dict[str, Any], name: str) -> int:
    """Read one required non-negative low-memory telemetry counter."""
    telemetry = report.get("telemetry") or {}
    value = telemetry.get(name)
    if value is None:
        raise RuntimeError(f"low-memory report is missing telemetry.{name}")
    value = int(value)
    if value < 0:
        raise RuntimeError(f"low-memory telemetry.{name} must be non-negative, got {value}")
    return value


def calibrated_communication_capacities(report: dict[str, Any], source_resolution: int) -> dict[str, int]:
    """Derive target capacities from one successful intermediate trajectory.

    Mesh-halo scatter/gather exchanges fixed-width mesh edges, so its dynamic
    particle-gather high-water is exactly zero.  Configuration capacities must
    nevertheless remain positive; one slot is retained for that inactive ABI.
    ``max_halo_share_ptcl`` is inactive too and is conservatively tied to the
    qualified migration capacity.
    """
    migration = _low_memory_high_water(report, "migration_high_water")
    gather = _low_memory_high_water(report, "gather_high_water")
    invalid = _low_memory_high_water(report, "routing_invalid_count")
    if invalid:
        raise RuntimeError(f"intermediate routing reported {invalid} invalid particles")
    migration_capacity = max(1, scaled_communication_capacity(migration, source_resolution))
    gather_capacity = max(1, scaled_communication_capacity(gather, source_resolution))
    return {
        "migration_high_water": migration,
        "gather_high_water": gather,
        "max_share_ptcl": migration_capacity,
        "max_halo_share_ptcl": migration_capacity,
        "max_share_gather_ptcl": gather_capacity,
    }


def particle_capacity(case: CaseSpec, factor: float) -> int:
    """Return and validate the fixed per-device particle capacity."""
    if not math.isfinite(factor) or factor < 1.05:
        raise ValueError("max_ptcl_factor must be finite and at least 1.05")
    capacity = math.ceil(case.resolution**3 / case.devices * factor)
    if capacity > INT32_MAX:
        raise ValueError(
            f"{case.label} factor {factor} needs capacity {capacity}, exceeding the signed-int32 routing limit"
        )
    return capacity


def _selected_gpu_info(all_devices: Iterable[GPUInfo], ids: tuple[int, ...]) -> list[GPUInfo]:
    by_id = {device.index: device for device in all_devices}
    missing = [device_id for device_id in ids if device_id not in by_id]
    if missing:
        raise RuntimeError(f"Unknown physical GPU ids: {missing}")
    return [by_id[device_id] for device_id in ids]


def _compute_processes() -> list[str]:
    output = _run_text([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits",
    ], check=False)
    return [line.strip() for line in output.splitlines() if line.strip() and "No running" not in line]


def _fabric_sections(output: str) -> list[str]:
    """Extract indented Fabric sections from ``nvidia-smi -q`` output."""
    lines = output.splitlines()
    sections = []
    for index, line in enumerate(lines):
        if line.strip().lower() != "fabric":
            continue
        base_indent = len(line) - len(line.lstrip())
        body = [line]
        for child in lines[index + 1:]:
            if child.strip() and len(child) - len(child.lstrip()) <= base_indent:
                break
            body.append(child)
        sections.append("\n".join(body))
    return sections


def _fabric_section_is_healthy(section: str) -> bool:
    """Accept both legacy ``Success`` and current ``NVML_SUCCESS`` status."""
    state = re.search(r"^\s*State\s*:\s*([^\r\n]+)", section, flags=re.IGNORECASE | re.MULTILINE)
    status = re.search(r"^\s*Status\s*:\s*([^\r\n]+)", section, flags=re.IGNORECASE | re.MULTILINE)
    if state is None or status is None:
        return False
    normalized_status = status.group(1).strip().lower()
    return state.group(1).strip().lower() == "completed" and normalized_status in {"success", "nvml_success"}


def _query_fabric(device_index: int) -> tuple[str, bool]:
    """Query one GPU, falling back when ``-d FABRIC`` is unsupported."""
    commands = (["nvidia-smi", "-i", str(device_index), "-q", "-d",
                 "FABRIC"], ["nvidia-smi", "-i", str(device_index), "-q"],
                )
    last_output = ""
    for command in commands:
        output = _run_text(command, check=False)
        last_output = output or last_output
        sections = _fabric_sections(output)
        if sections:
            return output, all(_fabric_section_is_healthy(section) for section in sections)
    return last_output, False


def validate_h200_node(selected: list[GPUInfo], *, require_eight: bool, allow_non_h200: bool,
                       output_dir: Path) -> dict[str, Any]:
    """Fail closed on an unsuitable production node and capture topology."""
    if require_eight and len(selected) != 8:
        raise RuntimeError(f"Production requires exactly eight selected GPUs, got {len(selected)}")
    if not allow_non_h200:
        wrong = [device.name for device in selected if "H200" not in device.name.upper()]
        if wrong:
            raise RuntimeError(f"Production requires H200 GPUs, got {wrong}")
    mig = [
        device.index for device in selected
        if device.mig_mode.strip().lower().strip("[]") not in {"disabled", "n/a", "not supported"}
    ]
    if mig:
        raise RuntimeError(f"MIG must be disabled on selected GPUs: {mig}")
    processes = _compute_processes()
    if processes:
        raise RuntimeError("Competing compute processes were found:\n" + "\n".join(processes))
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir).free / GIB
    if free_gib < MIN_OUTPUT_FREE_GIB:
        raise RuntimeError(
            f"Output filesystem has only {free_gib:.2f} GiB free; need at least {MIN_OUTPUT_FREE_GIB:.0f}"
        )
    topology = _run_text(["nvidia-smi", "topo", "-m"])
    if not allow_non_h200:
        topology_rows = {
            fields[0]: fields[1:]
            for line in topology.splitlines() if (fields := line.split()) and fields[0].startswith("GPU")
        }
        header = next((line.split() for line in topology.splitlines() if line.split() and line.split()[0] == "GPU0"),
                      None,
                      )
        # ``nvidia-smi topo -m`` does not expose a machine-readable mode.  Its
        # GPU rows are nevertheless stable enough to fail closed when any
        # selected pair is not joined by NVLink/NVSwitch.
        gpu_labels = [f"GPU{device.index}" for device in selected]
        if header is None or any(label not in topology_rows for label in gpu_labels):
            raise RuntimeError("Unable to parse the selected GPUs from nvidia-smi topology")
        for source in gpu_labels:
            row = topology_rows[source]
            for destination in gpu_labels:
                if source == destination:
                    continue
                column = int(destination.removeprefix("GPU"))
                if column >= len(row) or not row[column].startswith("NV"):
                    raise RuntimeError(f"Selected GPUs are not in one NVLink/NVSwitch domain: {source}->{destination}")
    fabric_reports = {}
    unhealthy_fabric = []
    for device in selected:
        output, healthy = _query_fabric(device.index)
        fabric_reports[str(device.index)] = output
        if not healthy:
            unhealthy_fabric.append(device.index)
    if not allow_non_h200 and unhealthy_fabric:
        raise RuntimeError(
            "NVLink fabric is not reported as Completed/Success for selected GPUs " + str(unhealthy_fabric)
        )
    driver = _run_text(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], check=False)
    return {
        "gpus": [dataclasses.asdict(device) for device in selected],
        "topology": topology,
        "fabric": fabric_reports,
        "driver_versions": sorted(set(line.strip() for line in driver.splitlines() if line.strip())),
        "platform": platform.platform(),
        "free_output_gib": free_gib,
    }


def allocator_environment(
    allocator: str, selected: list[GPUInfo], budget_gib: float, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Build a clean worker environment for one allocator policy."""
    env = dict(os.environ if base is None else base)
    for name in (
        "TF_GPU_ALLOCATOR", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_PLATFORMS",
        "JAX_SKIP_CUDA_CONSTRAINTS_CHECK",
    ):
        env.pop(name, None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device.index) for device in selected)
    if allocator == "async":
        env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif allocator == "bfc":
        min_total = min(device.memory_total_mib for device in selected)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        fraction = math.floor(bfc_memory_fraction(min_total, budget_gib) * 1_000_000) / 1_000_000
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{fraction:.6f}"
    else:
        raise ValueError(f"Unsupported allocator policy {allocator!r}")
    return env


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _latest_memory_evidence(output_root: Path) -> dict[str, Any] | None:
    """Return the newest persisted worker/probe memory report, if any."""
    candidates = []
    for name in ("supervisor_report.json", "probe_supervisor_report.json"):
        candidates.extend(output_root.rglob(name))
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        memory = report.get("resident_memory")
        if memory is None:
            continue
        return {
            "path": str(path.relative_to(output_root)),
            "status": report.get("status"),
            "stage": report.get("stage"),
            "case": report.get("case"),
            "probe": report.get("probe"),
            "allocator": report.get("allocator"),
            "memory_gate_gib": report.get("memory_gate_gib"),
            "resident_memory": memory,
        }
    return None


def _write_qualification_failure(output_root: Path, stage: str, error: BaseException) -> None:
    """Persist the failed supervisor gate without creating a success marker."""
    _atomic_json(
        output_root / "qualification_failure.json", {
            "schema": "pmpp-h200-qualification-failure-v1",
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "latest_memory_evidence": _latest_memory_evidence(output_root),
        },
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and getattr(value, "shape", object()) == ():
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _device_memory_stats(devices: Iterable[Any], *, require_peak: bool) -> list[dict[str, Any]]:
    """Capture JSON-native allocator statistics for selected devices."""
    rows = []
    for device in devices:
        stats = device.memory_stats() or {}
        if require_peak and stats.get("peak_bytes_in_use") is None:
            raise RuntimeError(f"JAX allocator peak telemetry is unavailable for {device}")
        rows.append({"device": str(device), **_jsonable(stats)})
    return rows


def _slice_json(index: Any) -> list[Any]:
    result = []
    for item in index:
        if isinstance(item, slice):
            result.append([item.start, item.stop, item.step])
        else:
            result.append(int(item))
    return result


def _save_npy_atomic(path: Path, array: Any) -> None:
    import numpy as np

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_density_artifacts(density: Any, output_dir: Path) -> dict[str, Any]:
    """Write one local density slab at a time and a small y projection."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    total = 0.0
    minimum = math.inf
    maximum = -math.inf
    finite = True
    ordered = sorted(density.addressable_shards, key=lambda shard: _slice_json(shard.index))
    for ordinal, shard in enumerate(ordered):
        host = np.asarray(jax.device_get(shard.data), dtype=np.float32)
        path = output_dir / f"density_shard_{ordinal:02d}.npy"
        _save_npy_atomic(path, host)
        total += float(np.sum(host, dtype=np.float64))
        minimum = min(minimum, float(np.min(host)))
        maximum = max(maximum, float(np.max(host)))
        finite = finite and bool(np.isfinite(host).all())
        shards.append({
            "path": path.name,
            "sha256": _sha256(path),
            "global_index": _slice_json(shard.index),
            "shape": list(host.shape),
            "device": str(shard.device),
        })
        del host
    projection = jnp.sum(density, axis=1, dtype=jnp.float32)
    projection_path = output_dir / "projection_y.npy"
    _save_npy_atomic(projection_path, np.asarray(jax.device_get(projection), dtype=np.float32))
    return {
        "format": "pmpp-density-shards-v1",
        "global_shape": list(density.shape),
        "dtype": str(density.dtype),
        "shards": shards,
        "projection_y": projection_path.name,
        "density_sum_float64": total,
        "density_min": minimum,
        "density_max": maximum,
        "density_finite": finite,
    }


def write_particle_artifacts(particles: Any, conf: Any, output_dir: Path) -> dict[str, Any]:
    """Persist canonical authoritative particle rows one addressable shard at a time."""
    import jax
    import numpy as np

    if particles.unused_index is None:
        raise RuntimeError("scientific particle artifacts require explicit authoritative validity masks")
    arrays = {
        "pmid": sorted(particles.pmid.addressable_shards, key=lambda shard: _slice_json(shard.index)),
        "disp": sorted(particles.disp.addressable_shards, key=lambda shard: _slice_json(shard.index)),
        "unused": sorted(particles.unused_index.addressable_shards, key=lambda shard: _slice_json(shard.index)),
    }
    shard_count = len(arrays["pmid"])
    if any(len(shards) != shard_count for shards in arrays.values()):
        raise RuntimeError("particle artifact arrays have inconsistent addressable shard counts")
    rows = []
    total_valid = 0
    for ordinal, (pmid_shard, disp_shard,
                  unused_shard) in enumerate(zip(arrays["pmid"], arrays["disp"], arrays["unused"])):
        indices = tuple(_slice_json(shard.index)[0] for shard in (pmid_shard, disp_shard, unused_shard))
        if len(set(json.dumps(index) for index in indices)) != 1:
            raise RuntimeError("particle artifact arrays have misaligned shard indices")
        unused = np.asarray(jax.device_get(unused_shard.data), dtype=np.bool_)
        valid = ~unused
        valid_count = int(np.count_nonzero(valid))
        pmid = np.asarray(jax.device_get(pmid_shard.data), dtype=np.int16)[valid]
        disp = np.asarray(jax.device_get(disp_shard.data), dtype=np.float32)[valid]
        pmid = np.ascontiguousarray(pmid)
        disp = np.ascontiguousarray(disp)
        pmid_path = output_dir / f"particle_pmid_shard_{ordinal:02d}.npy"
        disp_path = output_dir / f"particle_disp_shard_{ordinal:02d}.npy"
        _save_npy_atomic(pmid_path, pmid)
        _save_npy_atomic(disp_path, disp)
        rows.append({
            "global_index": indices[0],
            "valid_count": valid_count,
            "pmid_path": pmid_path.name,
            "pmid_sha256": _sha256(pmid_path),
            "disp_path": disp_path.name,
            "disp_sha256": _sha256(disp_path),
            "device": str(pmid_shard.device),
        })
        total_valid += valid_count
        del unused, valid, pmid, disp
    if total_valid != int(conf.ptcl_num):
        raise RuntimeError(f"particle artifact count mismatch: {total_valid} != {conf.ptcl_num}")
    return {
        "format": "pmpp-authoritative-particles-v1",
        "row_order": "valid rows in canonical authoritative array order",
        "pmid_dtype": "int16",
        "disp_dtype": "float32",
        "dimensions": int(conf.dim),
        "cell_size": float(conf.cell_size),
        "valid_particles": total_valid,
        "shards": rows,
    }


def write_power_spectrum_artifact(density: Any, conf: Any, output_dir: Path) -> dict[str, Any]:
    """Compute the distributed shell power in the worker and persist only 1D arrays."""
    import jax
    import numpy as np

    from .analysis import density_to_pk

    k, power, nmodes = density_to_pk(density, conf, mas="CIC")
    k, power, nmodes = jax.device_get((k, power, nmodes))
    arrays = {
        "k": np.asarray(k, dtype=np.float32),
        "power": np.asarray(power, dtype=np.float32),
        "nmodes": np.asarray(nmodes, dtype=np.int32),
    }
    paths = {}
    for name, values in arrays.items():
        path = output_dir / f"density_power_{name}.npy"
        _save_npy_atomic(path, values)
        paths[name] = {"path": path.name, "sha256": _sha256(path), "dtype": str(values.dtype), }
    return {
        "format": "pmpp-shell-power-v1",
        "mass_assignment": "CIC",
        "shells": int(arrays["k"].size),
        "arrays": paths,
    }


def _package_versions() -> dict[str, dict[str, str | None]]:
    names = ("pmpp", "jax", "jaxlib", "jax-cuda12-plugin", "nvidia-nccl-cu12", "numpy")
    versions = {}
    for name in names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        spec = importlib.util.find_spec(name.replace("-", "_"))
        versions[name] = {"version": version, "origin": None if spec is None else spec.origin, }
    return versions


def _repository_commit() -> str | None:
    """Return the source checkout commit without making Git a requirement."""
    try:
        value = _run_text(["git", "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def write_nbody_compilation_artifacts(particles: Any, cosmo: Any, conf: Any, output_dir: Path) -> dict[str, Any]:
    """Persist whole-scan HLO, buffer assignment, and compiled memory stats."""
    from .nbody.solver import lower_nbody_low_memory

    # The actual low-memory N-body input from LPT has no acceleration.  Lower
    # that signature, not the final-state signature that could claim a reuse
    # opportunity unavailable at solver entry.
    lowered = lower_nbody_low_memory(particles.replace(acc=None), cosmo, conf)
    compiled = lowered.compile()
    analysis = compiled.memory_analysis()
    fields = (
        "generated_code_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
        "temp_size_in_bytes", "host_generated_code_size_in_bytes", "host_argument_size_in_bytes",
        "host_output_size_in_bytes", "host_alias_size_in_bytes", "host_temp_size_in_bytes",
    )
    memory = {name: int(getattr(analysis, name)) for name in fields}
    hlo = compiled.as_text()
    hlo_path = output_dir / "nbody_low_memory.hlo.txt"
    assignment_path = output_dir / "nbody_low_memory.buffer_assignment.pb"
    _atomic_text(hlo_path, hlo)
    assignment = bytes(getattr(analysis, "serialized_buffer_assignment_proto", b""))
    _atomic_bytes(assignment_path, assignment)
    lowered_text = hlo.lower()

    local_disp = particles.disp.addressable_shards[0].data
    local_vel = particles.vel.addressable_shards[0].data
    required_state_alias_bytes = (
        math.prod(local_disp.shape) * local_disp.dtype.itemsize + math.prod(local_vel.shape) * local_vel.dtype.itemsize
    )
    typed_shapes = []
    for dtype, dimensions in re.findall(r"\b(f32|c64)\[([0-9,]+)\]", lowered_text):
        shape = tuple(int(value) for value in dimensions.split(","))
        typed_shapes.append((dtype, shape))
    local_mesh_elements = math.prod(int(value) for value in conf.local_mesh_shape)
    stacked_component_fields = sorted({
        f"{dtype}[{','.join(str(value) for value in shape)}]"
        for dtype, shape in typed_shapes
        if len(shape) == 4 and 3 in shape and math.prod(value for value in shape
                                                        if value != 3) >= max(1, local_mesh_elements // 4)
    })
    irfft_lines = [line.lower() for line in hlo.splitlines() if "fft_type=irfft" in line.lower()]
    irfft_shapes = []
    irfft_branches = set()
    for line in irfft_lines:
        match = re.search(r"\b(?:f32|f64)\[([0-9,]+)\]", line)
        if match is not None:
            irfft_shapes.append(tuple(int(value) for value in match.group(1).split(",")))
        branch = re.search(r"branch_([012])_fun", line)
        if branch is not None:
            irfft_branches.add(int(branch.group(1)))
    streamed_updates = sum(
        "dynamic_update_slice" in line and "gravity.py" in line for line in lowered_text.splitlines()
    )
    checks = {
        "particle_state_alias_bytes": memory["alias_size_in_bytes"] >= required_state_alias_bytes,
        "streamed_component_loop": streamed_updates > 0 and len(irfft_lines) >= 3 and len(irfft_lines) % 3 == 0,
        "scalar_irfft_outputs": bool(irfft_shapes) and all(len(shape) == 3 for shape in irfft_shapes),
        "no_stacked_component_mesh": not stacked_component_fields,
    }
    failed = [name for name, passed in checks.items() if not passed]
    evidence = {
        "memory_analysis": memory,
        "required_state_alias_bytes": required_state_alias_bytes,
        "hlo_path": hlo_path.name,
        "hlo_sha256": _sha256(hlo_path),
        "buffer_assignment_path": assignment_path.name,
        "buffer_assignment_sha256": _sha256(assignment_path),
        "buffer_assignment_bytes": len(assignment),
        "hlo_operation_counts": {
            "all_to_all": lowered_text.count("all-to-all") + lowered_text.count("all_to_all"),
            "fft": lowered_text.count("fft("),
            "irfft": len(irfft_lines),
            "streamed_acceleration_updates": streamed_updates,
        },
        "irfft_output_shapes": [list(shape) for shape in sorted(set(irfft_shapes))],
        "irfft_component_branches": sorted(irfft_branches),
        "stacked_component_fields": stacked_component_fields,
        "checks": checks,
        "passed": not failed,
    }
    if failed:
        raise RuntimeError("low-memory N-body compilation evidence failed: " + ", ".join(failed))
    return evidence


def _worker_devices(args: argparse.Namespace) -> tuple[Any, ...]:
    """Return and validate the exact visible worker GPU set."""
    import jax

    devices = tuple(jax.devices("gpu"))
    if len(devices) != args.worker_devices:
        raise RuntimeError(f"Worker expected exactly {args.worker_devices} visible GPUs, found {devices}")
    if not args.allow_non_h200 and any(float(getattr(device, "compute_capability", 0)) != 9.0 for device in devices):
        raise RuntimeError(f"Production worker requires sm_90 H200 devices, found {devices}")
    return devices


def _build_worker_configuration(
    args: argparse.Namespace, devices: tuple[Any, ...], *, one_nbody_step: bool = False,
) -> Any:
    """Build the exact-capacity configuration shared by workers and probes."""
    import jax.numpy as jnp

    from .core import Configuration
    from .distributed import MultiGPUConfiguration, create_compute_mesh

    resolution = int(args.worker_resolution)
    if resolution % len(devices):
        raise ValueError(f"resolution {resolution} is not divisible by {len(devices)} devices")
    max_ptcl_per_slice = particle_capacity(CaseSpec(resolution, len(devices)), args.max_ptcl_factor)
    a_start = 1 / 64
    a_stop = 2 / 64 if one_nbody_step else 1.0
    return Configuration(
        args.box_size / resolution, (resolution, ) * 3, mesh_shape=1, multigpu=MultiGPUConfiguration(
            compute_mesh=create_compute_mesh(devices), mode="mesh_halo", cuda_routing=True,
            cuda_routing_backend="bidir_mergepath",
        ), float_dtype=jnp.float32, pmid_dtype=jnp.int16, lpt_order=2, lpt_cache_strains=False, a_start=a_start,
        a_stop=a_stop, a_nbody_maxstep=1 / 64, max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=args.max_share_ptcl, max_halo_share_ptcl=args.max_halo_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
    )


def _validate_native_worker(conf: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed on routing activation, ABI, and production architecture."""
    from .distributed import extension_status

    if not bool(conf.cuda_routing) or conf.cuda_routing_backend != "bidir_mergepath":
        raise RuntimeError(
            "Low-memory production run requires active native bidir_mergepath routing; fallback detected"
        )
    status = extension_status()
    if int(status.get("record_format_version", -1)) != 3:
        raise RuntimeError(f"Native routing ABI v3 is required, got {status}")
    if not status.get("fused_primal_feature") or not status.get("fused_primal_registered"):
        raise RuntimeError("Native routing must manifest and register fused_drift_primal_i16_f32")
    if getattr(conf, "mGPU_halo_moving_low_memory", None) is None:
        raise RuntimeError("Native fused low-memory drift-route capability is unavailable for this configuration")
    architectures = {str(value).lower().removeprefix("sm_") for value in status.get("embedded_architectures", ())}
    if not args.allow_non_h200 and "90" not in architectures:
        raise RuntimeError(f"Native routing library must embed sm_90 code, got {sorted(architectures)}")
    return status


def _prepared_cosmology(args: argparse.Namespace, conf: Any) -> Any:
    """Build the worker cosmology with transfer and growth tables."""
    from .cosmology import Cosmology, boltzmann

    cosmo = Cosmology.from_sigma8(
        conf, args.sigma8, n_s=args.n_s, Omega_m=args.omega_m, Omega_b=args.omega_b, h=args.h,
    )
    return boltzmann(cosmo, conf)


def probe_worker_main(args: argparse.Namespace) -> int:
    """Run one exact-shape qualification probe in an isolated process."""
    from .qualification_probes import run_probe

    if args.probe_name not in REQUIRED_PROBES:
        raise ValueError(f"unknown probe {args.probe_name!r}")
    devices = _worker_devices(args)
    one_step = args.probe_name == "nbody_step"
    conf = _build_worker_configuration(args, devices, one_nbody_step=one_step)
    status = _validate_native_worker(conf, args)
    prepared_cosmo = _prepared_cosmology(args, conf) if one_step else None
    started = time.perf_counter()
    evidence = run_probe(
        args.probe_name, conf, prepared_cosmo=prepared_cosmo, route_migrants=args.probe_route_migrants,
    )
    elapsed = time.perf_counter() - started
    report = {
        "schema": "pmpp-h200-probe-v1",
        "status": "passed",
        "probe": args.probe_name,
        "case": {
            "resolution": int(args.worker_resolution),
            "devices": len(devices),
        },
        "capacity": {
            "max_ptcl_factor": float(args.max_ptcl_factor),
            "max_ptcl_per_slice": int(conf.max_ptcl_per_slice),
            "max_share_ptcl": int(conf.max_share_ptcl),
            "max_halo_share_ptcl": int(conf.max_halo_share_ptcl),
            "max_share_gather_ptcl": int(conf.max_share_gather_ptcl),
        },
        "elapsed_seconds": elapsed,
        "evidence": _jsonable(evidence),
        "routing": {
            "active": bool(conf.cuda_routing),
            "backend": conf.cuda_routing_backend,
            "extension": _jsonable(status),
        },
        "versions": _package_versions(),
        "repository_commit": _repository_commit(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES", "TF_GPU_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
            )
        },
    }
    _atomic_json(args.output_dir / "probe_report.json", report)
    return 0


def worker_main(args: argparse.Namespace) -> int:
    """Execute one case after the supervisor has fixed the environment."""
    import jax
    import jax.numpy as jnp

    from .forward import run_forward

    devices = _worker_devices(args)
    resolution = int(args.worker_resolution)
    conf = _build_worker_configuration(args, devices)
    local_particles = int(conf.ptcl_num // len(devices))
    max_ptcl_per_slice = int(conf.max_ptcl_per_slice)
    status = _validate_native_worker(conf, args)
    cosmo = _prepared_cosmology(args, conf)
    for device in devices:
        reset = getattr(device, "reset_peak_memory_stats", None)
        if reset is not None:
            reset()
    started = time.perf_counter()
    result = run_forward(
        args.seed, cosmo, conf, profile=args.worker_profile, noise_mode=args.worker_noise_mode, retain_particles=True,
    )
    density = result.density
    particles = result.particles
    if particles is None:
        raise RuntimeError("Worker requires retained particles for final capacity validation")
    jax.block_until_ready((particles, density))
    elapsed = time.perf_counter() - started
    # Both profiles retain the same particles and density here. Snapshot the
    # allocator before validation, export, or low-memory-only HLO evidence.
    forward_jax_memory = _device_memory_stats(devices, require_peak=True)
    per_device_valid = []
    count_valid = jax.jit(lambda unused: jnp.sum(~unused, dtype=jnp.int32))
    for shard in particles.unused_index.addressable_shards:
        count = count_valid(shard.data)
        per_device_valid.append(int(jax.device_get(count)))
    valid_particles = sum(per_device_valid)
    if valid_particles != conf.ptcl_num:
        raise RuntimeError(f"Particle count mismatch: {valid_particles} != {conf.ptcl_num}")
    if args.worker_profile == "low_memory":
        migration_high_water = result.telemetry.migration_high_water
        gather_high_water = result.telemetry.gather_high_water
        invalid_count = result.telemetry.routing_invalid_count
        if migration_high_water is None or gather_high_water is None or invalid_count is None:
            raise RuntimeError("low-memory worker did not return complete routing high-water telemetry")
        if invalid_count:
            raise RuntimeError(f"native routing reported {invalid_count} invalid particles")
        if migration_high_water > int(conf.max_share_ptcl):
            raise RuntimeError(
                f"migration high-water {migration_high_water} exceeded capacity {int(conf.max_share_ptcl)}"
            )
        if gather_high_water != 0:
            raise RuntimeError("mesh-halo low-memory worker unexpectedly reported a particle gather high-water")

    @jax.jit
    def state_is_finite(disp, vel, acc):
        return jnp.all(jnp.isfinite(disp)) & jnp.all(jnp.isfinite(vel)) & jnp.all(jnp.isfinite(acc))

    particle_state_finite = bool(jax.device_get(state_is_finite(particles.disp, particles.vel, particles.acc)))
    if not particle_state_finite:
        raise RuntimeError("Final particle state contains non-finite values")
    artifacts = None
    if args.save_density:
        artifacts = write_density_artifacts(density, args.output_dir)
    else:

        @jax.jit
        def density_health(value):
            return jnp.sum(value, dtype=jnp.float32), jnp.all(jnp.isfinite(value))

        density_sum, density_finite = jax.device_get(density_health(density))
        artifacts = {
            "format": None,
            "global_shape": list(density.shape),
            "dtype": str(density.dtype),
            "density_sum_float32": float(density_sum),
            "density_finite": bool(density_finite),
        }
    particle_artifacts = None
    power_spectrum_artifact = None
    if args.save_scientific_artifacts:
        if not args.save_density:
            raise RuntimeError("scientific artifacts require density shard output")
        particle_artifacts = write_particle_artifacts(particles, conf, args.output_dir)
        power_spectrum_artifact = write_power_spectrum_artifact(density, conf, args.output_dir)
    compilation = None
    if args.worker_profile == "low_memory" and not args.skip_compilation_artifacts:
        compilation = write_nbody_compilation_artifacts(particles, cosmo, conf, args.output_dir)
    memory = _device_memory_stats(devices, require_peak=True)
    report = {
        "case": {
            "resolution": resolution,
            "devices": len(devices)
        },
        "physics": {
            "profile": args.worker_profile,
            "noise_mode": args.worker_noise_mode,
            "seed": args.seed,
            "box_size": args.box_size,
            "omega_m": args.omega_m,
            "sigma8": args.sigma8,
            "n_s": args.n_s,
            "omega_b": args.omega_b,
            "h": args.h,
            "lpt_order": 2,
            "nbody_steps": int(conf.a_nbody_num),
        },
        "capacity": {
            "max_ptcl_factor": args.max_ptcl_factor,
            "max_ptcl_per_slice": max_ptcl_per_slice,
            "max_share_ptcl": int(conf.max_share_ptcl),
            "max_halo_share_ptcl": int(conf.max_halo_share_ptcl),
            "max_share_gather_ptcl": int(conf.max_share_gather_ptcl),
            "per_device_final_valid": per_device_valid,
            "final_occupancy_ratio": max(per_device_valid) / local_particles,
            "max_authoritative_occupancy": result.telemetry.max_authoritative_occupancy,
            "occupancy_high_water_ratio": result.telemetry.max_authoritative_occupancy / local_particles,
            "migration_high_water": result.telemetry.migration_high_water,
            "gather_high_water": result.telemetry.gather_high_water,
        },
        "routing": {
            "active": bool(conf.cuda_routing),
            "backend": conf.cuda_routing_backend,
            "extension": status,
            "invalid_count": result.telemetry.routing_invalid_count,
        },
        "versions": _package_versions(),
        "pmpp_commit": _repository_commit(),
        "elapsed_seconds": elapsed,
        "telemetry": _jsonable(result.telemetry),
        "forward_jax_memory": forward_jax_memory,
        "jax_memory": memory,
        "compilation_artifacts_skipped": bool(args.skip_compilation_artifacts),
        "artifacts": artifacts,
        "particle_artifacts": particle_artifacts,
        "power_spectrum_artifact": power_spectrum_artifact,
        "compilation": compilation,
        "valid_particles": valid_particles,
        "expected_particles": conf.ptcl_num,
        "particle_state_finite": particle_state_finite,
        "backend_platform_version": getattr(devices[0].client, "platform_version", None),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES", "TF_GPU_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
            )
        },
    }
    _atomic_json(args.output_dir / "worker_report.json", report)
    return 0


def _start_memory_sampler(path: Path) -> subprocess.Popen[str] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,memory.used", "--format=csv,noheader,nounits", "-lms", "100",
        ], stdout=handle, stderr=subprocess.STDOUT, text=True)
    except OSError:
        handle.close()
        return None
    process._pmpp_output_handle = handle  # type: ignore[attr-defined]
    return process


def _stop_memory_sampler(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    handle = getattr(process, "_pmpp_output_handle", None)
    if handle is not None:
        handle.close()


def parse_memory_trace(path: Path, physical_ids: tuple[int, ...]) -> dict[str, Any]:
    """Summarize a 100 ms nvidia-smi trace for selected physical GPUs."""
    maxima = {device_id: 0 for device_id in physical_ids}
    sample_counts = {device_id: 0 for device_id in physical_ids}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                device_id = int(fields[1])
                used_mib = int(fields[2])
            except ValueError:
                continue
            if device_id in maxima:
                maxima[device_id] = max(maxima[device_id], used_mib)
                sample_counts[device_id] += 1
    return {
        "samples": sum(sample_counts.values()),
        "samples_per_gpu": {
            str(key): value
            for key, value in sample_counts.items()
        },
        "max_used_mib": {
            str(key): value
            for key, value in maxima.items()
        },
        "max_used_gib": max(maxima.values(), default=0) / 1024,
        "trace_path": str(path),
    }


def _worker_command(
    args: argparse.Namespace, case: CaseSpec, output_dir: Path, save_density: bool, *, profile: str = "low_memory",
    noise_mode: str = "nested", save_scientific_artifacts: bool = False,
) -> list[str]:
    command = [
        sys.executable, "-m", "pmpp.forward_cli", "--worker", "--worker-resolution",
        str(case.resolution), "--worker-devices",
        str(case.devices), "--output-dir",
        str(output_dir), "--seed",
        str(args.seed), "--box-size",
        str(args.box_size), "--omega-m",
        str(args.omega_m), "--sigma8",
        str(args.sigma8), "--n-s",
        str(args.n_s), "--omega-b",
        str(args.omega_b), "--h",
        str(args.h), "--max-ptcl-factor",
        str(args.max_ptcl_factor), "--max-share-ptcl",
        str(args.max_share_ptcl), "--max-halo-share-ptcl",
        str(args.max_halo_share_ptcl), "--max-share-gather-ptcl",
        str(args.max_share_gather_ptcl), "--worker-profile", profile, "--worker-noise-mode", noise_mode,
    ]
    if save_density:
        command.append("--save-density")
    if save_scientific_artifacts:
        command.append("--save-scientific-artifacts")
    if args.skip_compilation_artifacts:
        command.append("--skip-compilation-artifacts")
    if args.allow_non_h200:
        command.append("--allow-non-h200")
    return command


def _probe_worker_command(args: argparse.Namespace, probe_name: str, case: CaseSpec, output_dir: Path, ) -> list[str]:
    """Build one isolated exact-shape probe command."""
    command = [
        sys.executable, "-m", "pmpp.forward_cli", "--probe-worker", "--probe-name", probe_name, "--worker-resolution",
        str(case.resolution), "--worker-devices",
        str(case.devices), "--output-dir",
        str(output_dir), "--seed",
        str(args.seed), "--box-size",
        str(args.box_size), "--omega-m",
        str(args.omega_m), "--sigma8",
        str(args.sigma8), "--n-s",
        str(args.n_s), "--omega-b",
        str(args.omega_b), "--h",
        str(args.h), "--max-ptcl-factor",
        str(args.max_ptcl_factor), "--max-share-ptcl",
        str(args.max_share_ptcl), "--max-halo-share-ptcl",
        str(args.max_halo_share_ptcl), "--max-share-gather-ptcl",
        str(args.max_share_gather_ptcl),
    ]
    if args.probe_route_migrants is not None:
        command.extend(("--probe-route-migrants", str(args.probe_route_migrants)))
    if args.allow_non_h200:
        command.append("--allow-non-h200")
    return command


def launch_probe(
    args: argparse.Namespace, probe_name: str, case: CaseSpec, allocator: str, selected: list[GPUInfo],
    budget_gib: float, output_dir: Path,
) -> dict[str, Any]:
    """Launch one required probe with independent allocator and memory state."""
    if probe_name not in REQUIRED_PROBES:
        raise ValueError(f"unknown required probe {probe_name!r}")
    case_selected = selected[:case.devices]
    if len(case_selected) != case.devices:
        raise RuntimeError(f"Probe {probe_name} needs {case.devices} GPUs")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = allocator_environment(allocator, case_selected, budget_gib)
    trace_path = output_dir / "nvidia_smi_memory.csv"
    sampler = _start_memory_sampler(trace_path)
    if sampler is None:
        raise RuntimeError(f"Unable to start the required memory sampler for probe {probe_name}")
    started = time.perf_counter()
    try:
        with (output_dir /
              "stdout.log").open("w",
                                 encoding="utf-8") as stdout, (output_dir /
                                                               "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                _probe_worker_command(args, probe_name, case, output_dir), env=env, stdout=stdout, stderr=stderr,
                text=True,
            )
    finally:
        elapsed = time.perf_counter() - started
        _stop_memory_sampler(sampler)
    memory = parse_memory_trace(trace_path, tuple(device.index for device in case_selected))
    supervisor_report = {
        "status": "failed",
        "probe": probe_name,
        "case": dataclasses.asdict(case),
        "allocator": allocator,
        "worker_exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "resident_memory": memory,
        "memory_gate_gib": budget_gib,
    }
    missing_samples = [device.index for device in case_selected if memory["samples_per_gpu"][str(device.index)] == 0]
    if missing_samples:
        supervisor_report.update(stage="memory_trace", error=f"missing GPU samples: {missing_samples}")
        _atomic_json(output_dir / "probe_supervisor_report.json", supervisor_report)
        raise RuntimeError(f"Probe {probe_name} has no memory samples for GPUs {missing_samples}")
    if completed.returncode:
        supervisor_report.update(stage="worker", error=f"worker exit {completed.returncode}")
        _atomic_json(output_dir / "probe_supervisor_report.json", supervisor_report)
        raise RuntimeError(
            f"Probe {probe_name}/{case.label}/{allocator} failed with exit {completed.returncode}; "
            f"see {output_dir / 'stderr.log'}"
        )
    report_path = output_dir / "probe_report.json"
    if not report_path.is_file():
        supervisor_report.update(stage="probe_report", error="probe report is missing")
        _atomic_json(output_dir / "probe_supervisor_report.json", supervisor_report)
        raise RuntimeError(f"Probe {probe_name} did not produce {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("probe") != probe_name:
        supervisor_report.update(stage="probe_report", error="probe evidence is inconsistent")
        _atomic_json(output_dir / "probe_supervisor_report.json", supervisor_report)
        raise RuntimeError(f"Probe {probe_name} returned inconsistent evidence")
    report.update(
        allocator=allocator, supervisor_elapsed_seconds=elapsed, resident_memory=memory, memory_gate_gib=budget_gib,
        memory_gate_passed=memory["max_used_gib"] <= budget_gib,
    )
    if not report["memory_gate_passed"]:
        supervisor_report.update(
            stage="memory_gate", error=f"peak {memory['max_used_gib']:.2f} GiB exceeded {budget_gib:.2f} GiB",
        )
        _atomic_json(output_dir / "probe_supervisor_report.json", supervisor_report)
        raise RuntimeError(f"Probe {probe_name} peak {memory['max_used_gib']:.2f} GiB exceeded {budget_gib:.2f} GiB")
    supervisor_report.update(status="passed", stage="complete", error=None)
    _atomic_json(output_dir / "probe_supervisor_report.json", supervisor_report)
    _atomic_json(report_path, report)
    return report


def run_required_probes(
    args: argparse.Namespace, case: CaseSpec, selected: list[GPUInfo], allocator: str, budget_gib: float,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Run every required probe in order and stop at the first failure."""
    reports = []
    for probe_name in REQUIRED_PROBES:
        reports.append(
            launch_probe(
                args, probe_name, case, allocator, selected, budget_gib, output_root / "probes" / probe_name,
            )
        )
    _atomic_json(
        output_root / "probes_summary.json", {
            "status": "passed",
            "case": dataclasses.asdict(case),
            "allocator": allocator,
            "probes": reports,
        },
    )
    return reports


def launch_case(
    args: argparse.Namespace, case: CaseSpec, allocator: str, selected: list[GPUInfo], budget_gib: float,
    output_dir: Path, *, save_density: bool, profile: str = "low_memory", noise_mode: str = "nested",
    memory_gate_override_gib: float | None = None, save_scientific_artifacts: bool = False,
) -> dict[str, Any]:
    """Launch one isolated worker and attach external memory telemetry."""
    case_selected = selected[:case.devices]
    if len(case_selected) != case.devices:
        raise RuntimeError(f"Case {case.label} needs {case.devices} GPUs")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = allocator_environment(allocator, case_selected, budget_gib)
    trace_path = output_dir / "nvidia_smi_memory.csv"
    sampler = _start_memory_sampler(trace_path)
    if sampler is None:
        _atomic_json(
            output_dir / "supervisor_report.json", {
                "status": "failed",
                "stage": "memory_sampler_start",
                "case": dataclasses.asdict(case),
                "allocator": allocator,
            },
        )
        raise RuntimeError("Unable to start the required 100 ms nvidia-smi memory sampler")
    started = time.perf_counter()
    try:
        with (output_dir /
              "stdout.log").open("w",
                                 encoding="utf-8") as stdout, (output_dir /
                                                               "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                _worker_command(
                    args, case, output_dir, save_density, profile=profile, noise_mode=noise_mode,
                    save_scientific_artifacts=save_scientific_artifacts,
                ), env=env, stdout=stdout, stderr=stderr, text=True,
            )
    finally:
        elapsed = time.perf_counter() - started
        _stop_memory_sampler(sampler)
    memory = parse_memory_trace(trace_path, tuple(device.index for device in case_selected))
    gate = case_memory_gate_gib(case, budget_gib) if memory_gate_override_gib is None else memory_gate_override_gib
    supervisor_report = {
        "status": "failed",
        "case": dataclasses.asdict(case),
        "allocator": allocator,
        "worker_exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "resident_memory": memory,
        "memory_gate_gib": gate,
    }
    missing_samples = [
        device_id for device_id in (device.index for device in case_selected)
        if memory["samples_per_gpu"][str(device_id)] == 0
    ]
    if missing_samples:
        supervisor_report.update(stage="memory_trace", error=f"missing GPU samples: {missing_samples}")
        _atomic_json(output_dir / "supervisor_report.json", supervisor_report)
        raise RuntimeError(f"Memory sampler produced no samples for selected GPUs {missing_samples}")
    if completed.returncode:
        supervisor_report.update(stage="worker", error=f"worker exit {completed.returncode}")
        _atomic_json(output_dir / "supervisor_report.json", supervisor_report)
        raise RuntimeError(
            f"Worker {case.label}/{allocator} failed with exit {completed.returncode}; see {output_dir / 'stderr.log'}"
        )
    report_path = output_dir / "worker_report.json"
    if not report_path.is_file():
        supervisor_report.update(stage="worker_report", error="worker report is missing")
        _atomic_json(output_dir / "supervisor_report.json", supervisor_report)
        raise RuntimeError(f"Worker did not produce {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["allocator"] = allocator
    report["supervisor_elapsed_seconds"] = elapsed
    report["resident_memory"] = memory
    report["memory_gate_gib"] = gate
    report["memory_gate_passed"] = memory["max_used_gib"] <= gate
    if not report["memory_gate_passed"]:
        supervisor_report.update(
            stage="memory_gate", error=f"peak {memory['max_used_gib']:.2f} GiB exceeded {gate:.2f} GiB",
        )
        _atomic_json(output_dir / "supervisor_report.json", supervisor_report)
        raise RuntimeError(f"{case.label} peak {memory['max_used_gib']:.2f} GiB exceeded gate {gate:.2f} GiB")
    supervisor_report.update(status="passed", stage="complete", error=None)
    _atomic_json(output_dir / "supervisor_report.json", supervisor_report)
    return report


def compare_density_artifacts(
    standard_report: dict[str, Any], low_memory_report: dict[str, Any], standard_dir: Path, low_memory_dir: Path, *,
    atol: float, rtol: float, relative_l2_tolerance: float, correlation_minimum: float = MIN_DENSITY_CROSS_CORRELATION,
) -> dict[str, Any]:
    """Stream density shards and enforce a centered field cross-correlation."""
    import numpy as np

    standard_artifacts = standard_report.get("artifacts") or {}
    low_memory_artifacts = low_memory_report.get("artifacts") or {}
    if standard_artifacts.get("format") != "pmpp-density-shards-v1":
        raise RuntimeError("standard scientific gate is missing density shards")
    if low_memory_artifacts.get("format") != "pmpp-density-shards-v1":
        raise RuntimeError("low-memory scientific gate is missing density shards")
    if standard_artifacts.get("global_shape") != low_memory_artifacts.get("global_shape"):
        raise RuntimeError("scientific gate density shapes differ")
    standard_shards = sorted(standard_artifacts["shards"], key=lambda row: row["global_index"])
    low_memory_shards = sorted(low_memory_artifacts["shards"], key=lambda row: row["global_index"])
    if len(standard_shards) != len(low_memory_shards):
        raise RuntimeError("scientific gate shard counts differ")
    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    allclose = True
    compared_values = 0
    standard_mean = 0.0
    low_memory_mean = 0.0
    standard_m2 = 0.0
    low_memory_m2 = 0.0
    covariance = 0.0
    chunk_values = 2**20
    for standard_row, low_memory_row in zip(standard_shards, low_memory_shards):
        if standard_row["global_index"] != low_memory_row["global_index"]:
            raise RuntimeError("scientific gate shard global indices differ")
        standard = np.load(standard_dir / standard_row["path"], mmap_mode="r", allow_pickle=False).reshape(-1)
        low_memory = np.load(low_memory_dir / low_memory_row["path"], mmap_mode="r", allow_pickle=False).reshape(-1)
        if standard.shape != low_memory.shape:
            raise RuntimeError("scientific gate local shard shapes differ")
        for start in range(0, standard.size, chunk_values):
            stop = min(start + chunk_values, standard.size)
            standard_chunk = np.asarray(standard[start:stop], dtype=np.float64)
            low_memory_chunk = np.asarray(low_memory[start:stop], dtype=np.float64)
            difference = low_memory_chunk - standard_chunk
            max_abs = max(max_abs, float(np.max(np.abs(difference), initial=0)))
            squared_error += float(np.dot(difference, difference))
            squared_reference += float(np.dot(standard_chunk, standard_chunk))
            allowed = atol + rtol * np.maximum(np.abs(standard_chunk), np.abs(low_memory_chunk))
            allclose = allclose and bool(np.all(np.abs(difference) <= allowed))
            chunk_count = int(difference.size)
            chunk_standard_mean = float(np.mean(standard_chunk))
            chunk_low_memory_mean = float(np.mean(low_memory_chunk))
            standard_centered = standard_chunk - chunk_standard_mean
            low_memory_centered = low_memory_chunk - chunk_low_memory_mean
            chunk_standard_m2 = float(np.dot(standard_centered, standard_centered))
            chunk_low_memory_m2 = float(np.dot(low_memory_centered, low_memory_centered))
            chunk_covariance = float(np.dot(standard_centered, low_memory_centered))
            combined_count = compared_values + chunk_count
            if compared_values:
                standard_delta = chunk_standard_mean - standard_mean
                low_memory_delta = chunk_low_memory_mean - low_memory_mean
                cross_weight = compared_values * chunk_count / combined_count
                standard_m2 += chunk_standard_m2 + standard_delta**2 * cross_weight
                low_memory_m2 += chunk_low_memory_m2 + low_memory_delta**2 * cross_weight
                covariance += chunk_covariance + standard_delta * low_memory_delta * cross_weight
                standard_mean += standard_delta * chunk_count / combined_count
                low_memory_mean += low_memory_delta * chunk_count / combined_count
            else:
                standard_mean = chunk_standard_mean
                low_memory_mean = chunk_low_memory_mean
                standard_m2 = chunk_standard_m2
                low_memory_m2 = chunk_low_memory_m2
                covariance = chunk_covariance
            compared_values = combined_count
    relative_l2 = math.sqrt(squared_error / max(squared_reference, sys.float_info.min))
    correlation_denominator = math.sqrt(max(standard_m2, 0.0) * max(low_memory_m2, 0.0))
    if correlation_denominator > 0:
        cross_correlation = max(-1.0, min(1.0, covariance / correlation_denominator))
    else:
        cross_correlation = 1.0 if squared_error == 0 else float("-inf")
    passed = math.isfinite(cross_correlation) and cross_correlation >= correlation_minimum
    return {
        "passed": passed,
        "global_shape": standard_artifacts["global_shape"],
        "compared_values": compared_values,
        "max_abs_error": max_abs,
        "relative_l2_error": relative_l2,
        "density_cross_correlation": cross_correlation,
        "density_cross_correlation_minimum": float(correlation_minimum),
        "standard_mean": standard_mean,
        "low_memory_mean": low_memory_mean,
        "supplemental_allclose": allclose,
        "supplemental_relative_l2_passed": relative_l2 <= relative_l2_tolerance,
        "atol": float(atol),
        "rtol": float(rtol),
        "relative_l2_tolerance": float(relative_l2_tolerance),
    }


def compare_particle_artifacts(
    standard_report: dict[str, Any], low_memory_report: dict[str, Any], standard_dir: Path, low_memory_dir: Path, *,
    rmse_cells_maximum: float = MAX_POSITION_RMSE_CELLS,
) -> dict[str, Any]:
    """Stream canonical particle rows and compare final positions in cell units."""
    import numpy as np

    standard_artifacts = standard_report.get("particle_artifacts") or {}
    low_memory_artifacts = low_memory_report.get("particle_artifacts") or {}
    expected_format = "pmpp-authoritative-particles-v1"
    if standard_artifacts.get("format") != expected_format:
        raise RuntimeError("standard scientific gate is missing canonical particle artifacts")
    if low_memory_artifacts.get("format") != expected_format:
        raise RuntimeError("low-memory scientific gate is missing canonical particle artifacts")
    for field in ("dimensions", "cell_size", "valid_particles"):
        if standard_artifacts.get(field) != low_memory_artifacts.get(field):
            raise RuntimeError(f"scientific gate particle {field} metadata differ")
    dimensions = int(standard_artifacts["dimensions"])
    cell_size = float(standard_artifacts["cell_size"])
    if dimensions <= 0 or not math.isfinite(cell_size) or cell_size <= 0:
        raise RuntimeError("scientific gate particle coordinate metadata are invalid")
    standard_shards = sorted(standard_artifacts["shards"], key=lambda row: row["global_index"])
    low_memory_shards = sorted(low_memory_artifacts["shards"], key=lambda row: row["global_index"])
    if len(standard_shards) != len(low_memory_shards):
        raise RuntimeError("scientific gate particle shard counts differ")
    compared_particles = 0
    pmid_mismatch_rows = 0
    squared_position_error_cells = 0.0
    max_coordinate_error_cells = 0.0
    chunk_rows = 2**18
    for standard_row, low_memory_row in zip(standard_shards, low_memory_shards):
        if standard_row["global_index"] != low_memory_row["global_index"]:
            raise RuntimeError("scientific gate particle shard global indices differ")
        if int(standard_row["valid_count"]) != int(low_memory_row["valid_count"]):
            raise RuntimeError("scientific gate particle shard valid counts differ")
        standard_pmid_path = standard_dir / standard_row["pmid_path"]
        low_memory_pmid_path = low_memory_dir / low_memory_row["pmid_path"]
        standard_disp_path = standard_dir / standard_row["disp_path"]
        low_memory_disp_path = low_memory_dir / low_memory_row["disp_path"]
        for path, checksum in ((standard_pmid_path,
                                standard_row["pmid_sha256"]), (low_memory_pmid_path, low_memory_row["pmid_sha256"]),
                               (standard_disp_path,
                                standard_row["disp_sha256"]), (low_memory_disp_path, low_memory_row["disp_sha256"]),
                               ):
            if not path.is_file() or _sha256(path) != checksum:
                raise RuntimeError(f"scientific gate particle artifact checksum failed: {path}")
        standard_pmid = np.load(standard_pmid_path, mmap_mode="r", allow_pickle=False)
        low_memory_pmid = np.load(low_memory_pmid_path, mmap_mode="r", allow_pickle=False)
        standard_disp = np.load(standard_disp_path, mmap_mode="r", allow_pickle=False)
        low_memory_disp = np.load(low_memory_disp_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(standard_row["valid_count"]), dimensions)
        if any(
            array.shape != expected_shape for array in (standard_pmid, low_memory_pmid, standard_disp, low_memory_disp)
        ):
            raise RuntimeError("scientific gate particle artifact shapes differ")
        for start in range(0, expected_shape[0], chunk_rows):
            stop = min(start + chunk_rows, expected_shape[0])
            standard_pmid_chunk = np.asarray(standard_pmid[start:stop])
            low_memory_pmid_chunk = np.asarray(low_memory_pmid[start:stop])
            pmid_mismatch_rows += int(np.count_nonzero(np.any(standard_pmid_chunk != low_memory_pmid_chunk, axis=1)))
            difference_cells = (
                np.asarray(low_memory_disp[start:stop], dtype=np.float64) -
                np.asarray(standard_disp[start:stop], dtype=np.float64)
            ) / cell_size
            squared_position_error_cells += float(np.dot(difference_cells.reshape(-1), difference_cells.reshape(-1)))
            max_coordinate_error_cells = max(
                max_coordinate_error_cells, float(np.max(np.abs(difference_cells), initial=0)),
            )
            compared_particles += stop - start
    denominator = max(compared_particles * dimensions, 1)
    position_rmse_cells = math.sqrt(squared_position_error_cells / denominator)
    passed = pmid_mismatch_rows == 0 and position_rmse_cells < rmse_cells_maximum
    return {
        "passed": passed,
        "compared_particles": compared_particles,
        "dimensions": dimensions,
        "pmid_exact": pmid_mismatch_rows == 0,
        "pmid_mismatch_rows": pmid_mismatch_rows,
        "position_rmse_cells": position_rmse_cells,
        "position_rmse_definition": "sqrt(sum((disp_low-disp_standard)^2)/(particles*dimensions))/cell_size",
        "max_coordinate_error_cells": max_coordinate_error_cells,
        "position_rmse_cells_maximum": float(rmse_cells_maximum),
    }


def compare_power_spectrum_artifacts(
    standard_report: dict[str, Any], low_memory_report: dict[str, Any], standard_dir: Path, low_memory_dir: Path, *,
    relative_error_maximum: float = MAX_SHELL_POWER_RELATIVE_ERROR,
) -> dict[str, Any]:
    """Compare worker-reduced shell power arrays without loading density cubes."""
    import numpy as np

    standard_artifact = standard_report.get("power_spectrum_artifact") or {}
    low_memory_artifact = low_memory_report.get("power_spectrum_artifact") or {}
    expected_format = "pmpp-shell-power-v1"
    if standard_artifact.get("format") != expected_format or low_memory_artifact.get("format") != expected_format:
        raise RuntimeError("scientific gate is missing shell power artifacts")
    if standard_artifact.get("mass_assignment") != low_memory_artifact.get("mass_assignment"):
        raise RuntimeError("scientific gate shell power mass-assignment metadata differ")

    def load_arrays(artifact, base_dir):
        result = {}
        for name in ("k", "power", "nmodes"):
            row = artifact["arrays"][name]
            path = base_dir / row["path"]
            if not path.is_file() or _sha256(path) != row["sha256"]:
                raise RuntimeError(f"scientific gate shell power checksum failed: {path}")
            result[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        return result

    standard = load_arrays(standard_artifact, standard_dir)
    low_memory = load_arrays(low_memory_artifact, low_memory_dir)
    if any(standard[name].shape != low_memory[name].shape for name in standard):
        raise RuntimeError("scientific gate shell power array shapes differ")
    if not np.array_equal(standard["nmodes"], low_memory["nmodes"]):
        raise RuntimeError("scientific gate shell mode counts differ")
    if not np.allclose(standard["k"], low_memory["k"], rtol=1e-7, atol=0):
        raise RuntimeError("scientific gate shell centers differ")
    reference = np.asarray(standard["power"], dtype=np.float64)
    candidate = np.asarray(low_memory["power"], dtype=np.float64)
    modes = np.asarray(standard["nmodes"], dtype=np.int64)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    populated = modes > 0
    reference_nonzero = np.abs(reference) > np.finfo(np.float32).tiny
    comparable = populated & finite & reference_nonzero
    if not np.any(comparable):
        raise RuntimeError("scientific gate shell power has no comparable populated shell")
    relative = np.abs(candidate[comparable] - reference[comparable]) / np.abs(reference[comparable])
    max_relative = float(np.max(relative))
    zero_reference_mismatch = bool(
        np.any(populated & finite & ~reference_nonzero & (np.abs(candidate) > np.finfo(np.float32).tiny))
    )
    nonfinite_populated = int(np.count_nonzero(populated & ~finite))
    passed = (nonfinite_populated == 0 and not zero_reference_mismatch and max_relative < relative_error_maximum)
    return {
        "passed": passed,
        "total_shells": int(reference.size),
        "populated_shells": int(np.count_nonzero(populated)),
        "compared_shells": int(np.count_nonzero(comparable)),
        "nonfinite_populated_shells": nonfinite_populated,
        "zero_reference_mismatch": zero_reference_mismatch,
        "max_shell_power_relative_error": max_relative,
        "max_shell_power_relative_error_maximum": float(relative_error_maximum),
    }


def compare_scientific_artifacts(
    standard_report: dict[str, Any], low_memory_report: dict[str, Any], standard_dir: Path, low_memory_dir: Path, *,
    position_rmse_cells_maximum: float = MAX_POSITION_RMSE_CELLS,
    density_cross_correlation_minimum: float = MIN_DENSITY_CROSS_CORRELATION,
    shell_power_relative_error_maximum: float = MAX_SHELL_POWER_RELATIVE_ERROR, atol: float = 5e-4, rtol: float = 5e-4,
    relative_l2_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Enforce all three full-trajectory standard-vs-low scientific gates."""
    particles = compare_particle_artifacts(
        standard_report, low_memory_report, standard_dir, low_memory_dir,
        rmse_cells_maximum=position_rmse_cells_maximum,
    )
    density = compare_density_artifacts(
        standard_report, low_memory_report, standard_dir, low_memory_dir, atol=atol, rtol=rtol,
        relative_l2_tolerance=relative_l2_tolerance, correlation_minimum=density_cross_correlation_minimum,
    )
    power = compare_power_spectrum_artifacts(
        standard_report, low_memory_report, standard_dir, low_memory_dir,
        relative_error_maximum=shell_power_relative_error_maximum,
    )
    return {
        "passed": particles["passed"] and density["passed"] and power["passed"],
        "particles": particles,
        "density": density,
        "power_spectrum": power,
    }


def run_scientific_gates(
    args: argparse.Namespace, selected: list[GPUInfo], allocator: str, budget_gib: float, output_root: Path,
) -> list[dict[str, Any]]:
    """Run fresh-process, full-trajectory standard-vs-low scientific gates."""
    resolutions = [resolution for resolution in args.scientific_resolutions if resolution <= args.case.resolution]
    if not resolutions:
        raise RuntimeError("no scientific-gate resolution is compatible with the selected case")
    production_target = args.case == CaseSpec(TARGET_RESOLUTION, 8) and not args.allow_non_h200
    missing_required = REQUIRED_PRODUCTION_SCIENTIFIC_RESOLUTIONS.difference(resolutions)
    if production_target and missing_required:
        raise RuntimeError(
            "production scientific evidence requires resolutions 64, 128, 256, and 512; "
            f"missing {sorted(missing_required)}"
        )
    reports = []
    for resolution in resolutions:
        requested_devices = 2 if production_target and resolution in (64, 128) else int(args.case.devices)
        device_count = min(requested_devices, len(selected))
        if resolution % device_count:
            raise RuntimeError(
                f"scientific-gate resolution {resolution} is not divisible by its {device_count}-GPU assignment"
            )
        case = CaseSpec(resolution, device_count)
        particle_capacity(case, args.max_ptcl_factor)
        case_dir = output_root / "scientific_gates" / case.label
        standard_dir = case_dir / "standard"
        low_memory_dir = case_dir / "low_memory"
        standard = launch_case(
            args, case, allocator, selected, budget_gib, standard_dir, save_density=True, profile="standard",
            noise_mode="nested", memory_gate_override_gib=budget_gib, save_scientific_artifacts=True,
        )
        low_memory = launch_case(
            args, case, allocator, selected, budget_gib, low_memory_dir, save_density=True, profile="low_memory",
            noise_mode="nested", memory_gate_override_gib=budget_gib, save_scientific_artifacts=True,
        )
        step_counts = (int(standard["physics"]["nbody_steps"]), int(low_memory["physics"]["nbody_steps"]))
        if step_counts != (63, 63):
            raise RuntimeError(
                f"scientific gate {case.label} must compare full 63-step trajectories, got {step_counts}"
            )
        comparison = compare_scientific_artifacts(
            standard, low_memory, standard_dir, low_memory_dir,
            position_rmse_cells_maximum=args.scientific_position_rmse_cells,
            density_cross_correlation_minimum=args.scientific_density_cross_correlation,
            shell_power_relative_error_maximum=args.scientific_shell_power_relative_error, atol=args.scientific_atol,
            rtol=args.scientific_rtol, relative_l2_tolerance=args.scientific_relative_l2,
        )
        gate_report = {
            "case": dataclasses.asdict(case),
            "required_production_evidence": resolution in REQUIRED_PRODUCTION_SCIENTIFIC_RESOLUTIONS,
            "nbody_steps": 63,
            "standard": {
                "worker_report": str((standard_dir / "worker_report.json").relative_to(output_root)),
                "elapsed_seconds": standard["elapsed_seconds"],
                "resident_memory": standard["resident_memory"],
            },
            "low_memory": {
                "worker_report": str((low_memory_dir / "worker_report.json").relative_to(output_root)),
                "elapsed_seconds": low_memory["elapsed_seconds"],
                "resident_memory": low_memory["resident_memory"],
            },
            "comparison": comparison,
        }
        _atomic_json(case_dir / "scientific_gate.json", gate_report)
        if not comparison["passed"]:
            raise RuntimeError(
                f"Scientific gate {case.label} failed: "
                f"position_rmse={comparison['particles']['position_rmse_cells']:.3e} cells, "
                f"density_correlation={comparison['density']['density_cross_correlation']:.9f}, "
                f"shell_relative_error={comparison['power_spectrum']['max_shell_power_relative_error']:.3e}"
            )
        reports.append(gate_report)
    _atomic_json(output_root / "scientific_gates_summary.json", {"status": "passed", "gates": reports, })
    return reports


def _write_completion_manifest(
    output_dir: Path, report: dict[str, Any], preflight: dict[str, Any], budget_gib: float,
    qualification_reports: list[dict[str, Any]], capacity_calibration: dict[str, Any],
    probe_reports: list[dict[str, Any]], scientific_gate_reports: list[dict[str, Any]],
) -> None:
    artifacts = report.get("artifacts") or {}
    if artifacts.get("format") != "pmpp-density-shards-v1":
        raise RuntimeError("Refusing to write completion manifest without density shards")
    expected = int(report["expected_particles"])
    if expected != TARGET_RESOLUTION**3 or int(report.get("valid_particles", -1)) != expected:
        raise RuntimeError("Production completion requires exactly 8,589,934,592 valid particles")
    if report.get("case") != {"resolution": TARGET_RESOLUTION, "devices": 8}:
        raise RuntimeError("Completion manifest is restricted to the exact 2048:8 production case")
    if not report.get("memory_gate_passed") or float(report["resident_memory"]["max_used_gib"]) > budget_gib:
        raise RuntimeError("Production resident-memory evidence exceeded the per-device budget")
    routing = report.get("routing") or {}
    extension = routing.get("extension") or {}
    if (
        not routing.get("active") or routing.get("backend") != "bidir_mergepath"
        or int(extension.get("record_format_version", -1)) != 3 or not extension.get("fused_primal_feature")
        or not extension.get("fused_primal_registered")
    ):
        raise RuntimeError("Production completion requires the active fused ABI-v3 bidirectional route")
    if int(routing.get("invalid_count", -1)) != 0:
        raise RuntimeError("Production completion requires zero invalid routed particles")
    migration = _low_memory_high_water(report, "migration_high_water")
    gather = _low_memory_high_water(report, "gather_high_water")
    if gather != 0:
        raise RuntimeError("Production mesh-halo completion requires a zero particle-gather high-water")
    capacity = report.get("capacity") or {}
    occupancy_ratio = float(capacity.get("occupancy_high_water_ratio", math.inf))
    configured_factor = float(capacity.get("max_ptcl_factor", 0))
    if recommended_particle_factor((occupancy_ratio, )) > configured_factor + 1e-12:
        raise RuntimeError("Final particle occupancy no longer has the required 0.02 capacity margin")
    if scaled_communication_capacity(migration, TARGET_RESOLUTION) > int(capacity.get("max_share_ptcl", 0)):
        raise RuntimeError("Final migration high-water no longer has the required 25% capacity margin")
    compilation = report.get("compilation") or {}
    if not compilation.get("passed") or not all((compilation.get("checks") or {}).values()):
        raise RuntimeError("Production completion requires passing N-body alias and scalar-field liveness evidence")
    if {row.get("probe") for row in probe_reports if row.get("status") == "passed"} != set(REQUIRED_PROBES):
        raise RuntimeError("Production completion requires every exact-shape probe to pass")
    scientific_resolutions = {
        int(row["case"]["resolution"])
        for row in scientific_gate_reports if (row.get("comparison") or {}).get("passed")
    }
    if not REQUIRED_PRODUCTION_SCIENTIFIC_RESOLUTIONS.issubset(scientific_resolutions):
        raise RuntimeError("Production completion requires every standard-vs-low scientific gate to pass")
    if not qualification_reports or any(not row.get("memory_gate_passed") for row in qualification_reports):
        raise RuntimeError("Production completion requires a memory-qualified intermediate ladder")
    mass = float(artifacts["density_sum_float64"])
    relative_mass_error = abs(mass - expected) / expected
    if relative_mass_error > 1e-6:
        raise RuntimeError(f"Density mass error {relative_mass_error:.3e} exceeds 1e-6")
    if not artifacts.get("density_finite", False):
        raise RuntimeError("Density contains non-finite values")
    if not report.get("particle_state_finite", False):
        raise RuntimeError("Final particle state contains non-finite values")
    shards = artifacts.get("shards") or []
    if len(shards) != 8:
        raise RuntimeError(f"Expected exactly eight density slabs, found {len(shards)}")
    for shard in shards:
        path = output_dir / shard["path"]
        if not path.is_file() or _sha256(path) != shard["sha256"]:
            raise RuntimeError(f"Density shard checksum failed: {path}")
    payload = {
        "schema": "pmpp-h200-forward-v1",
        "completed": True,
        "memory_budget_gib": budget_gib,
        "relative_mass_error": relative_mass_error,
        "preflight": preflight,
        "qualification_reports": qualification_reports,
        "probe_reports": probe_reports,
        "scientific_gate_reports": scientific_gate_reports,
        "capacity_calibration": capacity_calibration,
        **report,
    }
    _atomic_json(output_dir / "manifest.json", payload)


def supervisor_main(args: argparse.Namespace) -> int:
    """Run preflight, allocator qualification, and the gated resolution ladder."""
    args._qualification_stage = "input_validation"
    communication_capacities = (args.max_share_ptcl, args.max_halo_share_ptcl, args.max_share_gather_ptcl)
    if min(communication_capacities) <= 0 or max(communication_capacities) > INT32_MAX:
        raise ValueError("all communication capacities must be explicit positive integers")
    if not math.isfinite(args.scientific_position_rmse_cells) or args.scientific_position_rmse_cells <= 0:
        raise ValueError("scientific position RMSE threshold must be finite and positive")
    if (
        not math.isfinite(args.scientific_density_cross_correlation)
        or not -1 <= args.scientific_density_cross_correlation <= 1
    ):
        raise ValueError("scientific density cross-correlation threshold must be finite and in [-1, 1]")
    if not math.isfinite(args.scientific_shell_power_relative_error) or args.scientific_shell_power_relative_error <= 0:
        raise ValueError("scientific shell power relative-error threshold must be finite and positive")
    if args.case == CaseSpec(TARGET_RESOLUTION, 8) and not args.qualification_ladder and not args.allow_non_h200:
        raise ValueError("The production 2048:8 case requires the full qualification ladder")
    production_target = args.case == CaseSpec(TARGET_RESOLUTION, 8) and not args.allow_non_h200
    if production_target and args.allocator != "auto":
        raise ValueError("The production 2048:8 case requires allocator='auto' qualification")
    if production_target and not args.qualification_probes:
        raise ValueError("The production 2048:8 case requires all exact-shape qualification probes")
    if production_target and not args.scientific_gates:
        raise ValueError("The production 2048:8 case requires standard-vs-low scientific gates")
    if production_target and not REQUIRED_PRODUCTION_SCIENTIFIC_RESOLUTIONS.issubset(args.scientific_resolutions):
        raise ValueError("The production 2048:8 case requires scientific evidence at 64, 128, 256, and 512")
    if production_target and (
        args.scientific_position_rmse_cells > MAX_POSITION_RMSE_CELLS or args.scientific_density_cross_correlation
        < MIN_DENSITY_CROSS_CORRELATION or args.scientific_shell_power_relative_error > MAX_SHELL_POWER_RELATIVE_ERROR
    ):
        raise ValueError("Production scientific acceptance thresholds cannot be relaxed")
    output_root = args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Refusing to reuse nonempty qualification output directory: {output_root}")
    args._qualification_stage = "node_preflight"
    all_devices = query_gpu_info()
    selected = _selected_gpu_info(all_devices, args.devices)
    preflight = validate_h200_node(
        selected, require_eight=not args.allow_non_h200, allow_non_h200=args.allow_non_h200, output_dir=output_root,
    )
    min_total = min(device.memory_total_mib for device in selected)
    budget_gib = memory_budget_gib(min_total)
    if budget_gib <= 0:
        raise RuntimeError("Reported GPU memory leaves no positive production budget")
    _atomic_json(output_root / "preflight.json", {**preflight, "memory_budget_gib": budget_gib})
    target_case = args.case
    intermediate_cases = [CaseSpec(1024, 4), CaseSpec(1536, 8), CaseSpec(1792, 8)] if args.qualification_ladder else []
    intermediate_cases = [case for case in intermediate_cases if case != target_case]
    for case in [*intermediate_cases, target_case]:
        particle_capacity(case, args.max_ptcl_factor)
    selected_allocator = args.allocator
    allocator_candidates = []
    if selected_allocator == "auto":
        args._qualification_stage = "allocator_qualification"
        qualifier = CaseSpec(1024, 4)
        candidates = []
        failures = []
        for allocator in ("async", "bfc"):
            for repetition in range(2):
                path = output_root / "allocator_qualification" / f"{allocator}_rep{repetition}"
                try:
                    candidates.append(
                        launch_case(args, qualifier, allocator, selected, budget_gib, path, save_density=False)
                    )
                except RuntimeError as error:
                    failures.append({"allocator": allocator, "repetition": repetition, "error": str(error), })
        by_allocator = {}
        for allocator in ("async", "bfc"):
            rows = [row for row in candidates if row["allocator"] == allocator]
            if len(rows) != 2:
                continue
            elapsed = sorted(row["elapsed_seconds"] for row in rows)
            by_allocator[allocator] = {
                "peak": max(row["resident_memory"]["max_used_gib"] for row in rows),
                "median_elapsed": (elapsed[0] + elapsed[1]) / 2,
            }
        if not by_allocator:
            _atomic_json(
                output_root / "allocator_selection.json", {
                    "selected": None,
                    "candidates": {},
                    "runs": candidates,
                    "failures": failures,
                },
            )
            raise RuntimeError("Neither allocator completed two fresh-process qualification repeats")
        selected_allocator = min(
            by_allocator, key=lambda name: (by_allocator[name]["peak"], by_allocator[name]["median_elapsed"])
        )
        _atomic_json(
            output_root / "allocator_selection.json", {
                "selected": selected_allocator,
                "candidates": by_allocator,
                "runs": candidates,
                "failures": failures,
            },
        )
        allocator_candidates = candidates
    base_particle_factor = float(args.max_ptcl_factor)
    base_communication_capacities = {
        "max_share_ptcl": int(args.max_share_ptcl),
        "max_halo_share_ptcl": int(args.max_halo_share_ptcl),
        "max_share_gather_ptcl": int(args.max_share_gather_ptcl),
    }
    observed_occupancy_ratios = [float(row["capacity"]["occupancy_high_water_ratio"]) for row in allocator_candidates]
    args.max_ptcl_factor = max(base_particle_factor, recommended_particle_factor(observed_occupancy_ratios))
    ladder_reports = []
    calibration_source = None
    args._qualification_stage = "intermediate_ladder"
    for case in intermediate_cases:
        for attempt in range(3):
            particle_capacity(case, args.max_ptcl_factor)
            suffix = "" if attempt == 0 else f"_particle_capacity_recheck{attempt}"
            path = output_root / f"{case.label}{suffix}"
            report = launch_case(args, case, selected_allocator, selected, budget_gib, path, save_density=False, )
            if attempt:
                report["particle_capacity_recheck"] = True
            _low_memory_high_water(report, "migration_high_water")
            _low_memory_high_water(report, "gather_high_water")
            if _low_memory_high_water(report, "routing_invalid_count"):
                raise RuntimeError(f"{case.label} returned invalid routed particles")
            ladder_reports.append(report)
            observed_occupancy_ratios.append(float(report["capacity"]["occupancy_high_water_ratio"]))
            required_factor = max(base_particle_factor, recommended_particle_factor(observed_occupancy_ratios))
            if required_factor <= args.max_ptcl_factor + 1e-12:
                break
            args.max_ptcl_factor = required_factor
        else:
            raise RuntimeError(f"{case.label} particle-capacity calibration did not converge")
        calibration_source = (case, report)

    capacity_calibration: dict[str, Any] = {
        "base_factor": base_particle_factor,
        "production_factor": args.max_ptcl_factor,
        "observed_occupancy_ratios": observed_occupancy_ratios,
        "base_communication_capacities": base_communication_capacities,
    }
    if production_target:
        args._qualification_stage = "capacity_calibration"
        if calibration_source is None or calibration_source[0] != CaseSpec(1792, 8):
            raise RuntimeError("Production communication calibration requires a successful 1792:8 trajectory")
        source_case, source_report = calibration_source
        derived = calibrated_communication_capacities(source_report, source_case.resolution)
        target_particle_capacity = particle_capacity(target_case, args.max_ptcl_factor)
        if derived["max_share_ptcl"] > target_particle_capacity // 2:
            raise RuntimeError("Derived migration capacity exceeds the supported half-particle-capacity envelope")
        if derived["max_halo_share_ptcl"] > target_particle_capacity:
            raise RuntimeError("Derived halo capacity exceeds the target particle capacity")
        if derived["max_share_gather_ptcl"] > target_particle_capacity // 2:
            raise RuntimeError("Derived gather capacity exceeds the supported half-particle-capacity envelope")
        args.max_share_ptcl = derived["max_share_ptcl"]
        args.max_halo_share_ptcl = derived["max_halo_share_ptcl"]
        args.max_share_gather_ptcl = derived["max_share_gather_ptcl"]

        args._qualification_stage = "capacity_envelope_recheck"
        envelope_path = output_root / f"{source_case.label}_production_capacity_envelope"
        envelope_report = launch_case(
            args, source_case, selected_allocator, selected, budget_gib, envelope_path, save_density=False,
        )
        envelope_report["production_capacity_envelope_recheck"] = True
        ladder_reports.append(envelope_report)
        observed_occupancy_ratios.append(float(envelope_report["capacity"]["occupancy_high_water_ratio"]))
        if recommended_particle_factor(observed_occupancy_ratios) > args.max_ptcl_factor + 1e-12:
            raise RuntimeError("Production particle capacity changed during the final intermediate recheck")
        rederived = calibrated_communication_capacities(envelope_report, source_case.resolution)
        for name in ("max_share_ptcl", "max_halo_share_ptcl", "max_share_gather_ptcl"):
            if rederived[name] > int(getattr(args, name)):
                raise RuntimeError(f"Production {name} changed during the final intermediate recheck")
            if int(envelope_report["capacity"][name]) != int(getattr(args, name)):
                raise RuntimeError(f"Runtime clamped the qualified production {name}")
        capacity_calibration.update(
            production_factor=args.max_ptcl_factor, observed_occupancy_ratios=observed_occupancy_ratios,
            communication_source=dataclasses.asdict(source_case), communication=derived,
            envelope_recheck_worker_report=str((envelope_path / "worker_report.json").relative_to(output_root)),
            mesh_halo_gather_note="fixed-width mesh-edge exchange; dynamic particle gather high-water is zero",
        )

    # Exact-shape probes intentionally run only after all production capacities
    # are frozen and memory-qualified by the largest intermediate trajectory.
    probe_reports = []
    if args.qualification_probes:
        args._qualification_stage = "exact_shape_probes"
        probe_reports = run_required_probes(args, target_case, selected, selected_allocator, budget_gib, output_root, )
    scientific_gate_reports = []
    if args.scientific_gates:
        args._qualification_stage = "scientific_gates"
        scientific_gate_reports = run_scientific_gates(args, selected, selected_allocator, budget_gib, output_root, )

    particle_capacity(target_case, args.max_ptcl_factor)
    final_case = target_case.resolution == TARGET_RESOLUTION and target_case.devices == 8
    final_path = output_root / target_case.label
    args._qualification_stage = "final_trajectory"
    final_report = launch_case(
        args, target_case, selected_allocator, selected, budget_gib, final_path, save_density=final_case,
    )
    ladder_reports.append(final_report)
    final_migration = _low_memory_high_water(final_report, "migration_high_water")
    final_gather = _low_memory_high_water(final_report, "gather_high_water")
    final_invalid = _low_memory_high_water(final_report, "routing_invalid_count")
    args._qualification_stage = "final_acceptance"
    if final_invalid:
        raise RuntimeError(f"Final trajectory reported {final_invalid} invalid routed particles")
    if production_target:
        final_ratio = float(final_report["capacity"]["occupancy_high_water_ratio"])
        if recommended_particle_factor((final_ratio, )) > args.max_ptcl_factor + 1e-12:
            raise RuntimeError("Final trajectory exceeded the telemetry-derived particle safety margin")
        if scaled_communication_capacity(final_migration, TARGET_RESOLUTION) > args.max_share_ptcl:
            raise RuntimeError("Final trajectory exceeded the telemetry-derived migration safety margin")
        if final_gather != 0:
            raise RuntimeError("Final mesh-halo trajectory unexpectedly used a dynamic particle gather buffer")
        capacity_calibration["final_observed"] = {
            "occupancy_high_water_ratio": final_ratio,
            "migration_high_water": final_migration,
            "gather_high_water": final_gather,
            "routing_invalid_count": final_invalid,
        }
    _atomic_json(
        output_root / "ladder_summary.json", {
            "status": "cases_completed",
            "selected_allocator": selected_allocator,
            "memory_budget_gib": budget_gib,
            "particle_factor": args.max_ptcl_factor,
            "observed_occupancy_ratios": observed_occupancy_ratios,
            "capacity_calibration": capacity_calibration,
            "probe_reports": probe_reports,
            "scientific_gate_reports": scientific_gate_reports,
            "cases": ladder_reports,
        },
    )
    if production_target:
        args._qualification_stage = "completion_manifest"
        _write_completion_manifest(
            final_path, final_report, preflight, budget_gib, ladder_reports, capacity_calibration, probe_reports,
            scientific_gate_reports,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-name", choices=REQUIRED_PROBES, help=argparse.SUPPRESS)
    parser.add_argument("--probe-route-migrants", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-resolution", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--worker-devices", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-profile", choices=("standard", "low_memory"), default="low_memory", help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-noise-mode", choices=("standard", "nested"), default="nested", help=argparse.SUPPRESS)
    parser.add_argument("--devices", type=parse_device_ids, default=parse_device_ids("0,1,2,3,4,5,6,7"))
    parser.add_argument("--case", type=parse_case, default=parse_case("2048:8"))
    parser.add_argument("--qualification-ladder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qualification-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scientific-gates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scientific-resolutions", type=parse_resolutions, default=DEFAULT_SCIENTIFIC_RESOLUTIONS, )
    parser.add_argument("--scientific-atol", type=float, default=5e-4)
    parser.add_argument("--scientific-rtol", type=float, default=5e-4)
    parser.add_argument("--scientific-relative-l2", type=float, default=1e-4)
    parser.add_argument("--scientific-position-rmse-cells", type=float, default=MAX_POSITION_RMSE_CELLS)
    parser.add_argument("--scientific-density-cross-correlation", type=float, default=MIN_DENSITY_CROSS_CORRELATION, )
    parser.add_argument("--scientific-shell-power-relative-error", type=float, default=MAX_SHELL_POWER_RELATIVE_ERROR, )
    parser.add_argument("--allocator", choices=("auto", "async", "bfc"), default="auto")
    parser.add_argument("--allow-non-h200", action="store_true", help="Development-only relaxation of hardware checks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-density", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-scientific-artifacts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-compilation-artifacts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--box-size", type=float, default=1000.0)
    parser.add_argument("--omega-m", type=float, default=0.3883118322778484)
    parser.add_argument("--sigma8", type=float, default=0.8717719592215194)
    parser.add_argument("--n-s", type=float, default=0.9652)
    parser.add_argument("--omega-b", type=float, default=0.02233)
    parser.add_argument("--h", type=float, default=0.6737)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.05)
    parser.add_argument("--max-share-ptcl", type=int, default=1_750_000)
    parser.add_argument("--max-halo-share-ptcl", type=int, default=1_750_000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=6_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.probe_worker:
        return probe_worker_main(args)
    if args.worker:
        return worker_main(args)
    output_root = args.output_dir.resolve()
    preexisting_nonempty = output_root.exists() and any(output_root.iterdir())
    try:
        return supervisor_main(args)
    except Exception as error:
        # A reused directory may contain a prior successful manifest. Preserve
        # it byte-for-byte while refusing the new run.
        if not preexisting_nonempty:
            stage = getattr(args, "_qualification_stage", "supervisor_start")
            _write_qualification_failure(output_root, stage, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
