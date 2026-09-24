#!/usr/bin/env python3
"""Run a full, forward PM++ simulation in a 5 Mpc/h box on four H100s.

The default is 256^3 particles, 2LPT, 63 N-body steps from a=0.1 to 1,
hybrid mesh-halo routing, and the low-memory forward solver. The full density,
three axis projections, and a projection PNG are saved after the simulation. Capacity or
mass-conservation failures leave a JSON report with status="failed".
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="Directory for the density-projection PNG; defaults to the JSON directory")
    parser.add_argument("--npart", type=int, default=256)
    parser.add_argument("--box-size", type=float, default=5.0,
                        help="Comoving box length in Mpc/h")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--platform", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--policy", choices=("hybrid", "neighbor_only"), default="hybrid")
    parser.add_argument("--require-gpu-model", default="H100")
    parser.add_argument("--native-routing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pallas-cic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--a-start", type=float, default=0.1)
    parser.add_argument("--a-stop", type=float, default=1.0)
    parser.add_argument("--nbody-steps", type=int, default=63)
    parser.add_argument("--execution-runs", type=int, default=1,
                        help="Run twice to record a compile-free full-forward execution time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma8", type=float, default=0.80)
    parser.add_argument("--n-s", type=float, default=0.96)
    parser.add_argument("--omega-m", type=float, default=0.30)
    parser.add_argument("--omega-b", type=float, default=0.05)
    parser.add_argument("--h", type=float, default=0.70)
    parser.add_argument("--lpt-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.5)
    parser.add_argument("--max-ptcl-per-slice", type=int, default=None,
                        help="Explicit particle slots per GPU; overrides --max-ptcl-factor")
    parser.add_argument("--max-share-ptcl", type=int, default=2_000_000)
    parser.add_argument("--lpt-share-multiplier", type=float, default=1.5)
    parser.add_argument("--max-halo-share-ptcl", type=int, default=1_000_000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=1_000_000)
    parser.add_argument("--far-send-capacity", type=int, default=1_000_000)
    parser.add_argument("--far-recv-capacity", type=int, default=1_000_000)
    parser.add_argument("--far-chunk-size", type=int, default=16_384)
    parser.add_argument("--mass-relative-tolerance", type=float, default=2e-6)
    args = parser.parse_args()
    if args.npart <= 0 or args.npart > 32767 or args.npart % args.devices:
        parser.error("npart must be positive, fit int16, and be divisible by devices")
    if args.devices < 1 or args.box_size <= 0 or args.max_ptcl_factor < 1:
        parser.error("devices and box-size must be positive; max-ptcl-factor must be >= 1")
    if args.max_ptcl_per_slice is not None and args.max_ptcl_per_slice < args.npart**3 // args.devices:
        parser.error("max-ptcl-per-slice must hold at least the initial particles per GPU")
    if args.max_ptcl_per_slice is not None and args.max_ptcl_per_slice > args.npart**3:
        parser.error("max-ptcl-per-slice exceeds the total particle count; check for an extra digit")
    if not (args.sigma8 > 0 and args.n_s > 0 and args.h > 0 and
            0 < args.omega_b < args.omega_m < 1):
        parser.error("require sigma8, n-s, h > 0 and 0 < omega-b < omega-m < 1")
    if args.mesh_shape < 1:
        parser.error("mesh-shape must be positive")
    if not 0 < args.a_start < args.a_stop or args.nbody_steps < 1 or args.execution_runs < 1:
        parser.error("require 0 < a-start < a-stop and positive nbody-steps and execution-runs")
    if args.max_share_ptcl < 1 or args.max_halo_share_ptcl < 1 or args.max_share_gather_ptcl < 1:
        parser.error("share capacities must be positive")
    if args.lpt_share_multiplier < 1 or args.mass_relative_tolerance <= 0:
        parser.error("lpt-share-multiplier must be >= 1 and mass tolerance positive")
    if args.policy == "hybrid":
        if args.platform != "gpu" or not args.native_routing or args.devices < 4:
            parser.error("hybrid requires at least four GPUs and native routing")
        if min(args.far_send_capacity, args.far_recv_capacity, args.far_chunk_size) < 1:
            parser.error("hybrid far capacities and chunk size must be positive")
        if args.far_chunk_size > args.far_send_capacity:
            parser.error("far-chunk-size cannot exceed far-send-capacity")
    return args


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def device_memory(devices):
    result = {}
    for device in devices:
        try:
            stats = device.memory_stats() or {}
        except (AttributeError, RuntimeError):
            stats = {}
        result[str(device)] = {name: int(stats[name]) for name in
                               ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
                               if name in stats}
    return result


def block_tree(jax, value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def plot_density_projections(projections, box_size, output):
    """Render the three mean-density projections with one logarithmic scale."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    axes = (("x", "y", "z"), ("y", "x", "z"), ("z", "x", "y"))
    display = {
        axis: np.log10(1.0 + np.maximum(np.asarray(projections[axis]), 0.0))
        for axis, _, _ in axes
    }
    upper = max(0.1, float(np.percentile(np.concatenate(
        [values.ravel() for values in display.values()]), 99.9)))
    fig, panels = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    try:
        for panel, (axis, horizontal, vertical) in zip(panels, axes):
            image = panel.imshow(
                display[axis].T, origin="lower", extent=(0, box_size, 0, box_size),
                cmap="inferno", vmin=0, vmax=upper, interpolation="nearest",
            )
            panel.set_title(f"{axis}-axis projection")
            panel.set_xlabel(f"{horizontal} [Mpc/h]")
            panel.set_ylabel(f"{vertical} [Mpc/h]")
            panel.set_aspect("equal")
        colorbar = fig.colorbar(image, ax=panels, shrink=0.82)
        colorbar.set_label("log10(1 + density / mean density)")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        fig.savefig(temporary, format="png", dpi=170)
        temporary.replace(output)
    finally:
        plt.close(fig)


def run(args, report):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path.insert(0, str(ROOT))

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    import src.pmpp as pmpp
    from src.pmpp.cic import scatter
    from src.pmpp.cic.pallas import pallas_cic_supported
    from src.pmpp.core.configuration import Configuration
    from src.pmpp.core.utils import create_compute_mesh
    from src.pmpp.cosmology import Cosmology, boltzmann
    from src.pmpp.distributed.configuration import MultiGPUConfiguration
    from src.pmpp.distributed.cuda import enabled_for_configuration, extension_status
    from src.pmpp.initial_conditions import linear_modes, white_noise
    from src.pmpp.initial_conditions.lpt import lpt_low_memory_with_telemetry
    from src.pmpp.nbody import nbody
    from src.pmpp.nbody.solver import lower_nbody_low_memory, nbody_low_memory_with_telemetry

    if Path(pmpp.__file__).resolve().parent != (ROOT / "src" / "pmpp").resolve():
        raise RuntimeError("PM++ was not imported from this checkout")
    devices = list(jax.devices(args.platform))
    if len(devices) != args.devices:
        raise RuntimeError(f"expected exactly {args.devices} visible {args.platform} devices, got {devices}")
    if args.platform == "gpu" and args.require_gpu_model:
        kinds = [getattr(device, "device_kind", "") for device in devices]
        if any(args.require_gpu_model.lower() not in kind.lower() for kind in kinds):
            raise RuntimeError(f"expected {args.require_gpu_model} GPUs, found {kinds}")
    if args.policy == "hybrid" and not os.environ.get("PMPP_CUDA_ROUTING_LIBRARY"):
        raise RuntimeError("set PMPP_CUDA_ROUTING_LIBRARY to the qualified native library")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    mesh = create_compute_mesh(devices)
    local_particles = args.npart**3 // args.devices
    capacity = args.max_ptcl_per_slice or math.ceil(local_particles * args.max_ptcl_factor)
    if capacity > args.npart**3:
        raise ValueError("particle capacity per GPU exceeds the total particle count")
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64
    with enable_x64():
        conf = Configuration(
            args.box_size / args.npart, (args.npart,) * 3,
            mesh_shape=args.mesh_shape, float_dtype=jnp.float32, cosmo_dtype=jnp.float64,
            lpt_order=args.lpt_order, lpt_cache_strains=False,
            a_start=args.a_start, a_stop=args.a_stop,
            a_nbody_maxstep=(args.a_stop - args.a_start) / args.nbody_steps,
            nbody_cosmo_grad=False, pallas_cic=args.pallas_cic,
            multigpu=MultiGPUConfiguration(
                compute_mesh=mesh, mode="mesh_halo",
                cuda_routing=args.native_routing,
                cuda_routing_backend="bidir_mergepath",
                migration_policy=args.policy,
                far_send_capacity=args.far_send_capacity if args.policy == "hybrid" else None,
                far_recv_capacity=args.far_recv_capacity if args.policy == "hybrid" else None,
                far_chunk_size=args.far_chunk_size if args.policy == "hybrid" else None,
            ),
            max_ptcl_per_slice=capacity,
            max_share_ptcl=args.max_share_ptcl,
            max_halo_share_ptcl=args.max_halo_share_ptcl,
            max_share_gather_ptcl=args.max_share_gather_ptcl,
        )
        conf_lpt = conf.replace(max_share_ptcl=math.ceil(
            args.max_share_ptcl * args.lpt_share_multiplier))
        native = enabled_for_configuration(conf)
        status = extension_status()
        if args.native_routing and not native:
            raise RuntimeError("requested native routing is inactive")
        if args.policy == "hybrid" and not status.get("hybrid_registered", False):
            raise RuntimeError("native hybrid FFI target is not registered")
        if args.pallas_cic and not pallas_cic_supported(conf.float_dtype):
            raise RuntimeError("requested Pallas CIC is unavailable")

        report["runtime"] = {
            "git_commit": commit,
            "python": sys.executable, "pmpp_origin": str(Path(pmpp.__file__).resolve()),
            "jax": jax.__version__, "jaxlib": metadata.version("jaxlib"),
            "devices": [str(d) for d in devices],
            "device_kinds": [getattr(d, "device_kind", "") for d in devices],
            "routing_library": os.environ.get("PMPP_CUDA_ROUTING_LIBRARY"),
            "routing_manifest": os.environ.get("PMPP_CUDA_ROUTING_MANIFEST"),
            "native_routing_active": bool(native),
            "hybrid_registered": bool(status.get("hybrid_registered", False)),
            "pallas_cic_active": bool(args.pallas_cic),
            "nbody_steps_actual": int(conf.a_nbody_num),
            "max_ptcl_per_slice_actual": int(conf.max_ptcl_per_slice),
            "max_share_ptcl_actual": int(conf.max_share_ptcl),
            "max_share_ptcl_lpt_actual": int(conf_lpt.max_share_ptcl),
            "max_halo_share_ptcl_actual": int(conf.max_halo_share_ptcl),
            "max_share_gather_ptcl_actual": int(conf.max_share_gather_ptcl),
            "particle_spacing_mpc_h": float(conf.ptcl_spacing),
            "cosmology": {"sigma8": args.sigma8, "n_s": args.n_s, "Omega_m": args.omega_m,
                          "Omega_b": args.omega_b, "h": args.h},
        }
        save_report(args.output, report)

        def stage(name, fn):
            report["phase"] = name
            save_report(args.output, report)
            print(f"[{name}] start", flush=True)
            started = time.perf_counter()
            value = block_tree(jax, fn())
            seconds = time.perf_counter() - started
            report["timings_seconds"][name] = seconds
            report["memory_after_phase"][name] = device_memory(devices)
            save_report(args.output, report)
            print(f"[{name}] {seconds:.3f} s", flush=True)
            return value

        cosmo = stage("cosmology", lambda: boltzmann(Cosmology.from_sigma8(
            conf, sigma8=args.sigma8, n_s=args.n_s, Omega_m=args.omega_m,
            Omega_b=args.omega_b, h=args.h), conf))
        seed = jax.device_put(jnp.asarray(args.seed, dtype=jnp.int32), NamedSharding(mesh, P()))
        compiled_nbody = None

        def forward(prefix):
            nonlocal compiled_nbody
            modes = stage(prefix + "white_noise", lambda: white_noise(seed, conf))
            modes = stage(prefix + "linear_modes", lambda: linear_modes(modes, cosmo, conf))
            particles, lpt_moved, lpt_invalid = stage(
                prefix + "lpt", lambda: lpt_low_memory_with_telemetry(modes, cosmo, conf_lpt))
            del modes
            report["telemetry"]["lpt_max_moved"] = int(jax.device_get(lpt_moved))
            report["telemetry"]["lpt_invalid"] = int(jax.device_get(lpt_invalid))
            if report["telemetry"]["lpt_invalid"]:
                raise RuntimeError("LPT reported invalid or overflowed particle routing")
            if args.native_routing:
                if compiled_nbody is None:
                    compiled_nbody = stage(
                        "nbody_compile", lambda: lower_nbody_low_memory(particles, cosmo, conf).compile())
                    report["nbody_timing_excludes_compilation"] = True
                    analysis = compiled_nbody.memory_analysis()
                    if analysis is not None:
                        report["nbody_compiled_memory_bytes"] = {
                            name: int(getattr(analysis, name))
                            for name in ("argument_size_in_bytes", "output_size_in_bytes",
                                         "temp_size_in_bytes", "alias_size_in_bytes")
                            if getattr(analysis, name, None) is not None
                        }
                    save_report(args.output, report)
                particles, occupancy, moved, invalid = stage(
                    prefix + "nbody", lambda: nbody_low_memory_with_telemetry(
                        particles, cosmo, conf, compiled=compiled_nbody))
                report["telemetry"].update({
                    "nbody_max_occupancy": int(jax.device_get(occupancy)),
                    "nbody_max_moved": int(jax.device_get(moved)),
                    "nbody_invalid": int(jax.device_get(invalid)),
                })
                if report["telemetry"]["nbody_invalid"]:
                    raise RuntimeError("N-body reported invalid or overflowed particle routing")
            else:
                # The portable path is used only for a local CPU smoke run.
                particles = stage(prefix + "nbody", lambda: nbody(particles, cosmo, conf))
                report["telemetry"].update({
                    "nbody_max_occupancy": None, "nbody_max_moved": None,
                    "nbody_invalid": None,
                })
            density = stage(prefix + "scatter", lambda: scatter(particles, conf))
            del particles
            return density

        for run_index in range(args.execution_runs):
            prefix = "" if run_index == args.execution_runs - 1 else f"warmup_{run_index + 1}_"
            density = forward(prefix)
            if prefix:
                del density
        forward_stages = ("white_noise", "linear_modes", "lpt", "nbody", "scatter")
        report["forward_execution_seconds"] = sum(report["timings_seconds"][name] for name in forward_stages)
        report["forward_execution_excludes_compilation"] = args.execution_runs > 1
        save_report(args.output, report)

    density_host = stage("density_to_host", lambda: np.asarray(jax.device_get(density), dtype=np.float32))
    if density_host.shape != (args.npart,) * 3:
        raise RuntimeError(f"wrong density shape: {density_host.shape}")
    finite = bool(np.isfinite(density_host).all())
    total = float(np.sum(density_host, dtype=np.float64))
    expected = float(args.npart**3)
    error = total - expected
    tolerance = max(1e-3, expected * args.mass_relative_tolerance)
    report["numerical"] = {
        "density_sum": total, "expected_mass": expected,
        "mass_error": error, "mass_tolerance": tolerance,
        "density_min": float(np.min(density_host)),
        "density_max": float(np.max(density_host)),
        "finite": finite,
    }
    if not finite or abs(error) > tolerance:
        raise RuntimeError("final density failed finite-value or mass-conservation check")

    density_path = args.output.with_name(args.output.stem + "_density.npy")
    projections = {
        axis: density_host.mean(axis=index, dtype=np.float64).astype(np.float32)
        for index, axis in enumerate("xyz")
    }
    projection_paths = {
        axis: args.output.with_name(args.output.stem + f"_projection_{axis}.npy")
        for axis in "xyz"
    }
    plot_path = (args.plot_dir or args.output.parent) / (args.output.stem + "_projections.png")
    np.save(density_path, density_host, allow_pickle=False)
    for axis in "xyz":
        np.save(projection_paths[axis], projections[axis], allow_pickle=False)
    report["artifacts"] = {"density": str(density_path.resolve()),
                           **{f"projection_{axis}": str(projection_paths[axis].resolve())
                              for axis in "xyz"}}
    save_report(args.output, report)
    stage("render_density_plot", lambda: plot_density_projections(projections, args.box_size, plot_path))
    report["artifacts"]["projections_png"] = str(plot_path.resolve())
    report["status"] = "ok"
    report["phase"] = "complete"
    save_report(args.output, report)
    print(f"[complete] density sum={total:.6f}; saved {args.output}", flush=True)


def main():
    args = arguments()
    report = {
        "status": "running", "phase": "setup", "host": socket.gethostname(),
        "timings_include_first_call_compilation": True,
        "timing_note": "N-body compilation is timed separately; other first-call stages may compile",
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "timings_seconds": {}, "memory_after_phase": {}, "telemetry": {},
    }
    started = time.perf_counter()
    try:
        run(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        save_report(args.output, report)
        raise
    finally:
        report["total_wall_seconds"] = time.perf_counter() - started
        save_report(args.output, report)


if __name__ == "__main__":
    main()
