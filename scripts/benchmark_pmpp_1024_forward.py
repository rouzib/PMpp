#!/usr/bin/env python
"""Benchmark the PM++ forward pass on 4 GPUs with a 1024^3 production-style setup.

The default configuration mirrors the active settings in ``scripts/pmpp_1024_maps.py``:

- ``box_size = 1000``
- ``num_ptcl = 1024``
- ``mesh_shape = 1``
- ``a_start = 1/64``
- ``a_nbody_maxstep = 1/64``
- ``max_ptcl_per_slice ~= 1.07 * num_ptcl^3 / num_devices``
- ``max_share_ptcl = 350000``
- ``max_share_gather_ptcl = 1200000``

By default the script benchmarks the staged forward path:

1. ``Cosmology.from_sigma8``
2. ``boltzmann``
3. ``white_noise``
4. ``linear_modes``
5. ``lpt``
6. ``nbody`` (jitted)
7. ``scatter`` (jitted)

It reports the first run (compile + execute) and subsequent steady-state runs.
``--mode whole`` is available if you want to benchmark a monolithic jitted forward
function instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from src.boltzmann import boltzmann
from src.configuration import Configuration
from src.cosmo import Cosmology
from src.lpt import lpt
from src.modes import linear_modes, white_noise
from src.nbody import nbody, nbody_kappa
from src.scatter import scatter
from src.utils import create_compute_mesh, get_a_schedule


TARGET_Z = jnp.array(
    [
        0.017,
        0.052,
        0.087,
        0.123,
        0.16,
        0.197,
        0.236,
        0.275,
        0.314,
        0.355,
        0.397,
        0.44,
        0.484,
        0.529,
        0.576,
        0.623,
        0.673,
        0.723,
        0.776,
        0.83,
        0.886,
        0.944,
        1.003,
        1.065,
        1.13,
        1.197,
        1.266,
        1.338,
        1.413,
        1.492,
        1.573,
        1.659,
        1.748,
        1.841,
        1.938,
        2.041,
    ],
    dtype=jnp.float64,
)

DEFAULT_NS = 0.9652
DEFAULT_OMEGA_B = 0.02233
DEFAULT_H = 0.6737


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-ptcl", type=int, default=1024)
    parser.add_argument("--box-size", type=float, default=1000.0)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--a-start", type=float, default=1 / 64)
    parser.add_argument("--a-nbody-maxstep", type=float, default=1 / 64)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.07)
    parser.add_argument("--max-share-ptcl", type=int, default=350_000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=1_200_000)
    parser.add_argument("--lpt-share-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--solver",
        choices=("nbody", "nbody_kappa"),
        default="nbody",
        help="Select the forward solver to benchmark.",
    )
    parser.add_argument(
        "--mode",
        choices=("staged", "whole"),
        default="staged",
        help="Benchmark either the staged forward path or a monolithic jitted forward function.",
    )
    parser.add_argument(
        "--timed-runs",
        type=int,
        default=4,
        help="Number of steady-state runs after the first compile+execute run.",
    )
    parser.add_argument(
        "--no-map-schedule",
        dest="use_map_schedule",
        action="store_false",
        help="Disable the custom scale-factor schedule used by pmpp_1024_maps.py.",
    )
    parser.set_defaults(use_map_schedule=True)
    parser.add_argument(
        "--no-scatter",
        dest="include_scatter",
        action="store_false",
        help="Stop after nbody instead of scattering the final density field.",
    )
    parser.set_defaults(include_scatter=True)
    parser.add_argument(
        "--slice-width",
        type=float,
        default=102.5390625,
        help="Physical slice width used by the saved-map path in nbody_kappa.",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=REPO_ROOT / "scripts/cosmo_parameters_uniform.npy",
        help="Seed -> cosmology table used by pmpp_1024_maps.py.",
    )
    parser.add_argument("--omega-m", type=float, default=None)
    parser.add_argument("--sigma8", type=float, default=None)
    parser.add_argument("--n-s", type=float, default=DEFAULT_NS)
    parser.add_argument("--omega-b", type=float, default=DEFAULT_OMEGA_B)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument(
        "--compilation-cache-dir",
        type=Path,
        default=None,
        help="Optional JAX compilation cache directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file for timings and summary metrics.",
    )
    return parser.parse_args()


def maybe_set_compilation_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    from jax.experimental.compilation_cache import compilation_cache

    compilation_cache.set_cache_dir(str(cache_dir))


def resolve_gpu_devices(num_devices: int) -> list[jax.Device]:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpu_devices) < num_devices:
        raise RuntimeError(
            f"Requested {num_devices} GPU devices but only found {len(gpu_devices)}: {gpu_devices}"
        )
    return gpu_devices[:num_devices]


def block_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def to_python(tree):
    return jax.tree_util.tree_map(
        lambda x: x.item() if hasattr(x, "shape") and x.shape == () else x,
        jax.device_get(tree),
    )


def timed_call(fn, *args, **kwargs):
    start = perf_counter()
    out = fn(*args, **kwargs)
    block_tree(out)
    return out, perf_counter() - start


def load_cosmology_args(args: argparse.Namespace) -> tuple[float, float]:
    if args.omega_m is not None and args.sigma8 is not None:
        return args.omega_m, args.sigma8
    if not args.params_file.exists():
        raise FileNotFoundError(
            f"Could not find params file: {args.params_file}. "
            "Pass --omega-m and --sigma8 explicitly instead."
        )
    params = np.load(args.params_file)
    matching = params[params[:, 0].astype(int) == int(args.seed)]
    if matching.size == 0:
        raise ValueError(f"Seed {args.seed} not found in {args.params_file}")
    _, omega_m, sigma8 = matching[0]
    return float(omega_m), float(sigma8)


def init_conf(args: argparse.Namespace, gpu_devices: list[jax.Device]) -> Configuration:
    ptcl_grid_shape = (args.num_ptcl,) * 3
    ptcl_spacing = args.box_size / ptcl_grid_shape[0]
    compute_mesh = create_compute_mesh(gpu_devices)
    max_ptcl_per_slice = int((args.num_ptcl**3 / len(gpu_devices)) * args.max_ptcl_factor)

    conf = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=args.mesh_shape,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
        to_save_z=TARGET_Z.tolist() if args.use_map_schedule else None,
        a_start=args.a_start,
        a_nbody_maxstep=args.a_nbody_maxstep,
    )

    if args.use_map_schedule:
        conf = conf.replace(a_custom=get_a_schedule(TARGET_Z, conf))

    if args.solver == "nbody_kappa":
        if conf.to_save_a is None:
            raise ValueError("nbody_kappa requires a save schedule. Do not use --no-map-schedule.")
        slice_width_cells = args.slice_width / conf.cell_size
        slice_coords = get_slice_coords(slice_width_cells, len(conf.to_save_a))
        conf = conf.replace(
            slice_to_save=slice_coords,
            max_slice_width=math.ceil(slice_width_cells),
        )

    return conf


def init_lpt_conf(conf: Configuration, multiplier: float) -> Configuration:
    return conf.replace(max_share_ptcl=int(conf.max_share_ptcl * multiplier))


def get_slice_coords(slice_width: float, num_slices: int) -> list[int]:
    coords = [0]
    curr = 0
    round_up = True
    for _ in range(num_slices):
        step = math.ceil(slice_width) if round_up else math.floor(slice_width)
        curr += step
        coords.append(curr)
        round_up = not round_up
    return coords


def build_cosmo(conf: Configuration, args: argparse.Namespace, omega_m: float, sigma8: float):
    cosmo = Cosmology.from_sigma8(
        conf,
        sigma8=sigma8,
        n_s=args.n_s,
        Omega_m=omega_m,
        Omega_b=args.omega_b,
        h=args.h,
    )
    return boltzmann(cosmo, conf)


def summarize_forward(ptcl, dens, conf: Configuration, include_scatter: bool) -> dict[str, object]:
    summary = {
        "num_particles": int(conf.ptcl_num),
        "nbody_steps": int(conf.a_nbody_num),
        "mesh_shape": [int(v) for v in conf.mesh_shape],
        "used_particle_slots": int(jax.device_get(jnp.sum(~ptcl.unused_index))),
        "unused_particle_slots": int(jax.device_get(jnp.sum(ptcl.unused_index))),
        "acc_is_none": ptcl.acc is None,
    }
    if include_scatter:
        dens_mean = float(jax.device_get(jnp.mean(dens)))
        dens_sum = float(jax.device_get(jnp.sum(dens)))
        summary.update(
            {
                "density_mean": dens_mean,
                "density_sum": dens_sum,
                "expected_density_mean": 1.0,
                "expected_density_sum": float(conf.mesh_size),
                "density_mean_error": dens_mean - 1.0,
                "density_sum_error": dens_sum - float(conf.mesh_size),
            }
        )
    return summary


def summarize_kappa_maps(saved_maps, conf: Configuration) -> dict[str, object]:
    final_map = saved_maps[-1]
    map_mean = float(jax.device_get(jnp.mean(final_map)))
    map_sum = float(jax.device_get(jnp.sum(final_map)))
    return {
        "num_particles": int(conf.ptcl_num),
        "nbody_steps": int(conf.a_nbody_num),
        "mesh_shape": [int(v) for v in conf.mesh_shape],
        "num_saved_maps": int(saved_maps.shape[0]),
        "saved_maps_shape": [int(v) for v in saved_maps.shape],
        "final_map_mean": map_mean,
        "final_map_sum": map_sum,
        "slice_width_cells": int(conf.max_slice_width),
    }


def staged_forward_run(
    args: argparse.Namespace,
    conf: Configuration,
    conf_lpt: Configuration,
    seed: int,
    omega_m: float,
    sigma8: float,
):
    solver_fn = nbody_kappa if args.solver == "nbody_kappa" else nbody
    nbody_jit = jax.jit(solver_fn, static_argnames=("conf", "reverse"))
    scatter_jit = jax.jit(scatter, static_argnames=("conf",))

    stage_sec: dict[str, float] = {}
    total_start = perf_counter()

    cosmo, stage_sec["cosmo_and_boltzmann"] = timed_call(build_cosmo, conf, args, omega_m, sigma8)
    modes, stage_sec["white_noise"] = timed_call(white_noise, seed, conf)
    modes, stage_sec["linear_modes"] = timed_call(linear_modes, modes, cosmo, conf)
    ptcl_lpt, stage_sec["lpt"] = timed_call(lpt, modes, cosmo, conf_lpt)
    solver_out, stage_sec["nbody"] = timed_call(nbody_jit, ptcl_lpt, cosmo, conf)

    if args.solver == "nbody_kappa":
        if args.include_scatter:
            stage_sec["scatter"] = 0.0
        summary_start = perf_counter()
        summary = summarize_kappa_maps(solver_out, conf)
        stage_sec["summary"] = perf_counter() - summary_start
        total_sec = perf_counter() - total_start
        return {
            "total_sec": total_sec,
            "stage_sec": stage_sec,
            "summary": summary,
        }

    dens = None
    if args.include_scatter:
        dens, stage_sec["scatter"] = timed_call(scatter_jit, solver_out, conf)

    summary_start = perf_counter()
    summary = summarize_forward(solver_out, dens, conf, include_scatter=args.include_scatter)
    stage_sec["summary"] = perf_counter() - summary_start
    total_sec = perf_counter() - total_start

    return {
        "total_sec": total_sec,
        "stage_sec": stage_sec,
        "summary": summary,
    }


def whole_forward_summary(
    seed: int,
    omega_m: float,
    sigma8: float,
    n_s: float,
    omega_b: float,
    hubble: float,
    conf: Configuration,
    conf_lpt: Configuration,
    include_scatter: bool,
    solver_name: str,
):
    cosmo = Cosmology.from_sigma8(
        conf,
        sigma8=sigma8,
        n_s=n_s,
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=hubble,
    )
    cosmo = boltzmann(cosmo, conf)
    modes = white_noise(seed, conf)
    modes = linear_modes(modes, cosmo, conf)
    ptcl = lpt(modes, cosmo, conf_lpt)
    if solver_name == "nbody_kappa":
        saved_maps = nbody_kappa(ptcl, cosmo, conf)
        final_map = saved_maps[-1]
        return {
            "final_map_mean": jnp.mean(final_map),
            "final_map_sum": jnp.sum(final_map),
            "num_saved_maps": jnp.asarray(saved_maps.shape[0], dtype=jnp.int32),
        }

    ptcl = nbody(ptcl, cosmo, conf)

    used_slots = jnp.sum(~ptcl.unused_index)
    unused_slots = jnp.sum(ptcl.unused_index)

    if include_scatter:
        dens = scatter(ptcl, conf)
        return {
            "density_mean": jnp.mean(dens),
            "density_sum": jnp.sum(dens),
            "used_particle_slots": used_slots,
            "unused_particle_slots": unused_slots,
        }

    return {
        "used_particle_slots": used_slots,
        "unused_particle_slots": unused_slots,
    }


def whole_forward_run(
    args: argparse.Namespace,
    conf: Configuration,
    conf_lpt: Configuration,
    seed: int,
    omega_m: float,
    sigma8: float,
):
    forward_jit = jax.jit(
        whole_forward_summary,
        static_argnames=("conf", "conf_lpt", "include_scatter", "solver_name"),
    )
    out, total_sec = timed_call(
        forward_jit,
        seed,
        omega_m,
        sigma8,
        args.n_s,
        args.omega_b,
        args.h,
        conf,
        conf_lpt,
        args.include_scatter,
        args.solver,
    )
    summary = to_python(out)
    summary.update(
        {
            "num_particles": int(conf.ptcl_num),
            "nbody_steps": int(conf.a_nbody_num),
            "mesh_shape": [int(v) for v in conf.mesh_shape],
        }
    )
    if args.solver == "nbody_kappa":
        summary["saved_maps_shape"] = [len(conf.to_save_a), 3, conf.nMesh, conf.nMesh]
        summary["slice_width_cells"] = int(conf.max_slice_width)
        return {
            "total_sec": total_sec,
            "summary": summary,
        }
    if args.include_scatter:
        summary["expected_density_mean"] = 1.0
        summary["expected_density_sum"] = float(conf.mesh_size)
        summary["density_mean_error"] = summary["density_mean"] - 1.0
        summary["density_sum_error"] = summary["density_sum"] - float(conf.mesh_size)
    return {
        "total_sec": total_sec,
        "summary": summary,
    }


def print_header(args: argparse.Namespace, gpu_devices: list[jax.Device], conf: Configuration) -> None:
    print("PM++ forward benchmark", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"solver={args.solver}", flush=True)
    print(f"visible_devices={jax.devices()}", flush=True)
    print(f"selected_gpus={gpu_devices}", flush=True)
    print(
        "config="
        f"num_ptcl={args.num_ptcl}^3 "
        f"mesh_shape={args.mesh_shape} "
        f"num_devices={args.num_devices} "
        f"a_start={args.a_start} "
        f"a_nbody_maxstep={args.a_nbody_maxstep} "
        f"nbody_steps={conf.a_nbody_num} "
        f"scatter={'yes' if args.include_scatter else 'no'} "
        f"map_schedule={'yes' if args.use_map_schedule else 'no'}",
        flush=True,
    )
    if args.solver == "nbody_kappa":
        print(
            f"slice_width={args.slice_width} slice_width_cells={conf.max_slice_width} "
            f"num_saved_maps={len(conf.to_save_a)}",
            flush=True,
        )


def print_run(label: str, result: dict[str, object]) -> None:
    print(f"[{label}] total={result['total_sec']:.2f}s", flush=True)
    stage_sec = result.get("stage_sec")
    if stage_sec:
        for stage, duration in stage_sec.items():
            print(f"  {stage}: {duration:.2f}s", flush=True)
    summary = result["summary"]
    if "final_map_mean" in summary:
        print(
            "  final_map="
            f"mean={summary['final_map_mean']:.7f} "
            f"sum={summary['final_map_sum']:.1f} "
            f"saved_maps={summary['num_saved_maps']}",
            flush=True,
        )
        return
    if "density_mean" in summary:
        print(
            "  density="
            f"mean={summary['density_mean']:.7f} "
            f"sum={summary['density_sum']:.1f} "
            f"sum_error={summary['density_sum_error']:.1f}",
            flush=True,
        )
    print(
        "  slots="
        f"used={summary['used_particle_slots']} "
        f"unused={summary['unused_particle_slots']}",
        flush=True,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {path}", flush=True)


def main() -> int:
    args = parse_args()
    maybe_set_compilation_cache(args.compilation_cache_dir)

    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf = init_conf(args, gpu_devices)
    conf_lpt = init_lpt_conf(conf, args.lpt_share_multiplier)
    omega_m, sigma8 = load_cosmology_args(args)

    print_header(args, gpu_devices, conf)
    print(f"cosmology=Omega_m={omega_m} sigma8={sigma8}", flush=True)

    runner = staged_forward_run if args.mode == "staged" else whole_forward_run

    compile_run = runner(args, conf, conf_lpt, args.seed, omega_m, sigma8)
    print_run("compile+run", compile_run)

    steady_runs = []
    for run_idx in range(args.timed_runs):
        result = runner(args, conf, conf_lpt, args.seed, omega_m, sigma8)
        steady_runs.append(result)
        print_run(f"steady-{run_idx + 1}", result)

    payload = {
        "benchmark": "pmpp_forward",
        "solver": args.solver,
        "mode": args.mode,
        "seed": args.seed,
        "cosmology": {
            "Omega_m": omega_m,
            "sigma8": sigma8,
            "n_s": args.n_s,
            "Omega_b": args.omega_b,
            "h": args.h,
        },
        "config": {
            "box_size": args.box_size,
            "num_ptcl": args.num_ptcl,
            "mesh_shape": args.mesh_shape,
            "num_devices": args.num_devices,
            "a_start": args.a_start,
            "a_nbody_maxstep": args.a_nbody_maxstep,
            "nbody_steps": int(conf.a_nbody_num),
            "use_map_schedule": args.use_map_schedule,
            "include_scatter": args.include_scatter,
            "max_ptcl_per_slice": int(conf.max_ptcl_per_slice),
            "max_share_ptcl": int(conf.max_share_ptcl),
            "max_share_gather_ptcl": int(conf.max_share_gather_ptcl),
            "lpt_max_share_ptcl": int(conf_lpt.max_share_ptcl),
        },
        "compile_run": compile_run,
        "steady_runs": steady_runs,
        "selected_gpus": [str(device) for device in gpu_devices],
    }

    if args.output_json is not None:
        write_json(args.output_json, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
