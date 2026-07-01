#!/usr/bin/env python
"""Plot PMPP full N-body final-density projections across resolutions with nested noise.

The parent process launches one worker subprocess per resolution so each run gets an
isolated JAX compilation cache and fresh GPU memory state. Each worker runs:

    white_noise_nested -> linear_modes -> lpt -> nbody -> scatter

for a fixed box size, fixed seed, and fixed ``mesh_shape``, then writes a projected
density map and metrics. The parent process assembles the multi-panel plot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import numpy as np
from matplotlib import pyplot as plt

from pmpp.boltzmann import boltzmann
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise_nested
from pmpp.nbody import nbody
from pmpp.scatter import scatter
from pmpp.utils import create_compute_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--num-ptcls", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--mesh-shape", type=int, default=2)
    parser.add_argument("--projection", choices=("x", "y", "z"), default="y")
    parser.add_argument("--a-start", type=float, default=1 / 64)
    parser.add_argument("--a-stop", type=float, default=1.0)
    parser.add_argument("--a-nbody-maxstep", type=float, default=1 / 64)
    parser.add_argument("--max-ptcl-factor", type=float, default=2.25)
    parser.add_argument("--base-share-ptcl", type=int, default=12000)
    parser.add_argument("--share-ptcl-factor", type=float, default=8.0)
    parser.add_argument("--gather-share-multiplier", type=float, default=3.0)
    parser.add_argument("--lpt-share-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/tests/output/pmpp_nested_nbody_mesh2"),
    )
    parser.add_argument("--worker-num-ptcl", type=int, default=None)
    parser.add_argument("--worker-case-dir", type=Path, default=None)
    return parser.parse_args()


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


def to_numpy(array):
    if array is None:
        return None
    return np.asarray(jax.device_get(array))


def projected_density_image(density: np.ndarray, projection: str) -> tuple[np.ndarray, str, str]:
    if projection == "x":
        return density.sum(axis=0), "y mesh index", "z mesh index"
    if projection == "y":
        return density.sum(axis=1).T, "x mesh index", "z mesh index"
    if projection == "z":
        return density.sum(axis=2).T, "x mesh index", "y mesh index"
    raise ValueError(f"Unsupported projection {projection}")


def suggest_share_caps(num_ptcl: int, base_share_ptcl: int, share_ptcl_factor: float, gather_share_multiplier: float) -> tuple[int, int]:
    max_share_ptcl = max(base_share_ptcl, int(round((num_ptcl**2) * share_ptcl_factor)))
    max_share_gather_ptcl = max(int(base_share_ptcl * gather_share_multiplier), int(round(max_share_ptcl * gather_share_multiplier)))
    return max_share_ptcl, max_share_gather_ptcl


def make_conf(
    box_size: float,
    num_ptcl: int,
    mesh_shape: int,
    gpu_devices: list[jax.Device],
    a_start: float,
    a_stop: float,
    a_nbody_maxstep: float,
    max_ptcl_factor: float,
    base_share_ptcl: int,
    share_ptcl_factor: float,
    gather_share_multiplier: float,
) -> Configuration:
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / num_ptcl
    compute_mesh = create_compute_mesh(gpu_devices)
    max_ptcl_per_slice = int((num_ptcl**3 / len(gpu_devices)) * max_ptcl_factor)
    max_share_ptcl, max_share_gather_ptcl = suggest_share_caps(
        num_ptcl,
        base_share_ptcl,
        share_ptcl_factor,
        gather_share_multiplier,
    )
    return Configuration(
        ptcl_spacing=ptcl_spacing,
        ptcl_grid_shape=ptcl_grid_shape,
        mesh_shape=mesh_shape,
        compute_mesh=compute_mesh,
        a_start=a_start,
        a_stop=a_stop,
        a_nbody_maxstep=a_nbody_maxstep,
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=max_share_ptcl,
        max_share_gather_ptcl=max_share_gather_ptcl,
    )


def gpu_layout_from_conf(conf: Configuration) -> dict:
    offsets = [int(x) for x in np.asarray(conf.offsets)]
    slab_width = int(conf.local_mesh_shape[0])
    return {
        "nmesh": int(conf.nMesh),
        "offsets": offsets,
        "local_mesh_shape": [int(x) for x in conf.local_mesh_shape],
        "owned_x_slabs": [
            {
                "gpu": gpu_index,
                "start": start,
                "width": slab_width,
            }
            for gpu_index, start in enumerate(offsets)
        ],
    }


def decorate_x_slab_layout(ax, projection: str, gpu_layout: dict) -> None:
    if projection == "x":
        ax.text(
            0.03,
            0.04,
            "x slabs projected out",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 2},
        )
        return

    colors = ["#f4d06f", "#7fb3d5", "#d2b4de", "#82e0aa"]
    for slab in gpu_layout["owned_x_slabs"]:
        color = colors[slab["gpu"] % len(colors)]
        ax.axvspan(
            slab["start"] - 0.5,
            slab["start"] + slab["width"] - 0.5,
            color=color,
            alpha=0.10,
            lw=0,
        )
        ax.axvline(slab["start"] - 0.5, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axvline(gpu_layout["nmesh"] - 0.5, color=colors[0], linestyle="--", linewidth=1.0, alpha=0.8)


def run_worker(args: argparse.Namespace) -> None:
    if args.worker_case_dir is None:
        raise ValueError("--worker-case-dir is required with --worker-num-ptcl")

    case_dir = args.worker_case_dir
    case_dir.mkdir(parents=True, exist_ok=True)

    num_ptcl = int(args.worker_num_ptcl)
    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf = make_conf(
        box_size=args.box_size,
        num_ptcl=num_ptcl,
        mesh_shape=args.mesh_shape,
        gpu_devices=gpu_devices,
        a_start=args.a_start,
        a_stop=args.a_stop,
        a_nbody_maxstep=args.a_nbody_maxstep,
        max_ptcl_factor=args.max_ptcl_factor,
        base_share_ptcl=args.base_share_ptcl,
        share_ptcl_factor=args.share_ptcl_factor,
        gather_share_multiplier=args.gather_share_multiplier,
    )

    nbody_jit = jax.jit(nbody, static_argnames=("conf", "reverse"))
    scatter_jit = jax.jit(scatter, static_argnames=("conf",))

    timings: dict[str, float] = {}

    start = perf_counter()
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    block_tree(cosmo)
    timings["boltzmann"] = perf_counter() - start

    start = perf_counter()
    white = white_noise_nested(args.seed, conf)
    block_tree(white)
    timings["white_noise_nested"] = perf_counter() - start

    start = perf_counter()
    modes = linear_modes(white, cosmo, conf)
    block_tree(modes)
    timings["linear_modes"] = perf_counter() - start

    lpt_conf = conf.replace(max_share_ptcl=int(conf.max_share_ptcl * args.lpt_share_multiplier))
    start = perf_counter()
    ptcl_lpt = lpt(modes, cosmo, lpt_conf)
    block_tree(ptcl_lpt)
    timings["lpt"] = perf_counter() - start

    start = perf_counter()
    ptcl_final = nbody_jit(ptcl_lpt, cosmo, conf)
    block_tree(ptcl_final)
    timings["nbody"] = perf_counter() - start

    start = perf_counter()
    density = scatter_jit(ptcl_final, conf)
    density = to_numpy(density)
    timings["scatter"] = perf_counter() - start
    timings["total"] = sum(timings.values())

    projection, x_label, y_label = projected_density_image(density, args.projection)
    projection_path = case_dir / f"projection_{args.projection}.npy"
    np.save(projection_path, projection)

    result = {
        "num_ptcl": num_ptcl,
        "mesh_shape": args.mesh_shape,
        "projection": args.projection,
        "projection_path": str(projection_path),
        "projection_shape": list(projection.shape),
        "axis_labels": {"x": x_label, "y": y_label},
        "timings_sec": timings,
        "density_shape": list(density.shape),
        "density_mean": float(density.mean()),
        "density_sum": float(density.sum()),
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "nbody_steps": int(conf.a_nbody_num),
        "gpu_layout": gpu_layout_from_conf(conf),
        "capacity": {
            "max_ptcl_per_slice": int(conf.max_ptcl_per_slice),
            "max_share_ptcl": int(conf.max_share_ptcl),
            "max_share_gather_ptcl": int(conf.max_share_gather_ptcl),
            "lpt_max_share_ptcl": int(lpt_conf.max_share_ptcl),
        },
    }

    (case_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def run_parent(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for num_ptcl in args.num_ptcls:
        case_dir = output_dir / f"nbody_nested_mesh{args.mesh_shape}_{num_ptcl}"
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / "console.log"

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--box-size",
            str(args.box_size),
            "--seed",
            str(args.seed),
            "--num-devices",
            str(args.num_devices),
            "--mesh-shape",
            str(args.mesh_shape),
            "--projection",
            args.projection,
            "--a-start",
            str(args.a_start),
            "--a-stop",
            str(args.a_stop),
            "--a-nbody-maxstep",
            str(args.a_nbody_maxstep),
            "--max-ptcl-factor",
            str(args.max_ptcl_factor),
            "--base-share-ptcl",
            str(args.base_share_ptcl),
            "--share-ptcl-factor",
            str(args.share_ptcl_factor),
            "--gather-share-multiplier",
            str(args.gather_share_multiplier),
            "--lpt-share-multiplier",
            str(args.lpt_share_multiplier),
            "--worker-num-ptcl",
            str(num_ptcl),
            "--worker-case-dir",
            str(case_dir),
        ]

        print(f"[parent] running {num_ptcl}^3 -> {case_dir}")
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            raise RuntimeError(f"Worker failed for {num_ptcl}^3. See {log_path}")

        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        results.append(result)

    projection_arrays = [np.load(result["projection_path"]) for result in results]
    log_arrays = [np.log10(np.clip(array, 1e-8, None)) for array in projection_arrays]
    log_vmin = min(float(array.min()) for array in log_arrays)
    log_vmax = max(float(array.max()) for array in log_arrays)

    ncols = 2
    nrows = int(math.ceil(len(results) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    image_artist = None

    for ax, result, projection in zip(axes.flat, results, projection_arrays):
        log_projection = np.log10(np.clip(projection, 1e-8, None))
        image_artist = ax.imshow(
            log_projection,
            origin="lower",
            cmap="viridis",
            vmin=log_vmin,
            vmax=log_vmax,
            aspect="auto",
        )
        decorate_x_slab_layout(ax, result["projection"], result["gpu_layout"])
        ax.set_title(
            f"PMPP {result['num_ptcl']}^3, mesh_shape={result['mesh_shape']}\n"
            f"nMesh={result['gpu_layout']['nmesh']}, total={result['timings_sec']['total']:.1f}s, "
            f"nbody={result['timings_sec']['nbody']:.1f}s"
        )
        ax.set_xlabel(result["axis_labels"]["x"])
        ax.set_ylabel(result["axis_labels"]["y"])
        ax.text(
            0.02,
            0.02,
            f"mean={result['density_mean']:.6f}\n"
            f"sum={result['density_sum']:.1f}\n"
            f"steps={result['nbody_steps']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.42, "pad": 2},
        )

    for ax in axes.flat[len(results):]:
        ax.axis("off")

    fig.colorbar(image_artist, ax=axes.ravel().tolist(), shrink=0.90, label="log10 projected density")
    fig.suptitle(
        f"PMPP nested-noise final density projections at a={args.a_stop:g}\n"
        f"box={args.box_size:g}, seed={args.seed}, mesh_shape={args.mesh_shape}, projection={args.projection}",
        fontsize=15,
    )
    plot_path = output_dir / f"pmpp_nested_nbody_mesh{args.mesh_shape}_{args.projection}_a{args.a_stop:g}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    summary = {
        "parameters": {
            "box_size": args.box_size,
            "seed": args.seed,
            "num_devices": args.num_devices,
            "num_ptcls": args.num_ptcls,
            "mesh_shape": args.mesh_shape,
            "projection": args.projection,
            "a_start": args.a_start,
            "a_stop": args.a_stop,
            "a_nbody_maxstep": args.a_nbody_maxstep,
            "max_ptcl_factor": args.max_ptcl_factor,
            "base_share_ptcl": args.base_share_ptcl,
            "share_ptcl_factor": args.share_ptcl_factor,
            "gather_share_multiplier": args.gather_share_multiplier,
            "lpt_share_multiplier": args.lpt_share_multiplier,
        },
        "plot_path": str(plot_path),
        "cases": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved plot to {plot_path}")
    print(f"Saved summary to {summary_path}")


def main() -> None:
    args = parse_args()
    if args.worker_num_ptcl is not None:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
