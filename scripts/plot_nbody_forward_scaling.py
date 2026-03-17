#!/usr/bin/env python
"""Plot forward N-body parity and PMPP scaling versus mesh ratio and particle count.

This script does two things:

1. Runs a PMWD-vs-PMPP forward compare at a chosen ``mesh_shape`` and saves a
   density projection parity plot.
2. Sweeps PMPP-only forward runs over ``mesh_shape`` and ``num_ptcl`` and saves
   runtime and memory scaling plots.

The scaling sweep runs each case in a fresh subprocess so device memory stats are
isolated between cases.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from pmwd import (
    Configuration as ConfigurationPMWD,
    SimpleLCDM as SimpleLCDM_PMWD,
    boltzmann as boltzmann_pmwd,
    linear_modes as linear_modes_pmwd,
    lpt as lpt_pmwd,
    nbody as nbody_pmwd,
    scatter as scatter_pmwd,
    white_noise as white_noise_pmwd,
)

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from src.lpt import lpt as lpt_pmpp
from src.modes import linear_modes as linear_modes_pmpp
from src.modes import white_noise as white_noise_pmpp
from src.nbody import nbody as nbody_pmpp
from src.scatter import scatter as scatter_pmpp
from src.utils import create_compute_mesh


@dataclass(frozen=True)
class SweepCase:
    num_ptcl: int
    mesh_shape: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--compare-num-ptcl", type=int, default=8)
    parser.add_argument("--compare-mesh-shape", type=int, default=5)
    parser.add_argument("--scale-num-ptcls", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--scale-mesh-shapes", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--a-start", type=float, default=1 / 64)
    parser.add_argument("--a-stop", type=float, default=1 / 32)
    parser.add_argument("--a-nbody-maxstep", type=float, default=1 / 64)
    parser.add_argument("--max-ptcl-factor", type=float, default=2.5)
    parser.add_argument("--max-share-ptcl", type=int, default=12000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=30000)
    parser.add_argument("--lpt-share-multiplier", type=float, default=4.0)
    parser.add_argument("--scaling-steady-runs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/tests/output/nbody_forward_scaling"),
    )
    parser.add_argument("--worker-case-json", type=Path, default=None)
    parser.add_argument("--worker-output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_gpu_devices(num_devices: int) -> list[jax.Device]:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpu_devices) < num_devices:
        raise RuntimeError(
            f"Requested {num_devices} GPU devices but only found {len(gpu_devices)}: {gpu_devices}"
        )
    return gpu_devices[:num_devices]


def to_numpy(array):
    if array is None:
        return None
    return np.asarray(jax.device_get(array))


def block_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def make_conf(
    box_size: float,
    num_ptcl: int,
    mesh_shape: int,
    gpu_devices: list[jax.Device],
    a_start: float,
    a_stop: float,
    a_nbody_maxstep: float,
    max_ptcl_factor: float,
    max_share_ptcl: int,
    max_share_gather_ptcl: int,
) -> Configuration:
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / ptcl_grid_shape[0]
    compute_mesh = create_compute_mesh(gpu_devices)
    max_ptcl_per_slice = int((num_ptcl**3 / len(gpu_devices)) * max_ptcl_factor)
    return Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=mesh_shape,
        compute_mesh=compute_mesh,
        a_start=a_start,
        a_stop=a_stop,
        a_nbody_maxstep=a_nbody_maxstep,
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=max_share_ptcl,
        max_share_gather_ptcl=max_share_gather_ptcl,
    )


def make_conf_pmwd(conf_pmpp: Configuration) -> ConfigurationPMWD:
    return ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_stop=conf_pmpp.a_stop,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
    )


def init_pmwd_state(box_size: float, seed: int, num_ptcl: int, mesh_shape: int, conf_pmpp: Configuration):
    with jax.default_device(jax.devices("cpu")[0]):
        conf_pmwd = make_conf_pmwd(conf_pmpp)
        cosmo_pmwd = boltzmann_pmwd(SimpleLCDM_PMWD(conf_pmwd), conf_pmwd)
        modes_pmwd = white_noise_pmwd(seed, conf_pmwd)
        modes_pmwd = linear_modes_pmwd(modes_pmwd, cosmo_pmwd, conf_pmwd)
        ptcl_lpt_pmwd, _ = lpt_pmwd(modes_pmwd, cosmo_pmwd, conf_pmwd)

    target_device = resolve_gpu_devices(1)[0]
    ptcl_lpt_pmwd = jax.device_put(ptcl_lpt_pmwd, target_device)
    cosmo_pmwd = jax.device_put(cosmo_pmwd, target_device)
    return ptcl_lpt_pmwd, conf_pmwd, cosmo_pmwd


def init_pmpp_state(conf_pmpp: Configuration, seed: int, lpt_share_multiplier: float):
    cosmo_pmpp = boltzmann_pmpp(SimpleLCDM_PMPP(conf_pmpp), conf_pmpp)
    modes_pmpp = white_noise_pmpp(seed, conf_pmpp)
    modes_pmpp = linear_modes_pmpp(modes_pmpp, cosmo_pmpp, conf_pmpp)
    lpt_conf = conf_pmpp.replace(max_share_ptcl=int(conf_pmpp.max_share_ptcl * lpt_share_multiplier))
    ptcl_lpt_pmpp = lpt_pmpp(modes_pmpp, cosmo_pmpp, lpt_conf)
    return ptcl_lpt_pmpp, cosmo_pmpp


def gpu_layout_from_conf(conf) -> dict:
    global_starts = [int(offset) for offset in np.asarray(conf.offsets)]
    slab_width = int(conf.local_mesh_shape[0])
    owned = []
    for gpu_index, start in enumerate(global_starts):
        end = (start + slab_width) % int(conf.nMesh)
        owned.append({"gpu": gpu_index, "start": start, "end": end, "width": slab_width})

    halo_start = np.asarray(conf.halo_start)
    halo_end = np.asarray(conf.halo_end)
    return {
        "nmesh": int(conf.nMesh),
        "owned_x_slabs": owned,
        "halo_cell_bands": sorted(
            {
                int((halo_start[gpu_index, 0]) % conf.nMesh)
                for gpu_index in range(conf.num_devices)
            }
            | {
                int((halo_end[gpu_index, 0]) % conf.nMesh)
                for gpu_index in range(conf.num_devices)
            }
        ),
    }


def density_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | list[int]]:
    diff = right - left
    return {
        "shape": list(left.shape),
        "sum_left": float(left.sum()),
        "sum_right": float(right.sum()),
        "mean_left": float(left.mean()),
        "mean_right": float(right.mean()),
        "max_abs_diff": float(np.abs(diff).max()),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
    }


def projection_image(density: np.ndarray, projection: str) -> tuple[np.ndarray, str, str]:
    if projection == "x":
        return density.sum(axis=0), "y", "z"
    if projection == "y":
        return density.sum(axis=1).T, "z", "x"
    if projection == "z":
        return density.sum(axis=2).T, "y", "x"
    raise ValueError(f"Unsupported projection {projection}")


def decorate_gpu_layout(ax, projection: str, gpu_layout: dict) -> None:
    if projection == "x":
        ax.text(
            0.03,
            0.04,
            "x slabs are projected out",
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

    for halo_x in gpu_layout["halo_cell_bands"]:
        ax.axvspan(halo_x - 0.5, halo_x + 0.5, color="#ff6f61", alpha=0.14, lw=0)


def save_projection_plot(
    left_density: np.ndarray,
    right_density: np.ndarray,
    left_label: str,
    right_label: str,
    output_path: Path,
    gpu_layout: dict,
    title: str,
) -> None:
    projections = []
    for projection in ("x", "y", "z"):
        left_proj, x_label, y_label = projection_image(left_density, projection)
        right_proj, _, _ = projection_image(right_density, projection)
        diff_proj = right_proj - left_proj
        projections.append((projection, left_proj, right_proj, diff_proj, x_label, y_label))

    log_arrays = [
        np.log10(np.clip(array, 1e-8, None))
        for _, left_proj, right_proj, _, _, _ in projections
        for array in (left_proj, right_proj)
    ]
    log_vmin = min(float(array.min()) for array in log_arrays)
    log_vmax = max(float(array.max()) for array in log_arrays)
    diff_vmax = max(float(np.abs(diff).max()) for _, _, _, diff, _, _ in projections)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-12

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    log_im = None
    diff_im = None

    for row, (projection, left_proj, right_proj, diff_proj, x_label, y_label) in enumerate(projections):
        left_log = np.log10(np.clip(left_proj, 1e-8, None))
        right_log = np.log10(np.clip(right_proj, 1e-8, None))

        log_im = axes[row, 0].imshow(left_log, origin="lower", vmin=log_vmin, vmax=log_vmax, cmap="viridis", aspect="auto")
        axes[row, 0].set_title(left_label)

        axes[row, 1].imshow(right_log, origin="lower", vmin=log_vmin, vmax=log_vmax, cmap="viridis", aspect="auto")
        axes[row, 1].set_title(right_label)

        diff_im = axes[row, 2].imshow(
            diff_proj,
            origin="lower",
            cmap="coolwarm",
            vmin=-diff_vmax,
            vmax=diff_vmax,
            aspect="auto",
        )
        axes[row, 2].set_title(f"{right_label} - {left_label}")

        for col in range(3):
            axes[row, col].set_ylabel(y_label if col == 0 else "")
            axes[row, col].set_xlabel(x_label if row == 2 else "")
            decorate_gpu_layout(axes[row, col], projection, gpu_layout)
            if row != 2:
                axes[row, col].set_xticks([])
            if col != 0:
                axes[row, col].set_yticks([])

        axes[row, 0].text(
            -0.16,
            0.5,
            f"Project {projection}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )

    fig.colorbar(log_im, ax=axes[:, :2], shrink=0.86, label="log10 projected density")
    fig.colorbar(diff_im, ax=axes[:, 2], shrink=0.86, label="projected density residual")
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_compare(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf_pmpp = make_conf(
        box_size=args.box_size,
        num_ptcl=args.compare_num_ptcl,
        mesh_shape=args.compare_mesh_shape,
        gpu_devices=gpu_devices,
        a_start=args.a_start,
        a_stop=args.a_stop,
        a_nbody_maxstep=args.a_nbody_maxstep,
        max_ptcl_factor=args.max_ptcl_factor,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
    )
    ptcl_lpt_pmwd, conf_pmwd, cosmo_pmwd = init_pmwd_state(
        args.box_size,
        args.seed,
        args.compare_num_ptcl,
        args.compare_mesh_shape,
        conf_pmpp,
    )

    pmwd_start = perf_counter()
    ptcl_final_pmwd, _ = nbody_pmwd(ptcl_lpt_pmwd, None, cosmo_pmwd, conf_pmwd)
    dens_pmwd = to_numpy(scatter_pmwd(ptcl_final_pmwd, conf_pmwd))
    pmwd_elapsed = perf_counter() - pmwd_start

    nbody_pmpp_jit = jax.jit(nbody_pmpp, static_argnames=("conf", "reverse"))
    ptcl_lpt_pmpp, cosmo_pmpp = init_pmpp_state(conf_pmpp, args.seed, args.lpt_share_multiplier)
    pmpp_start = perf_counter()
    ptcl_final_pmpp = nbody_pmpp_jit(ptcl_lpt_pmpp, cosmo_pmpp, conf_pmpp)
    dens_pmpp = to_numpy(scatter_pmpp(ptcl_final_pmpp, conf_pmpp))
    pmpp_elapsed = perf_counter() - pmpp_start

    metrics = density_metrics(dens_pmwd, dens_pmpp)
    metrics["pmwd_elapsed_seconds"] = pmwd_elapsed
    metrics["pmpp_elapsed_seconds"] = pmpp_elapsed

    plot_path = output_dir / f"nbody_forward_pmwd_vs_pmpp_mesh{args.compare_mesh_shape}.png"
    save_projection_plot(
        dens_pmwd,
        dens_pmpp,
        "PMWD",
        "PMPP",
        plot_path,
        gpu_layout_from_conf(conf_pmpp),
        title=(
            f"Forward N-body density parity, mesh_shape={args.compare_mesh_shape}, "
            f"num_ptcl={args.compare_num_ptcl}^3"
        ),
    )

    return {
        "plot_path": str(plot_path),
        "metrics": metrics,
    }


def device_memory_stats(devices: list[jax.Device]) -> dict[str, object] | None:
    totals = {"bytes_in_use": 0, "peak_bytes_in_use": 0}
    per_device = []
    available = False
    for device in devices:
        try:
            stats = device.memory_stats()
        except Exception:
            stats = None
        if not stats:
            per_device.append({"device": str(device), "bytes_in_use": None, "peak_bytes_in_use": None})
            continue
        available = True
        bytes_in_use = stats.get("bytes_in_use")
        peak_bytes_in_use = stats.get("peak_bytes_in_use")
        per_device.append(
            {
                "device": str(device),
                "bytes_in_use": bytes_in_use,
                "peak_bytes_in_use": peak_bytes_in_use,
            }
        )
        totals["bytes_in_use"] += int(bytes_in_use)
        totals["peak_bytes_in_use"] += int(peak_bytes_in_use)
    if not available:
        return None
    return {"total": totals, "per_device": per_device}


def run_single_scaling_case(case: SweepCase, args: argparse.Namespace) -> dict[str, object]:
    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf = make_conf(
        box_size=args.box_size,
        num_ptcl=case.num_ptcl,
        mesh_shape=case.mesh_shape,
        gpu_devices=gpu_devices,
        a_start=args.a_start,
        a_stop=args.a_stop,
        a_nbody_maxstep=args.a_nbody_maxstep,
        max_ptcl_factor=args.max_ptcl_factor,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
    )
    nbody_pmpp_jit = jax.jit(nbody_pmpp, static_argnames=("conf", "reverse"))
    scatter_pmpp_jit = jax.jit(scatter_pmpp, static_argnames=("conf",))

    tracemalloc.start()
    mem_before = device_memory_stats(gpu_devices)
    host_before, _ = tracemalloc.get_traced_memory()

    compile_start = perf_counter()
    ptcl_lpt, cosmo = init_pmpp_state(conf, args.seed, args.lpt_share_multiplier)
    ptcl_final = nbody_pmpp_jit(ptcl_lpt, cosmo, conf)
    dens = scatter_pmpp_jit(ptcl_final, conf)
    block_tree((ptcl_final, dens))
    compile_elapsed = perf_counter() - compile_start

    host_after_compile, host_peak_compile = tracemalloc.get_traced_memory()
    mem_after_compile = device_memory_stats(gpu_devices)

    steady_times = []
    for _ in range(args.scaling_steady_runs):
        start = perf_counter()
        ptcl_final = nbody_pmpp_jit(ptcl_lpt, cosmo, conf)
        dens = scatter_pmpp_jit(ptcl_final, conf)
        block_tree((ptcl_final, dens))
        steady_times.append(perf_counter() - start)

    host_after_steady, host_peak_steady = tracemalloc.get_traced_memory()
    mem_after_steady = device_memory_stats(gpu_devices)
    tracemalloc.stop()

    dens_np = to_numpy(dens)
    result = {
        "num_ptcl": case.num_ptcl,
        "mesh_shape": case.mesh_shape,
        "mesh_n": int(conf.nMesh),
        "num_particles_total": int(case.num_ptcl**3),
        "num_mesh_cells_total": int(np.prod(conf.mesh_shape)),
        "compile_run_seconds": compile_elapsed,
        "steady_run_seconds": float(np.mean(steady_times)) if steady_times else None,
        "steady_run_seconds_all": steady_times,
        "host_bytes_delta_compile": int(host_after_compile - host_before),
        "host_peak_bytes_compile": int(host_peak_compile - host_before),
        "host_bytes_delta_steady": int(host_after_steady - host_after_compile),
        "host_peak_bytes_steady": int(host_peak_steady - host_after_compile),
        "device_memory_before": mem_before,
        "device_memory_after_compile": mem_after_compile,
        "device_memory_after_steady": mem_after_steady,
        "density_mean": float(dens_np.mean()),
        "density_sum": float(dens_np.sum()),
        "density_sum_error": float(dens_np.sum() - float(conf.mesh_size)),
    }
    return result


def save_scaling_plots(records: list[dict[str, object]], output_dir: Path) -> dict[str, str]:
    if not records:
        return {}

    mesh_shapes = sorted({int(record["mesh_shape"]) for record in records})
    num_ptcls = sorted({int(record["num_ptcl"]) for record in records})

    runtime_particles_path = output_dir / "nbody_forward_scaling_runtime_vs_particles.png"
    runtime_mesh_path = output_dir / "nbody_forward_scaling_runtime_vs_meshshape.png"
    memory_particles_path = output_dir / "nbody_forward_scaling_memory_vs_particles.png"
    memory_mesh_path = output_dir / "nbody_forward_scaling_memory_vs_meshshape.png"

    def peak_total(record: dict[str, object], key: str) -> float | None:
        stats = record.get(key)
        if not stats:
            return None
        return float(stats["total"]["peak_bytes_in_use"])

    compile_by_mesh = {mesh_shape: [] for mesh_shape in mesh_shapes}
    steady_by_mesh = {mesh_shape: [] for mesh_shape in mesh_shapes}
    mem_by_mesh = {mesh_shape: [] for mesh_shape in mesh_shapes}
    for mesh_shape in mesh_shapes:
        for num_ptcl in num_ptcls:
            record = next(
                (item for item in records if int(item["mesh_shape"]) == mesh_shape and int(item["num_ptcl"]) == num_ptcl),
                None,
            )
            if record is None:
                continue
            compile_by_mesh[mesh_shape].append((num_ptcl, float(record["compile_run_seconds"])))
            steady = record.get("steady_run_seconds")
            if steady is not None:
                steady_by_mesh[mesh_shape].append((num_ptcl, float(steady)))
            mem_peak = peak_total(record, "device_memory_after_compile")
            if mem_peak is not None:
                mem_by_mesh[mesh_shape].append((num_ptcl, mem_peak / 1024**3))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for mesh_shape in mesh_shapes:
        if compile_by_mesh[mesh_shape]:
            x, y = zip(*compile_by_mesh[mesh_shape])
            axes[0].plot(x, y, marker="o", linewidth=1.8, label=f"mesh_shape={mesh_shape}")
        if steady_by_mesh[mesh_shape]:
            x, y = zip(*steady_by_mesh[mesh_shape])
            axes[1].plot(x, y, marker="o", linewidth=1.8, label=f"mesh_shape={mesh_shape}")
    axes[0].set_title("Compile+run vs particle grid")
    axes[1].set_title("Steady run vs particle grid")
    for ax in axes:
        ax.set_xlabel("num_ptcl")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(runtime_particles_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for num_ptcl in num_ptcls:
        subset = [record for record in records if int(record["num_ptcl"]) == num_ptcl]
        subset = sorted(subset, key=lambda item: int(item["mesh_shape"]))
        if subset:
            axes[0].plot(
                [int(item["mesh_shape"]) for item in subset],
                [float(item["compile_run_seconds"]) for item in subset],
                marker="o",
                linewidth=1.8,
                label=f"num_ptcl={num_ptcl}",
            )
            steady_subset = [item for item in subset if item.get("steady_run_seconds") is not None]
            if steady_subset:
                axes[1].plot(
                    [int(item["mesh_shape"]) for item in steady_subset],
                    [float(item["steady_run_seconds"]) for item in steady_subset],
                    marker="o",
                    linewidth=1.8,
                    label=f"num_ptcl={num_ptcl}",
                )
    axes[0].set_title("Compile+run vs mesh ratio")
    axes[1].set_title("Steady run vs mesh ratio")
    for ax in axes:
        ax.set_xlabel("mesh_shape")
        ax.set_ylabel("seconds")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(runtime_mesh_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for mesh_shape in mesh_shapes:
        if mem_by_mesh[mesh_shape]:
            x, y = zip(*mem_by_mesh[mesh_shape])
            axes[0].plot(x, y, marker="o", linewidth=1.8, label=f"mesh_shape={mesh_shape}")
    for num_ptcl in num_ptcls:
        subset = [record for record in records if int(record["num_ptcl"]) == num_ptcl]
        subset = sorted(subset, key=lambda item: int(item["mesh_shape"]))
        subset = [
            (int(item["mesh_shape"]), peak_total(item, "device_memory_after_compile") / 1024**3)
            for item in subset
            if peak_total(item, "device_memory_after_compile") is not None
        ]
        if subset:
            x, y = zip(*subset)
            axes[1].plot(x, y, marker="o", linewidth=1.8, label=f"num_ptcl={num_ptcl}")
    axes[0].set_title("Device peak memory vs particle grid")
    axes[1].set_title("Device peak memory vs mesh ratio")
    axes[0].set_xlabel("num_ptcl")
    axes[1].set_xlabel("mesh_shape")
    for ax in axes:
        ax.set_ylabel("GiB across selected GPUs")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(memory_particles_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for mesh_shape in mesh_shapes:
        subset = [record for record in records if int(record["mesh_shape"]) == mesh_shape]
        subset = sorted(subset, key=lambda item: int(item["num_mesh_cells_total"]))
        if subset:
            axes[0].plot(
                [int(item["num_mesh_cells_total"]) for item in subset],
                [float(item["compile_run_seconds"]) for item in subset],
                marker="o",
                linewidth=1.8,
                label=f"mesh_shape={mesh_shape}",
            )
        mem_subset = [
            (int(item["num_mesh_cells_total"]), peak_total(item, "device_memory_after_compile") / 1024**3)
            for item in subset
            if peak_total(item, "device_memory_after_compile") is not None
        ]
        if mem_subset:
            x, y = zip(*mem_subset)
            axes[1].plot(x, y, marker="o", linewidth=1.8, label=f"mesh_shape={mesh_shape}")
    axes[0].set_title("Compile+run vs total mesh cells")
    axes[1].set_title("Device peak memory vs total mesh cells")
    axes[0].set_xlabel("total mesh cells")
    axes[1].set_xlabel("total mesh cells")
    axes[0].set_ylabel("seconds")
    axes[1].set_ylabel("GiB across selected GPUs")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(memory_mesh_path, dpi=180)
    plt.close(fig)

    return {
        "runtime_vs_particles": str(runtime_particles_path),
        "runtime_vs_meshshape": str(runtime_mesh_path),
        "memory_vs_particles": str(memory_particles_path),
        "memory_vs_meshshape": str(memory_mesh_path),
    }


def run_scaling_driver(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    worker_root = output_dir / "scaling_cases"
    worker_root.mkdir(parents=True, exist_ok=True)
    cases = [SweepCase(num_ptcl=num_ptcl, mesh_shape=mesh_shape) for num_ptcl in args.scale_num_ptcls for mesh_shape in args.scale_mesh_shapes]

    summary_records = []
    for case in cases:
        case_name = f"np{case.num_ptcl}_mesh{case.mesh_shape}"
        case_dir = worker_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        case_json = case_dir / "case.json"
        result_json = case_dir / "result.json"
        case_json.write_text(json.dumps(asdict(case), indent=2))

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--box-size",
            str(args.box_size),
            "--seed",
            str(args.seed),
            "--num-devices",
            str(args.num_devices),
            "--a-start",
            str(args.a_start),
            "--a-stop",
            str(args.a_stop),
            "--a-nbody-maxstep",
            str(args.a_nbody_maxstep),
            "--max-ptcl-factor",
            str(args.max_ptcl_factor),
            "--max-share-ptcl",
            str(args.max_share_ptcl),
            "--max-share-gather-ptcl",
            str(args.max_share_gather_ptcl),
            "--lpt-share-multiplier",
            str(args.lpt_share_multiplier),
            "--scaling-steady-runs",
            str(args.scaling_steady_runs),
            "--worker-case-json",
            str(case_json),
            "--worker-output-json",
            str(result_json),
        ]
        started = perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        elapsed = perf_counter() - started
        (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Scaling case {case_name} failed. See {case_dir / 'stderr.log'}")
        record = json.loads(result_json.read_text(encoding="utf-8"))
        record["driver_elapsed_seconds"] = elapsed
        summary_records.append(record)

    plots = save_scaling_plots(summary_records, output_dir)
    summary_path = output_dir / "nbody_forward_scaling_summary.json"
    payload = {"records": summary_records, "plots": plots}
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary_path": str(summary_path), "plots": plots, "records": summary_records}


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_case_json is None or args.worker_output_json is None:
        raise ValueError("worker mode requires --worker-case-json and --worker-output-json")
    case = SweepCase(**json.loads(args.worker_case_json.read_text(encoding="utf-8")))
    result = run_single_scaling_case(case, args)
    args.worker_output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_case_json is not None or args.worker_output_json is not None:
        return worker_main(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compare = run_compare(args, args.output_dir)
    scaling = run_scaling_driver(args, args.output_dir)

    summary = {
        "compare": compare,
        "scaling": {
            "summary_path": scaling["summary_path"],
            "plots": scaling["plots"],
        },
    }
    summary_path = args.output_dir / "nbody_forward_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved compare plot to {compare['plot_path']}")
    print(f"Saved scaling summary to {scaling['summary_path']}")
    for name, path in scaling["plots"].items():
        print(f"Saved {name} plot to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
