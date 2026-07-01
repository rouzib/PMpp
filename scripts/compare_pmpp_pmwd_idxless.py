#!/usr/bin/env python
"""Compare PMWD, current idx-free PMPP, and the previous idx-based PMPP.

The script mirrors the workflow in ``notebooks/mGPU_pmwd_local.ipynb`` and adds:
1. PMWD vs current PMPP projection plots,
2. previous-PMPP vs current-PMPP zero-diff checks,
3. GPU slab / halo overlays on the projections, and
4. a particle-state memory report for removing ``Particles.idx``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

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
    boltzmann,
    linear_modes as linear_modes_pmwd,
    lpt as lpt_pmwd,
    nbody as nbody_pmwd,
    scatter as scatter_pmwd,
    white_noise as white_noise_pmwd,
)

from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise
from pmpp.nbody import nbody
from pmpp.scatter import scatter
from pmpp.utils import create_compute_mesh, pmid_to_idx

CURRENT_PMPP_API = SimpleNamespace(
    Configuration=Configuration,
    SimpleLCDM=SimpleLCDM,
    white_noise=white_noise,
    linear_modes=linear_modes,
    lpt=lpt,
    nbody=nbody,
    scatter=scatter,
    create_compute_mesh=create_compute_mesh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-ptcl", type=int, default=64)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.5)
    parser.add_argument("--max-share-ptcl", type=int, default=10000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=30000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/tests/output"),
    )
    return parser.parse_args()


def resolve_gpu_devices(num_devices: int) -> list[jax.Device]:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if len(gpu_devices) < num_devices:
        raise RuntimeError(
            f"Requested {num_devices} GPU devices but only found {len(gpu_devices)}: {gpu_devices}"
        )
    return gpu_devices[:num_devices]


def local_nbytes(array) -> int:
    if array is None:
        return 0
    if hasattr(array, "addressable_shards"):
        return int(sum(shard.data.nbytes for shard in array.addressable_shards))
    return int(np.asarray(array).nbytes)


def mib(num_bytes: int) -> float:
    return num_bytes / 1024**2


def to_numpy(array):
    if array is None:
        return None
    return np.asarray(jax.device_get(array))


def particle_state_bytes(ptcl) -> dict[str, int]:
    fields = {
        "pmid": local_nbytes(ptcl.pmid),
        "disp": local_nbytes(ptcl.disp),
        "vel": local_nbytes(ptcl.vel),
        "acc": local_nbytes(ptcl.acc),
        "unused_index": local_nbytes(ptcl.unused_index),
        "halo_mask": local_nbytes(ptcl.halo_mask),
    }
    fields["total_without_idx"] = sum(fields.values())
    return fields


def idx_bytes_if_present(ptcl, conf) -> int:
    idx = pmid_to_idx(ptcl.pmid, conf, ptcl.unused_index, dtype=jnp.int32)
    return local_nbytes(idx)


def patch_head_source(text: str) -> str:
    text = text.replace("from jax import NamedSharding", "from jax.sharding import NamedSharding")
    text = text.replace(
        "return isinstance(x, np.ndarray) and x.dtype == float0",
        "return hasattr(x, 'dtype') and x.dtype == float0",
    )
    return text


def materialize_head_src_package(package_name: str = "src_head_idx") -> tuple[Path, str]:
    temp_root = Path(tempfile.mkdtemp(prefix="pmpp_head_idx_"))
    package_root = temp_root / package_name

    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "src"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    for rel_path in result.stdout.splitlines():
        src_text = subprocess.run(
            ["git", "show", f"HEAD:{rel_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        dest = package_root / Path(rel_path).relative_to("src")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(patch_head_source(src_text), encoding="utf-8")

    init_file = package_root / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    return temp_root, package_name


def materialize_working_tree_src_package(package_name: str = "src_clone_current") -> tuple[Path, str]:
    temp_root = Path(tempfile.mkdtemp(prefix="pmpp_current_clone_"))
    package_root = temp_root / package_name
    shutil.copytree(REPO_ROOT / "src", package_root)
    return temp_root, package_name


def load_materialized_api(temp_root: Path, package_name: str) -> SimpleNamespace:
    sys.path.insert(0, str(temp_root))

    configuration_mod = importlib.import_module(f"{package_name}.configuration")
    cosmo_mod = importlib.import_module(f"{package_name}.cosmo")
    lpt_mod = importlib.import_module(f"{package_name}.lpt")
    modes_mod = importlib.import_module(f"{package_name}.modes")
    nbody_mod = importlib.import_module(f"{package_name}.nbody")
    scatter_mod = importlib.import_module(f"{package_name}.scatter")
    utils_mod = importlib.import_module(f"{package_name}.utils")

    return SimpleNamespace(
        Configuration=configuration_mod.Configuration,
        SimpleLCDM=cosmo_mod.SimpleLCDM,
        white_noise=modes_mod.white_noise,
        linear_modes=modes_mod.linear_modes,
        lpt=lpt_mod.lpt,
        nbody=nbody_mod.nbody,
        scatter=scatter_mod.scatter,
        create_compute_mesh=utils_mod.create_compute_mesh,
    )


def load_head_idx_api() -> tuple[SimpleNamespace, Path]:
    temp_root, package_name = materialize_head_src_package()
    return load_materialized_api(temp_root, package_name), temp_root


def load_working_tree_clone_api() -> tuple[SimpleNamespace, Path]:
    temp_root, package_name = materialize_working_tree_src_package()
    return load_materialized_api(temp_root, package_name), temp_root


def cleanup_head_idx_package(temp_root: Path) -> None:
    if str(temp_root) in sys.path:
        sys.path.remove(str(temp_root))
    shutil.rmtree(temp_root, ignore_errors=True)


def init_pmwd(box_size: float, seed: int, num_ptcl: int, mesh_shape: int):
    with jax.default_device(jax.devices("cpu")[0]):
        ptcl_grid_shape = (num_ptcl,) * 3
        ptcl_spacing = box_size / ptcl_grid_shape[0]
        conf = ConfigurationPMWD(
            ptcl_spacing,
            ptcl_grid_shape,
            mesh_shape=mesh_shape,
            a_start=1 / 64,
            a_nbody_maxstep=1 / 64,
            a_stop=1 / 32,
        )
        cosmo = SimpleLCDM(conf)
        cosmo = boltzmann(cosmo, conf)

        modes = white_noise_pmwd(seed, conf)
        modes = linear_modes_pmwd(modes, cosmo, conf)
        ptcl_lpt, _ = lpt_pmwd(modes, cosmo, conf)

    target_device = resolve_gpu_devices(1)[0]
    ptcl_lpt = jax.device_put(ptcl_lpt, target_device)
    cosmo = jax.device_put(cosmo, target_device)
    return ptcl_lpt, conf, cosmo


def init_pmpp(api, args: argparse.Namespace, gpu_devices: list[jax.Device]):
    ptcl_grid_shape = (args.num_ptcl,) * 3
    ptcl_spacing = args.box_size / ptcl_grid_shape[0]
    compute_mesh = api.create_compute_mesh(gpu_devices)
    max_ptcl_per_slice = int((args.num_ptcl**3 / len(gpu_devices)) * args.max_ptcl_factor)

    conf = api.Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=args.mesh_shape,
        compute_mesh=compute_mesh,
        a_start=1 / 64,
        a_nbody_maxstep=1 / 64,
        a_stop=1 / 32,
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
    )
    cosmo = api.SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)
    modes = api.white_noise(args.seed, conf)
    modes = api.linear_modes(modes, cosmo, conf)

    lpt_conf = conf.replace(max_share_ptcl=conf.max_share_ptcl * 2)
    ptcl_lpt = api.lpt(modes, cosmo, lpt_conf)
    return ptcl_lpt, conf, cosmo


def run_pmpp(label: str, api, args: argparse.Namespace, gpu_devices: list[jax.Device]) -> dict:
    nbody_pmpp = jax.jit(api.nbody, static_argnames=("conf", "reverse"))
    start = perf_counter()
    ptcl_lpt, conf, cosmo = init_pmpp(api, args, gpu_devices)
    ptcl_final = nbody_pmpp(ptcl_lpt, cosmo, conf)
    density = to_numpy(api.scatter(ptcl_final, conf))
    elapsed = perf_counter() - start
    return {
        "label": label,
        "ptcl_lpt": ptcl_lpt,
        "ptcl_final": ptcl_final,
        "conf": conf,
        "cosmo": cosmo,
        "density": density,
        "elapsed": elapsed,
    }


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
        "local_mesh_shape": [int(x) for x in conf.local_mesh_shape],
        "offsets": global_starts,
        "owned_x_slabs": owned,
        "particle_storage_slice_start": [int(x) for x in np.asarray(conf.slice_start)],
        "particle_storage_slice_end": [int(x) for x in np.asarray(conf.slice_end)],
        "halo_start": halo_start.astype(int).tolist(),
        "halo_end": halo_end.astype(int).tolist(),
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


def density_metrics(left: np.ndarray, right: np.ndarray) -> dict:
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


def projection_summary(left: np.ndarray, right: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for axis_name, axis in zip(("x", "y", "z"), (0, 1, 2)):
        left_proj = left.sum(axis=axis)
        right_proj = right.sum(axis=axis)
        diff = right_proj - left_proj
        report[axis_name] = {
            "sum_left": float(left_proj.sum()),
            "sum_right": float(right_proj.sum()),
            "max_abs_diff": float(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "rms_diff": float(np.sqrt(np.mean(diff**2))),
        }
    return report


def array_diff_metrics(left, right) -> dict:
    left_np = to_numpy(left)
    right_np = to_numpy(right)
    if left_np is None and right_np is None:
        return {"present": False}
    if left_np.dtype == np.bool_ or right_np.dtype == np.bool_:
        diff = left_np != right_np
        return {
            "present": True,
            "dtype": str(left_np.dtype),
            "shape": list(left_np.shape),
            "exact_equal": bool(np.array_equal(left_np, right_np)),
            "mismatch_count": int(diff.sum()),
        }

    abs_diff = np.abs(right_np - left_np)
    return {
        "present": True,
        "dtype": str(left_np.dtype),
        "shape": list(left_np.shape),
        "exact_equal": bool(np.array_equal(left_np, right_np)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(abs_diff**2))),
    }


def particle_diff_metrics(left_ptcl, right_ptcl) -> dict:
    return {
        "pmid": array_diff_metrics(left_ptcl.pmid, right_ptcl.pmid),
        "disp": array_diff_metrics(left_ptcl.disp, right_ptcl.disp),
        "vel": array_diff_metrics(left_ptcl.vel, right_ptcl.vel),
        "acc": array_diff_metrics(left_ptcl.acc, right_ptcl.acc),
        "unused_index": array_diff_metrics(left_ptcl.unused_index, right_ptcl.unused_index),
        "halo_mask": array_diff_metrics(left_ptcl.halo_mask, right_ptcl.halo_mask),
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
        ax.text(
            slab["start"] + slab["width"] / 2 - 0.5,
            0.98,
            f"GPU {slab['gpu']}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )

    ax.axvline(gpu_layout["nmesh"] - 0.5, color=colors[0], linestyle="--", linewidth=1.0, alpha=0.8)

    for halo_x in gpu_layout["halo_cell_bands"]:
        ax.axvspan(halo_x - 0.5, halo_x + 0.5, color="#ff6f61", alpha=0.14, lw=0)


def save_projection_plot(
    left_density: np.ndarray,
    right_density: np.ndarray,
    left_label: str,
    right_label: str,
    output_path: Path,
    gpu_layout: dict,
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
    fig.suptitle(
        f"{left_label} vs {right_label} density projections\n"
        "Background shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pmwd_plot_path = args.output_dir / "mgpu_pmwd_vs_idxless_with_gpu_layout.png"
    baseline_plot_path = args.output_dir / "mgpu_idx_baseline_vs_idxless_with_gpu_layout.png"
    metrics_path = args.output_dir / "mgpu_pmwd_local_idxless_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)

    pmwd_start = perf_counter()
    ptcl_lpt_pmwd, conf_pmwd, cosmo_pmwd = init_pmwd(args.box_size, args.seed, args.num_ptcl, args.mesh_shape)
    ptcl_final_pmwd, _ = nbody_pmwd(ptcl_lpt_pmwd, None, cosmo_pmwd, conf_pmwd)
    dens_pmwd = to_numpy(scatter_pmwd(ptcl_final_pmwd, conf_pmwd))
    pmwd_elapsed = perf_counter() - pmwd_start

    baseline_api, temp_root = load_head_idx_api()
    try:
        baseline_run = run_pmpp("baseline_idx", baseline_api, args, gpu_devices)
        idxless_run = run_pmpp("idxless", CURRENT_PMPP_API, args, gpu_devices)
    finally:
        cleanup_head_idx_package(temp_root)

    current_clone_api, current_clone_root = load_working_tree_clone_api()
    try:
        current_clone_run = run_pmpp("current_clone", current_clone_api, args, gpu_devices)
    finally:
        cleanup_head_idx_package(current_clone_root)

    gpu_layout = gpu_layout_from_conf(idxless_run["conf"])

    save_projection_plot(
        dens_pmwd,
        idxless_run["density"],
        "PMWD",
        "PMPP idxless",
        pmwd_plot_path,
        gpu_layout,
    )
    save_projection_plot(
        baseline_run["density"],
        idxless_run["density"],
        "PMPP baseline idx",
        "PMPP idxless",
        baseline_plot_path,
        gpu_layout,
    )

    pmwd_vs_idxless = density_metrics(dens_pmwd, idxless_run["density"])
    pmwd_vs_idxless["projection_metrics"] = projection_summary(dens_pmwd, idxless_run["density"])

    baseline_vs_idxless = density_metrics(baseline_run["density"], idxless_run["density"])
    baseline_vs_idxless["projection_metrics"] = projection_summary(
        baseline_run["density"],
        idxless_run["density"],
    )

    current_clone_vs_current = density_metrics(current_clone_run["density"], idxless_run["density"])
    current_clone_vs_current["projection_metrics"] = projection_summary(
        current_clone_run["density"],
        idxless_run["density"],
    )

    particle_state_diff = {
        "lpt": particle_diff_metrics(baseline_run["ptcl_lpt"], idxless_run["ptcl_lpt"]),
        "final": particle_diff_metrics(baseline_run["ptcl_final"], idxless_run["ptcl_final"]),
    }

    lpt_bytes = particle_state_bytes(idxless_run["ptcl_lpt"])
    final_bytes = particle_state_bytes(idxless_run["ptcl_final"])
    idx_lpt_bytes = idx_bytes_if_present(idxless_run["ptcl_lpt"], idxless_run["conf"])
    idx_final_bytes = idx_bytes_if_present(idxless_run["ptcl_final"], idxless_run["conf"])

    report = {
        "parameters": {
            "box_size": args.box_size,
            "seed": args.seed,
            "num_ptcl": args.num_ptcl,
            "mesh_shape": args.mesh_shape,
            "num_devices": args.num_devices,
            "max_ptcl_factor": args.max_ptcl_factor,
            "max_share_ptcl": args.max_share_ptcl,
            "max_share_gather_ptcl": args.max_share_gather_ptcl,
            "max_ptcl_per_slice": int(idxless_run["conf"].max_ptcl_per_slice),
            "reserved_particle_slots": int(idxless_run["conf"].max_ptcl_per_slice * idxless_run["conf"].num_devices),
            "physical_particles": int(idxless_run["conf"].ptcl_num),
        },
        "runtime_seconds": {
            "pmwd_total": pmwd_elapsed,
            "pmpp_baseline_idx_total": baseline_run["elapsed"],
            "pmpp_idxless_total": idxless_run["elapsed"],
        },
        "gpu_layout": gpu_layout,
        "pmwd_vs_idxless": pmwd_vs_idxless,
        "baseline_idx_vs_idxless": baseline_vs_idxless,
        "current_clone_vs_current": current_clone_vs_current,
        "particle_state_differences": particle_state_diff,
        "memory": {
            "lpt_state_bytes": lpt_bytes,
            "final_state_bytes": final_bytes,
            "idx_bytes_if_present_lpt": idx_lpt_bytes,
            "idx_bytes_if_present_final": idx_final_bytes,
            "lpt_state_with_idx_bytes": lpt_bytes["total_without_idx"] + idx_lpt_bytes,
            "final_state_with_idx_bytes": final_bytes["total_without_idx"] + idx_final_bytes,
            "lpt_idx_savings_fraction": idx_lpt_bytes / (lpt_bytes["total_without_idx"] + idx_lpt_bytes),
            "final_idx_savings_fraction": idx_final_bytes / (final_bytes["total_without_idx"] + idx_final_bytes),
            "idx_bytes_per_reserved_slot": 4,
            "idx_bytes_per_physical_particle": 4,
        },
        "artifacts": {
            "pmwd_vs_idxless_plot": str(pmwd_plot_path),
            "baseline_idx_vs_idxless_plot": str(baseline_plot_path),
            "metrics_json": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved PMWD vs idxless plot to {pmwd_plot_path}")
    print(f"Saved baseline-vs-idxless plot to {baseline_plot_path}")
    print(f"Saved metrics report to {metrics_path}")
    print(
        "PMWD vs idxless density diff:",
        f"max_abs={report['pmwd_vs_idxless']['max_abs_diff']:.6g}",
        f"mean_abs={report['pmwd_vs_idxless']['mean_abs_diff']:.6g}",
        f"rms={report['pmwd_vs_idxless']['rms_diff']:.6g}",
    )
    print(
        "Baseline idx vs idxless density diff:",
        f"max_abs={report['baseline_idx_vs_idxless']['max_abs_diff']:.6g}",
        f"mean_abs={report['baseline_idx_vs_idxless']['mean_abs_diff']:.6g}",
        f"rms={report['baseline_idx_vs_idxless']['rms_diff']:.6g}",
    )
    print(
        "Current clone vs current density diff:",
        f"max_abs={report['current_clone_vs_current']['max_abs_diff']:.6g}",
        f"mean_abs={report['current_clone_vs_current']['mean_abs_diff']:.6g}",
        f"rms={report['current_clone_vs_current']['rms_diff']:.6g}",
    )
    print(
        "Idx memory savings:",
        f"lpt={idx_lpt_bytes} bytes ({mib(idx_lpt_bytes):.3f} MiB),",
        f"final={idx_final_bytes} bytes ({mib(idx_final_bytes):.3f} MiB)",
    )


if __name__ == "__main__":
    main()
