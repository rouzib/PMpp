#!/usr/bin/env python
"""Compare notebook-style real-input nbody forward values and gradients against PMWD."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PM
from pmwd.lpt import lpt as lpt_pmwd
from pmwd.modes import linear_modes as linear_modes_pmwd
from pmwd.modes import white_noise as white_noise_pmwd
from pmwd.nbody import nbody as nbody_pmwd
from pmwd.scatter import scatter as scatter_pmwd

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import SimpleLCDM as SimpleLCDM_PP
from src.lpt import lpt as lpt_pmpp
from src.modes import linear_modes as linear_modes_pmpp
from src.modes import white_noise as white_noise_pmpp
from src.nbody import nbody as nbody_pmpp
from src.scatter import scatter as scatter_pmpp
from src.utils import create_compute_mesh, pmid_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--num-ptcl", type=int, default=4)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-seed", type=int, default=0)
    parser.add_argument("--a-start", type=float, default=1 / 60)
    parser.add_argument("--a-stop", type=float, default=1 / 15)
    parser.add_argument("--a-nbody-maxstep", type=float, default=1 / 60)
    parser.add_argument("--max-ptcl-factor", type=float, default=2.5)
    parser.add_argument("--max-share-ptcl", type=int, default=4000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=8000)
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


def to_numpy(array) -> np.ndarray:
    return np.asarray(jax.device_get(array))


def init_confs(args: argparse.Namespace, gpu_devices: list[jax.Device]) -> tuple[Configuration, ConfigurationPMWD]:
    ptcl_grid_shape = (args.num_ptcl,) * 3
    ptcl_spacing = args.box_size / ptcl_grid_shape[0]
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=args.mesh_shape,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(args.num_ptcl**3 / len(gpu_devices) * args.max_ptcl_factor),
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
        a_start=args.a_start,
        a_stop=args.a_stop,
        a_nbody_maxstep=args.a_nbody_maxstep,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_stop=conf_pmpp.a_stop,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def notebook_target_density(seed: int, conf_pmwd):
    cosmo = boltzmann_pmwd(SimpleLCDM_PM(conf_pmwd), conf_pmwd)
    modes = white_noise_pmwd(seed, conf_pmwd)
    modes = linear_modes_pmwd(modes, cosmo, conf_pmwd)
    ptcl, _ = lpt_pmwd(modes, cosmo, conf_pmwd)
    ptcl, _ = nbody_pmwd(ptcl, None, cosmo, conf_pmwd)
    return scatter_pmwd(ptcl, conf_pmwd)


def pmwd_forward(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmwd(base_cosmo, conf)
    modes = linear_modes_pmwd(modes_real, cosmo, conf)
    ptcl, _ = lpt_pmwd(modes, cosmo, conf)
    ptcl, _ = nbody_pmwd(ptcl, None, cosmo, conf)
    dens = scatter_pmwd(ptcl, conf)
    return ptcl, dens


def pmpp_forward(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmpp(base_cosmo, conf)
    modes = linear_modes_pmpp(modes_real, cosmo, conf)
    ptcl = lpt_pmpp(modes, cosmo, conf.replace(max_share_ptcl=conf.max_share_ptcl * 4))
    ptcl = nbody_pmpp(ptcl, cosmo, conf)
    dens = scatter_pmpp(ptcl, conf)
    return ptcl, dens


def first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp) -> np.ndarray:
    particle_keys = np.asarray(jax.device_get(pmid_to_idx(ptcl_pmwd.pmid, conf_pmwd)))
    slot_keys = np.asarray(jax.device_get(pmid_to_idx(ptcl_pmpp.pmid, conf_pmpp, ptcl_pmpp.unused_index)))

    first_slot = np.full(particle_keys.shape[0], -1, dtype=np.int32)
    key_to_particle = {int(key): pid for pid, key in enumerate(particle_keys)}

    for slot, key in enumerate(slot_keys):
        if key < 0:
            continue
        pid = key_to_particle.get(int(key))
        if pid is not None and first_slot[pid] < 0:
            first_slot[pid] = slot

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise RuntimeError(f"Missing PMPP slots for particle ids: {missing[:10].tolist()}")

    return first_slot


def vector_metrics(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float) -> tuple[dict[str, float], np.ndarray]:
    diff = candidate - reference
    abs_diff = np.abs(diff)
    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "allclose": bool(np.allclose(candidate, reference, atol=atol, rtol=rtol)),
    }
    return metrics, diff


def cosmo_grad_metrics(cosmo_grad_pmwd, cosmo_grad_pmpp) -> dict[str, dict[str, float | bool]]:
    report = {}
    for field_name in ("A_s_1e9", "n_s", "Omega_m", "Omega_b", "h"):
        ref = np.asarray(jax.device_get(getattr(cosmo_grad_pmwd, field_name)))
        cand = np.asarray(jax.device_get(getattr(cosmo_grad_pmpp, field_name)))
        same_nan_mask = bool(np.array_equal(np.isnan(ref), np.isnan(cand)))
        finite = np.isfinite(ref) & np.isfinite(cand)
        entry = {"same_nan_mask": same_nan_mask}
        if finite.any():
            diff = np.abs(cand[finite] - ref[finite])
            entry.update(
                {
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "allclose": bool(np.allclose(cand[finite], ref[finite], atol=1e-4, rtol=1e-4)),
                }
            )
        else:
            entry.update(
                {
                    "max_abs_diff": 0.0,
                    "mean_abs_diff": 0.0,
                    "allclose": True,
                }
            )
        entry["pmwd_all_nan"] = bool(np.isnan(ref).all())
        entry["pmpp_all_nan"] = bool(np.isnan(cand).all())
        report[field_name] = entry
    return report


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
        "num_devices": int(conf.num_devices),
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


def projection_image(array: np.ndarray, projection: str) -> tuple[np.ndarray, str, str]:
    if projection == "x":
        return array.sum(axis=0), "y", "z"
    if projection == "y":
        return array.sum(axis=1).T, "z", "x"
    if projection == "z":
        return array.sum(axis=2).T, "y", "x"
    raise ValueError(f"Unsupported projection {projection}")


def projection_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for projection in ("x", "y", "z"):
        ref_proj, _, _ = projection_image(reference, projection)
        cand_proj, _, _ = projection_image(candidate, projection)
        diff = cand_proj - ref_proj
        report[projection] = {
            "sum_reference": float(ref_proj.sum()),
            "sum_candidate": float(cand_proj.sum()),
            "max_abs_diff": float(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "rms_diff": float(np.sqrt(np.mean(diff**2))),
        }
    return report


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


def save_particle_plot(
    disp_pmwd: np.ndarray,
    disp_pmpp: np.ndarray,
    vel_pmwd: np.ndarray,
    vel_pmpp: np.ndarray,
    acc_pmwd: np.ndarray,
    acc_pmpp: np.ndarray,
    density_metrics: dict[str, float],
    output_path: Path,
) -> None:
    def _scatter(ax, ref, cand, title, color):
        lim = float(max(np.max(np.abs(ref)), np.max(np.abs(cand)), 1e-14))
        ax.scatter(ref, cand, s=10, alpha=0.22, linewidths=0, color=color)
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title)
        ax.set_xlabel("PMWD")
        ax.set_ylabel("PMPP")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    _scatter(axes[0, 0], disp_pmwd.ravel(), disp_pmpp.ravel(), "disp parity", "#1f77b4")
    _scatter(axes[0, 1], vel_pmwd.ravel(), vel_pmpp.ravel(), "vel parity", "#d55e00")
    _scatter(axes[0, 2], acc_pmwd.ravel(), acc_pmpp.ravel(), "acc parity", "#009e73")

    axes[1, 0].hist(np.linalg.norm(disp_pmpp - disp_pmwd, axis=1), bins=30, log=True, color="#1f77b4", alpha=0.85)
    axes[1, 0].set_title("disp residual norms")
    axes[1, 1].hist(np.linalg.norm(vel_pmpp - vel_pmwd, axis=1), bins=30, log=True, color="#d55e00", alpha=0.85)
    axes[1, 1].set_title("vel residual norms")
    axes[1, 2].hist(np.linalg.norm(acc_pmpp - acc_pmwd, axis=1), bins=30, log=True, color="#009e73", alpha=0.85)
    axes[1, 2].set_title("acc residual norms")
    for ax in axes[1]:
        ax.set_xlabel("||delta||")
        ax.set_ylabel("count")

    axes[1, 2].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"density max |delta| = {density_metrics['max_abs_diff']:.3e}",
                f"density mean |delta| = {density_metrics['mean_abs_diff']:.3e}",
                f"density allclose = {density_metrics['allclose']}",
            ]
        ),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3},
    )

    fig.suptitle("N-body particle-state parity for the notebook objective path", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_density_plot(
    dens_pmwd: np.ndarray,
    dens_pmpp: np.ndarray,
    metrics: dict[str, float],
    gpu_layout: dict,
    output_path: Path,
) -> None:
    projections = []
    for projection in ("x", "y", "z"):
        ref_proj, x_label, y_label = projection_image(dens_pmwd, projection)
        cand_proj, _, _ = projection_image(dens_pmpp, projection)
        diff_proj = cand_proj - ref_proj
        projections.append((projection, ref_proj, cand_proj, diff_proj, x_label, y_label))

    log_arrays = [
        np.log10(np.clip(array, 1e-8, None))
        for _, ref_proj, cand_proj, _, _, _ in projections
        for array in (ref_proj, cand_proj)
    ]
    log_vmin = min(float(array.min()) for array in log_arrays)
    log_vmax = max(float(array.max()) for array in log_arrays)
    diff_vmax = max(float(np.abs(diff).max()) for _, _, _, diff, _, _ in projections)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-12

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    log_im = None
    diff_im = None

    for row, (projection, ref_proj, cand_proj, diff_proj, x_label, y_label) in enumerate(projections):
        ref_log = np.log10(np.clip(ref_proj, 1e-8, None))
        cand_log = np.log10(np.clip(cand_proj, 1e-8, None))

        log_im = axes[row, 0].imshow(
            ref_log,
            origin="lower",
            cmap="viridis",
            vmin=log_vmin,
            vmax=log_vmax,
            aspect="auto",
        )
        axes[row, 0].set_title("PMWD")

        axes[row, 1].imshow(
            cand_log,
            origin="lower",
            cmap="viridis",
            vmin=log_vmin,
            vmax=log_vmax,
            aspect="auto",
        )
        axes[row, 1].set_title("PMPP")

        diff_im = axes[row, 2].imshow(
            diff_proj,
            origin="lower",
            cmap="coolwarm",
            vmin=-diff_vmax,
            vmax=diff_vmax,
            aspect="auto",
        )
        axes[row, 2].set_title("PMPP - PMWD")

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

    axes[0, 2].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"max |delta| = {metrics['max_abs_diff']:.3e}",
                f"mean |delta| = {metrics['mean_abs_diff']:.3e}",
                f"rms |delta| = {metrics['rms_diff']:.3e}",
                f"allclose = {metrics['allclose']}",
            ]
        ),
        transform=axes[0, 2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
    )

    fig.colorbar(log_im, ax=axes[:, :2], shrink=0.86, label="log10 projected density")
    fig.colorbar(diff_im, ax=axes[:, 2], shrink=0.86, label="projected density residual")
    fig.suptitle(
        "Final density parity after LPT + nbody + scatter\n"
        "Background shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gradient_plot(
    grad_pmwd: np.ndarray,
    grad_pmpp: np.ndarray,
    metrics: dict[str, float],
    gpu_layout: dict,
    output_path: Path,
) -> None:
    projections = []
    for projection in ("x", "y", "z"):
        ref_proj, x_label, y_label = projection_image(grad_pmwd, projection)
        cand_proj, _, _ = projection_image(grad_pmpp, projection)
        diff_proj = cand_proj - ref_proj
        projections.append((projection, ref_proj, cand_proj, diff_proj, x_label, y_label))

    signal_vmax = max(
        float(np.max(np.abs(array)))
        for _, ref_proj, cand_proj, _, _, _ in projections
        for array in (ref_proj, cand_proj)
    )
    signal_vmax = signal_vmax if signal_vmax > 0 else 1e-12
    diff_vmax = max(float(np.max(np.abs(diff_proj))) for _, _, _, diff_proj, _, _ in projections)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-12

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    signal_im = None
    diff_im = None

    for row, (projection, ref_proj, cand_proj, diff_proj, x_label, y_label) in enumerate(projections):
        signal_im = axes[row, 0].imshow(
            ref_proj,
            origin="lower",
            cmap="RdBu_r",
            vmin=-signal_vmax,
            vmax=signal_vmax,
            aspect="auto",
        )
        axes[row, 0].set_title("PMWD")

        axes[row, 1].imshow(
            cand_proj,
            origin="lower",
            cmap="RdBu_r",
            vmin=-signal_vmax,
            vmax=signal_vmax,
            aspect="auto",
        )
        axes[row, 1].set_title("PMPP")

        diff_im = axes[row, 2].imshow(
            diff_proj,
            origin="lower",
            cmap="RdBu_r",
            vmin=-diff_vmax,
            vmax=diff_vmax,
            aspect="auto",
        )
        axes[row, 2].set_title("PMPP - PMWD")

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

    axes[0, 2].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"max |delta| = {metrics['max_abs_diff']:.3e}",
                f"mean |delta| = {metrics['mean_abs_diff']:.3e}",
                f"rms |delta| = {metrics['rms_diff']:.3e}",
                f"allclose = {metrics['allclose']}",
            ]
        ),
        transform=axes[0, 2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
    )

    fig.colorbar(signal_im, ax=axes[:, :2], shrink=0.86, label="projected mode gradient")
    fig.colorbar(diff_im, ax=axes[:, 2], shrink=0.86, label="projected gradient residual")
    fig.suptitle(
        "Notebook-style mode-gradient parity through nbody\n"
        "Background shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    particle_plot_path = args.output_dir / "nbody_particle_forward_parity.png"
    density_plot_path = args.output_dir / "nbody_density_forward_parity.png"
    gradient_plot_path = args.output_dir / "nbody_mode_gradient_parity.png"
    metrics_path = args.output_dir / "nbody_gradient_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf_pmpp, conf_pmwd = init_confs(args, gpu_devices)

    target_dens = notebook_target_density(args.target_seed, conf_pmwd)
    base_cosmo_pmwd = SimpleLCDM_PM(conf_pmwd)
    base_cosmo_pmpp = SimpleLCDM_PP(conf_pmpp)
    modes_real_pmwd = white_noise_pmwd(args.seed, conf_pmwd, real=True)
    modes_real_pmpp = white_noise_pmpp(args.seed, conf_pmpp, real=True)

    start = perf_counter()
    pmwd_forward_jit = jax.jit(pmwd_forward, static_argnames=("conf",))
    pmpp_forward_jit = jax.jit(pmpp_forward, static_argnames=("conf",))
    ptcl_pmwd, dens_pmwd = pmwd_forward_jit(modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
    ptcl_pmpp, dens_pmpp = pmpp_forward_jit(modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)
    forward_elapsed = perf_counter() - start

    first_slot = first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp)

    disp_pmwd = to_numpy(ptcl_pmwd.disp)
    vel_pmwd = to_numpy(ptcl_pmwd.vel)
    acc_pmwd = to_numpy(ptcl_pmwd.acc)
    disp_pmpp = to_numpy(ptcl_pmpp.disp)[first_slot]
    vel_pmpp = to_numpy(ptcl_pmpp.vel)[first_slot]
    acc_pmpp = to_numpy(ptcl_pmpp.acc)[first_slot]
    dens_pmwd_np = to_numpy(dens_pmwd)
    dens_pmpp_np = to_numpy(dens_pmpp)

    disp_metrics, _ = vector_metrics(disp_pmwd, disp_pmpp, atol=5e-3, rtol=1e-4)
    vel_metrics, _ = vector_metrics(vel_pmwd, vel_pmpp, atol=1e-3, rtol=1e-4)
    acc_metrics, _ = vector_metrics(acc_pmwd, acc_pmpp, atol=1e-3, rtol=1e-4)
    density_metrics, _ = vector_metrics(dens_pmwd_np, dens_pmpp_np, atol=1e-3, rtol=1e-4)
    gpu_layout = gpu_layout_from_conf(conf_pmpp)
    density_projection_metrics = projection_metrics(dens_pmwd_np, dens_pmpp_np)

    def loss_pmwd(tgt_dens, modes_real, cosmo, conf):
        dens = pmwd_forward(modes_real, cosmo, conf)[1]
        return (dens - tgt_dens).var()

    def loss_pmpp(tgt_dens, modes_real, cosmo, conf):
        dens = pmpp_forward(modes_real, cosmo, conf)[1]
        return (dens - tgt_dens).var()

    start = perf_counter()
    grad_pmwd_fn = jax.jit(jax.grad(loss_pmwd, argnums=(1, 2)), static_argnames=("conf",))
    grad_pmpp_fn = jax.jit(jax.grad(loss_pmpp, argnums=(1, 2)), static_argnames=("conf",))
    grad_modes_pmwd, grad_cosmo_pmwd = grad_pmwd_fn(target_dens, modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
    grad_modes_pmpp, grad_cosmo_pmpp = grad_pmpp_fn(target_dens, modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)
    grad_elapsed = perf_counter() - start

    grad_modes_pmwd_np = to_numpy(grad_modes_pmwd)
    grad_modes_pmpp_np = to_numpy(grad_modes_pmpp)
    mode_grad_metrics, _ = vector_metrics(grad_modes_pmwd_np, grad_modes_pmpp_np, atol=1e-4, rtol=1e-4)
    mode_gradient_projection_metrics = projection_metrics(grad_modes_pmwd_np, grad_modes_pmpp_np)
    cosmo_metrics = cosmo_grad_metrics(grad_cosmo_pmwd, grad_cosmo_pmpp)

    save_particle_plot(
        disp_pmwd,
        disp_pmpp,
        vel_pmwd,
        vel_pmpp,
        acc_pmwd,
        acc_pmpp,
        density_metrics,
        particle_plot_path,
    )
    save_density_plot(dens_pmwd_np, dens_pmpp_np, density_metrics, gpu_layout, density_plot_path)
    save_gradient_plot(grad_modes_pmwd_np, grad_modes_pmpp_np, mode_grad_metrics, gpu_layout, gradient_plot_path)

    report = {
        "config": {
            "box_size": args.box_size,
            "num_ptcl": args.num_ptcl,
            "seed": args.seed,
            "target_seed": args.target_seed,
            "a_start": args.a_start,
            "a_stop": args.a_stop,
            "a_nbody_maxstep": args.a_nbody_maxstep,
            "num_devices": args.num_devices,
        },
        "runtime_seconds": {
            "forward": forward_elapsed,
            "mode_gradient": grad_elapsed,
        },
        "disp_metrics": disp_metrics,
        "vel_metrics": vel_metrics,
        "acc_metrics": acc_metrics,
        "density_metrics": density_metrics,
        "density_projection_metrics": density_projection_metrics,
        "mode_gradient_metrics": mode_grad_metrics,
        "mode_gradient_projection_metrics": mode_gradient_projection_metrics,
        "cosmo_gradient_metrics": cosmo_metrics,
        "gpu_layout": gpu_layout,
        "artifacts": {
            "particle_plot": str(particle_plot_path),
            "density_plot": str(density_plot_path),
            "gradient_plot": str(gradient_plot_path),
            "metrics_json": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved particle plot to {particle_plot_path}")
    print(f"Saved density plot to {density_plot_path}")
    print(f"Saved gradient plot to {gradient_plot_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(report["mode_gradient_metrics"], indent=2))


if __name__ == "__main__":
    main()
