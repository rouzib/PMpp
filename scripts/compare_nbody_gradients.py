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
    parser.add_argument("--num-ptcl", type=int, default=128)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-seed", type=int, default=0)
    parser.add_argument("--a-start", type=float, default=1 / 64)
    parser.add_argument("--a-stop", type=float, default=1)
    parser.add_argument("--a-nbody-maxstep", type=float, default=1 / 64)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.4)
    parser.add_argument("--max-share-ptcl", type=int, default=40000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=120000)
    parser.add_argument("--projection", choices=("all", "x", "y", "z"), default="all")
    parser.add_argument("--gpu-layout-panels", choices=("all", "pmpp", "none"), default="all")
    parser.add_argument("--forward-plot-layout", choices=("full", "pair"), default="full")
    parser.add_argument("--forward-plot-themes", nargs="*", choices=("light", "dark"), default=[])
    parser.add_argument("--forward-transparent", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--gradient-plot-layout", choices=("full", "pair"), default="full")
    parser.add_argument("--gradient-plot-themes", nargs="*", choices=("light", "dark"), default=[])
    parser.add_argument("--gradient-transparent", action="store_true")
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


def pmpp_forward_adjoint(modes_real, base_cosmo, conf):
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


def selected_projections(projection: str) -> tuple[str, ...]:
    if projection == "all":
        return ("x", "y", "z")
    return (projection,)


def should_draw_gpu_layout(panel: str, gpu_layout_panels: str) -> bool:
    if gpu_layout_panels == "none":
        return False
    if gpu_layout_panels == "all":
        return True
    return panel == "pmpp"


def artifact_path(output_dir: Path, stem: str, projection: str) -> Path:
    suffix = "" if projection == "all" else f"_{projection}"
    return output_dir / f"{stem}{suffix}.png"


def themed_artifact_path(output_dir: Path, stem: str, projection: str, theme: str) -> Path:
    suffix = "" if projection == "all" else f"_{projection}"
    return output_dir / f"{stem}{suffix}_{theme}.png"


MINIMAL_PLOT_THEMES = {
    "light": {
        "text_color": "#111111",
        "spine_color": "#111111",
        "tick_color": "#111111",
    },
    "dark": {
        "text_color": "#f4f4f4",
        "spine_color": "#f4f4f4",
        "tick_color": "#f4f4f4",
    },
}


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
            alpha=0.16,
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
        ax.axvspan(halo_x - 0.5, halo_x + 0.5, color="#ff6f61", alpha=0.20, lw=0)


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
    projection: str,
    gpu_layout_panels: str,
) -> None:
    projections = []
    for projection_name in selected_projections(projection):
        ref_proj, x_label, y_label = projection_image(dens_pmwd, projection_name)
        cand_proj, _, _ = projection_image(dens_pmpp, projection_name)
        diff_proj = cand_proj - ref_proj
        projections.append((projection_name, ref_proj, cand_proj, diff_proj, x_label, y_label))

    log_arrays = [
        np.log10(np.clip(array, 1e-8, None))
        for _, ref_proj, cand_proj, _, _, _ in projections
        for array in (ref_proj, cand_proj)
    ]
    log_vmin = min(float(array.min()) for array in log_arrays)
    log_vmax = max(float(array.max()) for array in log_arrays)
    diff_vmax = max(float(np.abs(diff).max()) for _, _, _, diff, _, _ in projections)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-12

    fig, axes = plt.subplots(
        len(projections),
        3,
        figsize=(13, max(4.4 * len(projections), 4.8)),
        constrained_layout=True,
        squeeze=False,
    )
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

        for col, panel in enumerate(("pmwd", "pmpp", "diff")):
            axes[row, col].set_ylabel(y_label if col == 0 else "")
            axes[row, col].set_xlabel(x_label if row == len(projections) - 1 else "")
            if should_draw_gpu_layout(panel, gpu_layout_panels):
                decorate_gpu_layout(axes[row, col], projection, gpu_layout)
            if row != len(projections) - 1:
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
    layout_caption = (
        "PM++ panel shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands"
        if gpu_layout_panels == "pmpp"
        else "Background shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands"
    )
    fig.suptitle(
        "Final density parity after LPT + nbody + scatter\n"
        f"{layout_caption}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_density_pair_plot(
    dens_pmwd: np.ndarray,
    dens_pmpp: np.ndarray,
    gpu_layout: dict,
    output_path: Path,
    projection: str,
    gpu_layout_panels: str,
    theme: str,
    transparent: bool,
) -> None:
    projections = []
    for projection_name in selected_projections(projection):
        ref_proj, _, _ = projection_image(dens_pmwd, projection_name)
        cand_proj, _, _ = projection_image(dens_pmpp, projection_name)
        diff_proj = cand_proj - ref_proj
        projections.append(
            (
                projection_name,
                np.log10(np.clip(ref_proj, 1e-8, None)),
                np.log10(np.clip(cand_proj, 1e-8, None)),
                diff_proj,
            )
        )

    style = MINIMAL_PLOT_THEMES[theme]
    log_vmin = min(
        float(array.min())
        for _, ref_proj, cand_proj, _ in projections
        for array in (ref_proj, cand_proj)
    )
    log_vmax = max(
        float(array.max())
        for _, ref_proj, cand_proj, _ in projections
        for array in (ref_proj, cand_proj)
    )
    diff_vmax = max(float(np.abs(diff_proj).max()) for _, _, _, diff_proj in projections)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-12

    fig, axes = plt.subplots(
        len(projections),
        3,
        figsize=(13.4, max(4.2 * len(projections), 4.2)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    image_artist = None
    diff_artist = None

    for row, (projection_name, ref_proj, cand_proj, diff_proj) in enumerate(projections):
        image_artist = axes[row, 0].imshow(
            ref_proj,
            origin="lower",
            cmap="viridis",
            vmin=log_vmin,
            vmax=log_vmax,
            aspect="auto",
        )
        axes[row, 1].imshow(
            cand_proj,
            origin="lower",
            cmap="viridis",
            vmin=log_vmin,
            vmax=log_vmax,
            aspect="auto",
        )
        diff_artist = axes[row, 2].imshow(
            diff_proj,
            origin="lower",
            cmap="coolwarm",
            vmin=-diff_vmax,
            vmax=diff_vmax,
            aspect="auto",
        )

        axes[row, 0].set_title("PMWD", color=style["text_color"], fontsize=16, pad=10)
        axes[row, 1].set_title("PM++", color=style["text_color"], fontsize=16, pad=10)
        axes[row, 2].set_title("Residual", color=style["text_color"], fontsize=16, pad=10)

        if should_draw_gpu_layout("pmpp", gpu_layout_panels):
            decorate_gpu_layout(axes[row, 1], projection_name, gpu_layout)
        if should_draw_gpu_layout("pmwd", gpu_layout_panels):
            decorate_gpu_layout(axes[row, 0], projection_name, gpu_layout)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_alpha(0.0 if transparent else 1.0)
            for spine in ax.spines.values():
                spine.set_edgecolor(style["spine_color"])
                spine.set_linewidth(0.8)

    cbar_signal = fig.colorbar(
        image_artist,
        ax=axes[:, :2],
        shrink=0.92,
        fraction=0.046,
        pad=0.03,
    )
    cbar_signal.outline.set_edgecolor(style["spine_color"])
    cbar_signal.ax.tick_params(colors=style["tick_color"], labelsize=11)
    for label in cbar_signal.ax.get_yticklabels():
        label.set_color(style["tick_color"])
    cbar_signal.set_label("", color=style["text_color"])

    cbar_diff = fig.colorbar(
        diff_artist,
        ax=axes[:, 2],
        shrink=0.92,
        fraction=0.046,
        pad=0.03,
    )
    cbar_diff.outline.set_edgecolor(style["spine_color"])
    cbar_diff.ax.tick_params(colors=style["tick_color"], labelsize=11)
    for label in cbar_diff.ax.get_yticklabels():
        label.set_color(style["tick_color"])
    cbar_diff.set_label("", color=style["text_color"])

    fig.savefig(output_path, dpi=220, transparent=transparent, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_gradient_pair_plot(
    grad_pmwd: np.ndarray,
    grad_pmpp: np.ndarray,
    gpu_layout: dict,
    output_path: Path,
    projection: str,
    theme: str,
    transparent: bool,
) -> None:
    projections = []
    for projection_name in selected_projections(projection):
        ref_proj, _, _ = projection_image(grad_pmwd, projection_name)
        cand_proj, _, _ = projection_image(grad_pmpp, projection_name)
        projections.append((projection_name, ref_proj, cand_proj))

    style = MINIMAL_PLOT_THEMES[theme]
    signal_vmax = max(
        float(np.max(np.abs(array)))
        for _, ref_proj, cand_proj in projections
        for array in (ref_proj, cand_proj)
    )
    signal_vmax = signal_vmax if signal_vmax > 0 else 1e-12

    fig, axes = plt.subplots(
        len(projections),
        2,
        figsize=(9.6, max(4.2 * len(projections), 4.2)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    image_artist = None

    for row, (projection_name, ref_proj, cand_proj) in enumerate(projections):
        image_artist = axes[row, 0].imshow(
            ref_proj,
            origin="lower",
            cmap="RdBu_r",
            vmin=-signal_vmax,
            vmax=signal_vmax,
            aspect="auto",
        )
        axes[row, 1].imshow(
            cand_proj,
            origin="lower",
            cmap="RdBu_r",
            vmin=-signal_vmax,
            vmax=signal_vmax,
            aspect="auto",
        )

        axes[row, 0].set_title("PMWD", color=style["text_color"], fontsize=16, pad=10)
        axes[row, 1].set_title("PM++", color=style["text_color"], fontsize=16, pad=10)

        decorate_gpu_layout(axes[row, 1], projection_name, gpu_layout)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_alpha(0.0 if transparent else 1.0)
            for spine in ax.spines.values():
                spine.set_edgecolor(style["spine_color"])
                spine.set_linewidth(0.8)

    cbar = fig.colorbar(image_artist, ax=axes.ravel().tolist(), shrink=0.92, fraction=0.046, pad=0.03)
    cbar.outline.set_edgecolor(style["spine_color"])
    cbar.ax.tick_params(colors=style["tick_color"], labelsize=11)
    for label in cbar.ax.get_yticklabels():
        label.set_color(style["tick_color"])
    cbar.set_label("", color=style["text_color"])

    fig.savefig(output_path, dpi=220, transparent=transparent, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_gradient_plot(
    grad_pmwd: np.ndarray,
    grad_pmpp: np.ndarray,
    metrics: dict[str, float],
    gpu_layout: dict,
    output_path: Path,
    projection: str,
    gpu_layout_panels: str,
) -> None:
    projections = []
    for projection_name in selected_projections(projection):
        ref_proj, x_label, y_label = projection_image(grad_pmwd, projection_name)
        cand_proj, _, _ = projection_image(grad_pmpp, projection_name)
        diff_proj = cand_proj - ref_proj
        projections.append((projection_name, ref_proj, cand_proj, diff_proj, x_label, y_label))

    signal_vmax = max(
        float(np.max(np.abs(array)))
        for _, ref_proj, cand_proj, _, _, _ in projections
        for array in (ref_proj, cand_proj)
    )
    signal_vmax = signal_vmax if signal_vmax > 0 else 1e-12
    diff_vmax = max(float(np.max(np.abs(diff_proj))) for _, _, _, diff_proj, _, _ in projections)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-12

    fig, axes = plt.subplots(
        len(projections),
        3,
        figsize=(13, max(4.4 * len(projections), 4.8)),
        constrained_layout=True,
        squeeze=False,
    )
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

        for col, panel in enumerate(("pmwd", "pmpp", "diff")):
            axes[row, col].set_ylabel(y_label if col == 0 else "")
            axes[row, col].set_xlabel(x_label if row == len(projections) - 1 else "")
            if should_draw_gpu_layout(panel, gpu_layout_panels):
                decorate_gpu_layout(axes[row, col], projection, gpu_layout)
            if row != len(projections) - 1:
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
    layout_caption = (
        "PM++ panel shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands"
        if gpu_layout_panels == "pmpp"
        else "Background shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands"
    )
    fig.suptitle(
        "Notebook-style mode-gradient parity through nbody\n"
        f"{layout_caption}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.forward_plot_layout == "pair" and not args.forward_plot_themes:
        args.forward_plot_themes = ["light", "dark"]
    if args.gradient_plot_layout == "pair" and not args.gradient_plot_themes:
        args.gradient_plot_themes = ["light", "dark"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pmpp_forward_impl = pmpp_forward_adjoint

    particle_plot_path = args.output_dir / "nbody_particle_forward_parity.png"
    density_plot_path = artifact_path(args.output_dir, "nbody_density_forward_parity", args.projection)
    density_pair_plot_paths = {
        theme: themed_artifact_path(args.output_dir, "nbody_density_forward_pair", args.projection, theme)
        for theme in args.forward_plot_themes
    }
    gradient_plot_path = artifact_path(args.output_dir, "nbody_mode_gradient_parity", args.projection)
    gradient_pair_plot_paths = {
        theme: themed_artifact_path(args.output_dir, "nbody_mode_gradient_pair", args.projection, theme)
        for theme in args.gradient_plot_themes
    }
    metrics_path = args.output_dir / "nbody_gradient_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf_pmpp, conf_pmwd = init_confs(args, gpu_devices)

    base_cosmo_pmwd = SimpleLCDM_PM(conf_pmwd)
    base_cosmo_pmpp = SimpleLCDM_PP(conf_pmpp)
    modes_real_pmwd = white_noise_pmwd(args.seed, conf_pmwd, real=True)
    modes_real_pmpp = white_noise_pmpp(args.seed, conf_pmpp, real=True)

    start = perf_counter()
    pmwd_forward_jit = jax.jit(pmwd_forward, static_argnames=("conf",))
    pmpp_forward_jit = jax.jit(pmpp_forward_impl, static_argnames=("conf",))
    ptcl_pmwd, dens_pmwd = pmwd_forward_jit(modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
    ptcl_pmpp, dens_pmpp = pmpp_forward_jit(modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)
    forward_elapsed = perf_counter() - start

    dens_pmwd_np = to_numpy(dens_pmwd)
    dens_pmpp_np = to_numpy(dens_pmpp)

    density_metrics, _ = vector_metrics(dens_pmwd_np, dens_pmpp_np, atol=1e-3, rtol=1e-4)
    gpu_layout = gpu_layout_from_conf(conf_pmpp)
    density_projection_metrics = projection_metrics(dens_pmwd_np, dens_pmpp_np)

    if args.forward_plot_layout == "pair":
        for theme, themed_path in density_pair_plot_paths.items():
            save_density_pair_plot(
                dens_pmwd_np,
                dens_pmpp_np,
                gpu_layout,
                themed_path,
                args.projection,
                args.gpu_layout_panels,
                theme,
                args.forward_transparent,
            )
    else:
        save_density_plot(
            dens_pmwd_np,
            dens_pmpp_np,
            density_metrics,
            gpu_layout,
            density_plot_path,
            args.projection,
            args.gpu_layout_panels,
        )

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
            "projection": args.projection,
            "gpu_layout_panels": args.gpu_layout_panels,
            "forward_plot_layout": args.forward_plot_layout,
            "forward_plot_themes": args.forward_plot_themes,
            "forward_transparent": args.forward_transparent,
            "forward_only": args.forward_only,
            "gradient_plot_layout": args.gradient_plot_layout,
            "gradient_plot_themes": args.gradient_plot_themes,
            "gradient_transparent": args.gradient_transparent,
            # "pmpp_nbody_path": args.pmpp_nbody_path,
        },
        "runtime_seconds": {
            "forward": forward_elapsed,
        },
        "density_metrics": density_metrics,
        "density_projection_metrics": density_projection_metrics,
        "gpu_layout": gpu_layout,
        "artifacts": {
            "metrics_json": str(metrics_path),
        },
    }
    if args.forward_plot_layout == "pair":
        report["artifacts"]["density_pair_plots"] = {
            theme: str(path) for theme, path in density_pair_plot_paths.items()
        }
    else:
        report["artifacts"]["density_plot"] = str(density_plot_path)

    if not args.forward_only:
        first_slot = first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp)

        disp_pmwd = to_numpy(ptcl_pmwd.disp)
        vel_pmwd = to_numpy(ptcl_pmwd.vel)
        acc_pmwd = to_numpy(ptcl_pmwd.acc)
        disp_pmpp = to_numpy(ptcl_pmpp.disp)[first_slot]
        vel_pmpp = to_numpy(ptcl_pmpp.vel)[first_slot]
        acc_pmpp = to_numpy(ptcl_pmpp.acc)[first_slot]

        disp_metrics, _ = vector_metrics(disp_pmwd, disp_pmpp, atol=5e-3, rtol=1e-4)
        vel_metrics, _ = vector_metrics(vel_pmwd, vel_pmpp, atol=1e-3, rtol=1e-4)
        acc_metrics, _ = vector_metrics(acc_pmwd, acc_pmpp, atol=1e-3, rtol=1e-4)

        def loss_pmwd(tgt_dens, modes_real, cosmo, conf):
            dens = pmwd_forward(modes_real, cosmo, conf)[1]
            return (dens - tgt_dens).var()

        def loss_pmpp(tgt_dens, modes_real, cosmo, conf):
            dens = pmpp_forward_impl(modes_real, cosmo, conf)[1]
            return (dens - tgt_dens).var()

        target_dens = notebook_target_density(args.target_seed, conf_pmwd)
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
        if args.gradient_plot_layout == "pair":
            for theme, themed_path in gradient_pair_plot_paths.items():
                save_gradient_pair_plot(
                    grad_modes_pmwd_np,
                    grad_modes_pmpp_np,
                    gpu_layout,
                    themed_path,
                    args.projection,
                    theme,
                    args.gradient_transparent,
                )
        else:
            save_gradient_plot(
                grad_modes_pmwd_np,
                grad_modes_pmpp_np,
                mode_grad_metrics,
                gpu_layout,
                gradient_plot_path,
                args.projection,
                args.gpu_layout_panels,
            )

        report["runtime_seconds"]["mode_gradient"] = grad_elapsed
        report["disp_metrics"] = disp_metrics
        report["vel_metrics"] = vel_metrics
        report["acc_metrics"] = acc_metrics
        report["mode_gradient_metrics"] = mode_grad_metrics
        report["mode_gradient_projection_metrics"] = mode_gradient_projection_metrics
        report["cosmo_gradient_metrics"] = cosmo_metrics
        report["artifacts"]["particle_plot"] = str(particle_plot_path)
        if args.gradient_plot_layout == "pair":
            report["artifacts"]["gradient_pair_plots"] = {
                theme: str(path) for theme, path in gradient_pair_plot_paths.items()
            }
        else:
            report["artifacts"]["gradient_plot"] = str(gradient_plot_path)

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.forward_plot_layout == "pair":
        for theme, path in density_pair_plot_paths.items():
            print(f"Saved density pair plot ({theme}) to {path}")
    else:
        print(f"Saved density plot to {density_plot_path}")
    if not args.forward_only:
        print(f"Saved particle plot to {particle_plot_path}")
        if args.gradient_plot_layout == "pair":
            for theme, path in gradient_pair_plot_paths.items():
                print(f"Saved gradient pair plot ({theme}) to {path}")
        else:
            print(f"Saved gradient plot to {gradient_plot_path}")
    print(f"Saved metrics to {metrics_path}")
    if args.forward_only:
        print(json.dumps(report["density_metrics"], indent=2))
    else:
        print(json.dumps(report["mode_gradient_metrics"], indent=2))


if __name__ == "__main__":
    main()
