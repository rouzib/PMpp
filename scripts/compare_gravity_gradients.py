#!/usr/bin/env python
"""Compare PMPP gravity forward values and gradients against PMWD."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    gravity as gravity_pmwd,
    linear_modes as linear_modes_pmwd,
    lpt as lpt_pmwd,
    nbody as nbody_pmwd,
    white_noise as white_noise_pmwd,
)

from src.configuration import Configuration
from src.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from src.gravity import gravity as gravity_pmpp
from src.particles import Particles
from src.utils import create_compute_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-ptcl", type=int, default=16)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.25)
    parser.add_argument("--max-share-ptcl", type=int, default=11000)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=35000)
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


def init_conf(args: argparse.Namespace, gpu_devices: list[jax.Device]) -> Configuration:
    ptcl_grid_shape = (args.num_ptcl,) * 3
    ptcl_spacing = args.box_size / ptcl_grid_shape[0]
    compute_mesh = create_compute_mesh(gpu_devices)
    max_ptcl_per_slice = int((args.num_ptcl**3 / len(gpu_devices)) * args.max_ptcl_factor)

    return Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=args.mesh_shape,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
        to_save_z=[1, 2 / 3, 1 / 3, 0],
        a_start=1 / 60,
        a_nbody_maxstep=1 / 60,
    )


def init_pmwd_state(conf: Configuration, args: argparse.Namespace):
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf.ptcl_spacing,
        ptcl_grid_shape=conf.ptcl_grid_shape,
        mesh_shape=conf.mesh_shape,
        a_start=conf.a_start,
        a_nbody_maxstep=conf.a_nbody_maxstep,
    )
    cosmo_pmwd = SimpleLCDM_PMWD(conf_pmwd)
    cosmo_pmwd = boltzmann_pmwd(cosmo_pmwd, conf_pmwd)

    modes = white_noise_pmwd(args.seed, conf_pmwd)
    modes = linear_modes_pmwd(modes, cosmo_pmwd, conf_pmwd)
    ptcl_lpt_pmwd, _ = lpt_pmwd(modes, cosmo_pmwd, conf_pmwd)
    ptcl_final_pmwd, _ = nbody_pmwd(ptcl_lpt_pmwd, None, cosmo_pmwd, conf_pmwd)
    return conf_pmwd, cosmo_pmwd, ptcl_final_pmwd


def distribute_particle_ids(ptcl_pmwd, conf: Configuration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pid_payload = jnp.repeat(jnp.arange(conf.ptcl_num, dtype=jnp.int32)[:, None], 3, axis=1)

    pid_slots = []
    unused_slots = []
    halo_slots = []
    for i in range(conf.num_devices):
        gpu_id = conf.devices_index[i]
        _, _, pid_vel, _, unused_index, halo_mask = Particles.distribute_ptcl_pos(
            ptcl_pmwd.pmid,
            ptcl_pmwd.disp,
            pid_payload,
            None,
            conf,
            gpu_id,
        )
        pid_slots.append(np.asarray(pid_vel[:, 0]))
        unused_slots.append(np.asarray(unused_index))
        halo_slots.append(np.asarray(halo_mask))

    return (
        np.concatenate(pid_slots, axis=0),
        np.concatenate(unused_slots, axis=0),
        np.concatenate(halo_slots, axis=0),
    )


def first_slot_mapping(pid_slots: np.ndarray, unused_slots: np.ndarray, ptcl_num: int) -> np.ndarray:
    valid = ~unused_slots
    first_slot = np.full(ptcl_num, -1, dtype=np.int32)
    for slot, pid in enumerate(pid_slots):
        if valid[slot] and first_slot[pid] < 0:
            first_slot[pid] = slot

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise RuntimeError(f"Missing PMPP slots for particle ids: {missing[:10].tolist()}")
    return first_slot


def duplicate_slot_metrics(
    values_pmpp_slots: np.ndarray,
    pid_slots: np.ndarray,
    unused_slots: np.ndarray,
) -> dict[str, float | int]:
    valid = ~unused_slots
    unique_ids, counts = np.unique(pid_slots[valid], return_counts=True)
    dup_ids = unique_ids[counts > 1]

    max_abs = 0.0
    mean_abs = 0.0
    samples = 0
    for pid in dup_ids:
        slots = np.flatnonzero(valid & (pid_slots == pid))
        ref = values_pmpp_slots[slots[0]]
        delta = np.abs(values_pmpp_slots[slots[1:]] - ref)
        if delta.size:
            max_abs = max(max_abs, float(delta.max()))
            mean_abs += float(delta.mean())
            samples += 1

    return {
        "duplicated_particle_count": int(dup_ids.size),
        "max_duplicate_count": int(counts.max(initial=0)),
        "max_abs_duplicate_diff": max_abs,
        "mean_abs_duplicate_diff": 0.0 if samples == 0 else mean_abs / samples,
    }


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    diff = candidate - reference
    abs_err = np.abs(diff)
    rel_err = abs_err / np.maximum(np.abs(reference), 1e-12)
    abs_norm = np.linalg.norm(diff, axis=1)
    rel_norm = abs_norm / np.maximum(np.linalg.norm(reference, axis=1), 1e-12)

    metrics = {
        "max_abs_diff": float(abs_err.max()),
        "mean_abs_diff": float(abs_err.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "max_rel_diff": float(rel_err.max()),
        "mean_rel_diff": float(rel_err.mean()),
        "max_abs_norm_diff": float(abs_norm.max()),
        "mean_abs_norm_diff": float(abs_norm.mean()),
        "max_rel_norm_diff": float(rel_norm.max()),
        "mean_rel_norm_diff": float(rel_norm.mean()),
        "allclose_atol_1e-5_rtol_1e-4": bool(np.allclose(candidate, reference, atol=1e-5, rtol=1e-4)),
    }
    return metrics, diff, abs_norm


def current_mesh_position(ptcl, conf: Configuration) -> np.ndarray:
    pmid = to_numpy(ptcl.pmid).astype(np.float64)
    disp = to_numpy(ptcl.disp).astype(np.float64)
    cell_size = float(np.asarray(conf.cell_size))
    return (pmid + disp / cell_size) % float(conf.nMesh)


def gpu_layout_from_conf(conf: Configuration) -> dict:
    offsets = [int(x) for x in np.asarray(conf.offsets)]
    slab_width = int(conf.local_mesh_shape[0])
    halo_start = np.asarray(conf.halo_start)
    halo_end = np.asarray(conf.halo_end)

    owned = []
    for gpu, start in enumerate(offsets):
        owned.append({"gpu": gpu, "start": start, "width": slab_width})

    halo_bands = sorted(
        {
            int(halo_start[gpu, 0] % conf.nMesh)
            for gpu in range(conf.num_devices)
        }
        | {
            int(halo_end[gpu, 0] % conf.nMesh)
            for gpu in range(conf.num_devices)
        }
    )

    return {
        "nmesh": int(conf.nMesh),
        "owned_x_slabs": owned,
        "halo_cell_bands": halo_bands,
    }


def draw_wrapped_span(ax, start: float, end: float, width: float, **kwargs) -> None:
    if end <= width:
        ax.axvspan(start, end, **kwargs)
        return
    ax.axvspan(start, width, **kwargs)
    ax.axvspan(0.0, end - width, **kwargs)


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

    width = float(gpu_layout["nmesh"])
    colors = ["#f4d06f", "#7fb3d5", "#d2b4de", "#82e0aa"]
    for slab in gpu_layout["owned_x_slabs"]:
        color = colors[slab["gpu"] % len(colors)]
        start = float(slab["start"])
        end = start + float(slab["width"])
        draw_wrapped_span(ax, start, end, width, color=color, alpha=0.10, lw=0)
        ax.axvline(start, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            (start + slab["width"] / 2.0) % width,
            0.98,
            f"GPU {slab['gpu']}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )

    for halo_x in gpu_layout["halo_cell_bands"]:
        draw_wrapped_span(ax, float(halo_x), float(halo_x + 1), width, color="#ff6f61", alpha=0.14, lw=0)


def residual_projection_image(
    mesh_pos: np.ndarray,
    residual: np.ndarray,
    projection: str,
    bins: int,
    nmesh: int,
) -> tuple[np.ndarray, str, str]:
    if projection == "x":
        xcoord, ycoord, xlabel, ylabel = mesh_pos[:, 1], mesh_pos[:, 2], "y", "z"
    elif projection == "y":
        xcoord, ycoord, xlabel, ylabel = mesh_pos[:, 0], mesh_pos[:, 2], "x", "z"
    elif projection == "z":
        xcoord, ycoord, xlabel, ylabel = mesh_pos[:, 0], mesh_pos[:, 1], "x", "y"
    else:
        raise ValueError(f"Unsupported projection {projection}")

    edges = np.linspace(0.0, float(nmesh), bins + 1)
    weighted, _, _ = np.histogram2d(xcoord, ycoord, bins=[edges, edges], weights=residual)
    counts, _, _ = np.histogram2d(xcoord, ycoord, bins=[edges, edges])
    image = weighted / np.maximum(counts, 1.0)
    return image.T, xlabel, ylabel


def save_parity_plot(
    reference: np.ndarray,
    candidate: np.ndarray,
    abs_norm: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
    title: str,
    xlabel_prefix: str,
    ylabel_prefix: str,
    extra_lines: list[str] | None = None,
) -> None:
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for comp, ax in enumerate(flat_axes):
        x = reference[:, comp]
        y = candidate[:, comp]
        lim = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-12))
        ax.scatter(x, y, s=5, alpha=0.18, linewidths=0, color="#1f77b4")
        ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{xlabel_prefix} {labels[comp]}")
        ax.set_ylabel(f"{ylabel_prefix} {labels[comp]}")
        ax.set_title(f"{labels[comp]} parity")

    ax = axes[1, 1]
    ax.hist(abs_norm, bins=60, color="#d55e00", alpha=0.85, log=True)
    ax.set_xlabel("|delta| per particle")
    ax.set_ylabel("count")
    ax.set_title("Residual norm distribution")
    text_lines = [
        f"max |delta| = {metrics['max_abs_diff']:.3e}",
        f"mean |delta| = {metrics['mean_abs_diff']:.3e}",
        f"rms delta = {metrics['rms_diff']:.3e}",
        f"max ||delta|| = {metrics['max_abs_norm_diff']:.3e}",
        f"allclose = {metrics['allclose_atol_1e-5_rtol_1e-4']}",
    ]
    if extra_lines:
        text_lines.extend([""] + extra_lines)
    ax.text(
        0.03,
        0.97,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3},
    )

    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_projection_plot(
    mesh_pos: np.ndarray,
    abs_norm: np.ndarray,
    gpu_layout: dict,
    nmesh: int,
    output_path: Path,
    title: str,
) -> None:
    bins = max(32, 2 * nmesh)
    projections = [
        (projection, *residual_projection_image(mesh_pos, abs_norm, projection, bins, nmesh))
        for projection in ("x", "y", "z")
    ]
    images = [np.log10(np.clip(image, 1e-16, None)) for _, image, _, _ in projections]
    vmin = min(float(image.min()) for image in images)
    vmax = max(float(image.max()) for image in images)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    image_artist = None
    for ax, (projection, image, xlabel, ylabel), image_log in zip(axes, projections, images):
        image_artist = ax.imshow(
            image_log,
            origin="lower",
            extent=(0, nmesh, 0, nmesh),
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Project {projection}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        decorate_gpu_layout(ax, projection, gpu_layout)

    fig.colorbar(image_artist, ax=axes, shrink=0.88, label="log10 mean |delta| per bin")
    fig.suptitle(
        title + "\nBackground shading = owned x-slab per GPU, red bands = 1-cell particle halo exchange bands",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forward_plot_path = args.output_dir / "gravity_forward_parity.png"
    grad_plot_path = args.output_dir / "gravity_disp_grad_parity.png"
    projection_plot_path = args.output_dir / "gravity_disp_grad_residual_projections.png"
    metrics_path = args.output_dir / "gravity_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf = init_conf(args, gpu_devices)
    conf_pmwd, cosmo_pmwd, ptcl_final_pmwd = init_pmwd_state(conf, args)
    cosmo_pmpp = SimpleLCDM_PMPP(conf)
    ptcl_final_pmpp = Particles.from_ptcl(ptcl_final_pmwd, conf)

    pid_slots, unused_slots, halo_slots = distribute_particle_ids(ptcl_final_pmwd, conf)
    first_slot = first_slot_mapping(pid_slots, unused_slots, conf.ptcl_num)
    first_slot_j = jnp.array(first_slot)

    start = perf_counter()
    acc_pmwd = to_numpy(gravity_pmwd(conf.a_start, ptcl_final_pmwd, cosmo_pmwd, conf_pmwd))
    acc_pmpp_slots = to_numpy(gravity_pmpp(conf.a_start, ptcl_final_pmpp, cosmo_pmpp, conf))
    acc_pmpp = acc_pmpp_slots[first_slot]
    forward_elapsed = perf_counter() - start

    def loss_gravity_pmwd(disp):
        ptcl = ptcl_final_pmwd.replace(disp=disp)
        acc = gravity_pmwd(conf.a_start, ptcl, cosmo_pmwd, conf_pmwd)
        return 0.5 * jnp.sum(acc**2)

    def loss_gravity_pmpp(disp):
        ptcl = ptcl_final_pmpp.replace(disp=disp)
        acc = gravity_pmpp(conf.a_start, ptcl, cosmo_pmpp, conf)[first_slot_j]
        return 0.5 * jnp.sum(acc**2)

    grad_pmwd_fn = jax.jit(jax.grad(loss_gravity_pmwd))
    grad_pmpp_fn = jax.jit(jax.grad(loss_gravity_pmpp))

    start = perf_counter()
    grad_pmwd = to_numpy(grad_pmwd_fn(ptcl_final_pmwd.disp))
    grad_pmpp_slots = to_numpy(grad_pmpp_fn(ptcl_final_pmpp.disp))
    grad_pmpp = grad_pmpp_slots[first_slot]
    grad_elapsed = perf_counter() - start

    omega_0 = float(np.asarray(cosmo_pmwd.Omega_m))

    def omega_loss_pmwd(omega_m):
        cosmo = cosmo_pmwd.replace(Omega_m=jnp.asarray(omega_m, dtype=cosmo_pmwd.Omega_m.dtype))
        acc = gravity_pmwd(conf.a_start, ptcl_final_pmwd, cosmo, conf_pmwd)
        return 0.5 * jnp.sum(acc**2)

    def omega_loss_pmpp(omega_m):
        cosmo = cosmo_pmpp.replace(Omega_m=jnp.asarray(omega_m, dtype=cosmo_pmpp.Omega_m.dtype))
        acc = gravity_pmpp(conf.a_start, ptcl_final_pmpp, cosmo, conf)[first_slot_j]
        return 0.5 * jnp.sum(acc**2)

    grad_omega_pmwd = float(jax.device_get(jax.jit(jax.grad(omega_loss_pmwd))(omega_0)))
    grad_omega_pmpp = float(jax.device_get(jax.jit(jax.grad(omega_loss_pmpp))(omega_0)))

    forward_metrics, forward_diff, forward_abs_norm = vector_metrics(acc_pmwd, acc_pmpp)
    grad_metrics, grad_diff, grad_abs_norm = vector_metrics(grad_pmwd, grad_pmpp)
    forward_dup_metrics = duplicate_slot_metrics(acc_pmpp_slots, pid_slots, unused_slots)
    grad_dup_metrics = duplicate_slot_metrics(grad_pmpp_slots, pid_slots, unused_slots)
    mesh_pos = current_mesh_position(ptcl_final_pmwd, conf)
    gpu_layout = gpu_layout_from_conf(conf)

    omega_metrics = {
        "pmwd_grad": grad_omega_pmwd,
        "pmpp_grad": grad_omega_pmpp,
        "abs_diff": abs(grad_omega_pmpp - grad_omega_pmwd),
        "rel_diff": abs(grad_omega_pmpp - grad_omega_pmwd) / max(abs(grad_omega_pmwd), 1e-12),
        "allclose_atol_1e-5_rtol_1e-4": bool(np.isclose(grad_omega_pmpp, grad_omega_pmwd, atol=1e-5, rtol=1e-4)),
    }

    save_parity_plot(
        acc_pmwd,
        acc_pmpp,
        forward_abs_norm,
        forward_metrics,
        forward_plot_path,
        "Gravity forward parity: PMWD vs PMPP",
        "PMWD acc",
        "PMPP acc",
        extra_lines=[
            f"duplicate slot max |delta| = {forward_dup_metrics['max_abs_duplicate_diff']:.3e}",
        ],
    )
    save_parity_plot(
        grad_pmwd,
        grad_pmpp,
        grad_abs_norm,
        grad_metrics,
        grad_plot_path,
        "Gravity disp-gradient parity: PMWD vs PMPP",
        "PMWD grad",
        "PMPP grad",
        extra_lines=[
            f"duplicate slot max |delta| = {grad_dup_metrics['max_abs_duplicate_diff']:.3e}",
            f"Omega_m grad |delta| = {omega_metrics['abs_diff']:.3e}",
            f"Omega_m grad allclose = {omega_metrics['allclose_atol_1e-5_rtol_1e-4']}",
        ],
    )
    save_projection_plot(
        mesh_pos,
        grad_abs_norm,
        gpu_layout,
        int(conf.nMesh),
        projection_plot_path,
        "Gravity disp-gradient residuals in current mesh-space coordinates",
    )

    worst_ids = np.argsort(grad_abs_norm)[-10:][::-1]
    report = {
        "config": {
            "box_size": args.box_size,
            "seed": args.seed,
            "num_ptcl": args.num_ptcl,
            "mesh_shape": args.mesh_shape,
            "num_devices": args.num_devices,
            "nmesh": int(conf.nMesh),
            "max_ptcl_per_slice": int(conf.max_ptcl_per_slice),
            "max_share_ptcl": int(conf.max_share_ptcl),
            "max_share_gather_ptcl": int(conf.max_share_gather_ptcl),
        },
        "runtime_seconds": {
            "forward": forward_elapsed,
            "disp_gradient": grad_elapsed,
        },
        "valid_pmpp_slots": int((~unused_slots).sum()),
        "halo_valid_slots": int((halo_slots & ~unused_slots).sum()),
        "duplicate_valid_slots": int((~unused_slots).sum() - np.unique(pid_slots[~unused_slots]).size),
        "forward_metrics": forward_metrics,
        "disp_gradient_metrics": grad_metrics,
        "omega_m_gradient": omega_metrics,
        "forward_duplicate_slot_metrics": forward_dup_metrics,
        "disp_gradient_duplicate_slot_metrics": grad_dup_metrics,
        "gpu_layout": gpu_layout,
        "worst_particles": [
            {
                "particle_id": int(pid),
                "abs_norm_diff": float(grad_abs_norm[pid]),
                "pmwd_forward": acc_pmwd[pid].tolist(),
                "pmpp_forward": acc_pmpp[pid].tolist(),
                "forward_diff": forward_diff[pid].tolist(),
                "pmwd_grad": grad_pmwd[pid].tolist(),
                "pmpp_grad": grad_pmpp[pid].tolist(),
                "grad_diff": grad_diff[pid].tolist(),
                "mesh_pos": mesh_pos[pid].tolist(),
            }
            for pid in worst_ids
        ],
        "artifacts": {
            "forward_plot": str(forward_plot_path),
            "disp_gradient_plot": str(grad_plot_path),
            "projection_plot": str(projection_plot_path),
            "metrics_json": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved forward plot to {forward_plot_path}")
    print(f"Saved gradient plot to {grad_plot_path}")
    print(f"Saved projection plot to {projection_plot_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(report["disp_gradient_metrics"], indent=2))
    print(json.dumps(report["omega_m_gradient"], indent=2))


if __name__ == "__main__":
    main()
