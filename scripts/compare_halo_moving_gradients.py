#!/usr/bin/env python
"""Compare the halo-move helper VJP against the true local VJP."""

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

from src.particles import Particles
from src.steps import _halo_move_vjp
from tests.test_utils import init_conf as _init_test_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--num-ptcl", type=int, default=8)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--max-ptcl-factor", type=float, default=2.5)
    parser.add_argument("--max-share-ptcl", type=int, default=32)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=128)
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


def init_conf(args: argparse.Namespace, gpu_devices: list[jax.Device]):
    return _init_test_conf(
        num_ptcl=args.num_ptcl,
        mesh_shape=args.mesh_shape,
        box_size=args.box_size,
        num_devices=len(gpu_devices),
        max_ptcl_per_slice=args.max_ptcl_factor,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
    )


def to_numpy(array) -> np.ndarray:
    return np.asarray(jax.device_get(array))


def build_probe_state(conf):
    key = jax.random.PRNGKey(0)
    ptcl = Particles.gen_grid(conf, vel=True, acc=True)

    key_vel, key_acc, key_cot = jax.random.split(key, 3)
    vel = jax.random.normal(key_vel, ptcl.vel.shape, dtype=ptcl.vel.dtype) * 0.2
    acc = jax.random.normal(key_acc, ptcl.acc.shape, dtype=ptcl.acc.dtype) * 0.2

    pmid = ptcl.pmid.reshape(conf.num_devices, conf.max_ptcl_per_slice, 3)
    disp = ptcl.disp.reshape(conf.num_devices, conf.max_ptcl_per_slice, 3)
    unused = ptcl.unused_index.reshape(conf.num_devices, conf.max_ptcl_per_slice)
    slot_ids = jnp.arange(conf.max_ptcl_per_slice)[None, :]

    x = pmid[..., 0]
    move_right = (x == 2) & (~unused) & (slot_ids % 17 == 0)
    move_left = (x == 5) & (~unused) & (slot_ids % 19 == 0)
    crossing_mask = (move_right | move_left).reshape(ptcl.unused_index.shape)

    shift_x = move_right.astype(disp.dtype) * (1.25 * conf.cell_size)
    shift_x -= move_left.astype(disp.dtype) * (1.15 * conf.cell_size)
    shift = jnp.zeros_like(disp).at[..., 0].set(shift_x)
    disp = (disp + shift).reshape(ptcl.disp.shape)

    cot_disp = jax.random.normal(key_cot, ptcl.disp.shape, dtype=ptcl.disp.dtype)
    cot_vel = jax.random.normal(key_vel, ptcl.vel.shape, dtype=ptcl.vel.dtype)
    cot_acc = jax.random.normal(key_acc, ptcl.acc.shape, dtype=ptcl.acc.dtype)

    return ptcl, disp, vel, acc, crossing_mask, cot_disp, cot_vel, cot_acc


def compare_fields(reference: np.ndarray, probe: np.ndarray) -> dict[str, float]:
    diff = probe - reference
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
    }


def gpu_layout_from_conf(conf) -> dict:
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
            "x slabs projected out",
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

    for halo_x in gpu_layout["halo_cell_bands"]:
        draw_wrapped_span(ax, float(halo_x), float(halo_x + 1), width, color="#ff6f61", alpha=0.14, lw=0)


def residual_projection_image(mesh_pos: np.ndarray, residual: np.ndarray, projection: str, bins: int, nmesh: int):
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


def save_slot_count_plot(count_path: Path, conf, x_before, x_after, crossing_mask, valid_before, valid_after):
    bins = np.arange(conf.nMesh + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    before_counts, _ = np.histogram(x_before[valid_before], bins=bins)
    after_counts, _ = np.histogram(x_after[valid_after], bins=bins)
    crossing_counts, _ = np.histogram(x_before[valid_before & crossing_mask], bins=bins)

    axes[0].bar(bins[:-1] - 0.15, before_counts, width=0.3, label="before", color="#7fb3d5")
    axes[0].bar(bins[:-1] + 0.15, after_counts, width=0.3, label="after", color="#ff8c42")
    axes[0].set_title("Valid Slot Counts By x Cell")
    axes[0].set_xlabel("x cell")
    axes[0].set_ylabel("slot count")
    axes[0].legend(frameon=False)

    axes[1].bar(bins[:-1], crossing_counts, width=0.8, color="#d1495b")
    axes[1].set_title("Forced Crossing Input Slots")
    axes[1].set_xlabel("x cell")
    axes[1].set_ylabel("slot count")

    for ax in axes:
        ax.set_xticks(np.arange(conf.nMesh))
        for halo_x in gpu_layout_from_conf(conf)["halo_cell_bands"]:
            ax.axvspan(halo_x, halo_x + 1, color="#ff6f61", alpha=0.12, lw=0)

    fig.savefig(count_path, dpi=180)
    plt.close(fig)


def save_parity_plot(path: Path, reference, helper, valid_mask, crossing_mask, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    field_names = ["disp", "vel", "acc"]
    colors = {
        "other": "#4e79a7",
        "crossing": "#d1495b",
    }

    for col, name in enumerate(field_names):
        ref = reference[name][valid_mask].reshape(-1)
        probe_helper = helper[name][valid_mask].reshape(-1)
        crossing = np.repeat(crossing_mask[valid_mask], 3)

        lo = min(ref.min(), probe_helper.min())
        hi = max(ref.max(), probe_helper.max())

        ax = axes[col]
        ax.scatter(ref[~crossing], probe_helper[~crossing], s=6, alpha=0.30, color=colors["other"], linewidths=0)
        ax.scatter(ref[crossing], probe_helper[crossing], s=10, alpha=0.65, color=colors["crossing"], linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.8)
        ax.set_title(f"Helper VJP: {name}")
        ax.set_xlabel("true VJP")
        ax.set_ylabel("helper")
        ax.text(
            0.03,
            0.94,
            f"max |d| = {metrics[name]['max_abs_diff']:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.80, "pad": 2},
        )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_residual_plot(path: Path, mesh_pos, valid_mask, residual_norm, conf):
    gpu_layout = gpu_layout_from_conf(conf)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)

    for ax, projection in zip(axes, ("x", "y", "z")):
        image, xlabel, ylabel = residual_projection_image(
            mesh_pos[valid_mask],
            residual_norm[valid_mask],
            projection,
            bins=conf.nMesh,
            nmesh=conf.nMesh,
        )
        im = ax.imshow(
            image,
            origin="lower",
            extent=(0, conf.nMesh, 0, conf.nMesh),
            cmap="magma",
            aspect="equal",
        )
        decorate_gpu_layout(ax, projection, gpu_layout)
        ax.set_title(f"Helper Residual Projection: {projection}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf = init_conf(args, gpu_devices)

    ptcl, disp, vel, acc, crossing_mask, cot_disp, cot_vel, cot_acc = build_probe_state(conf)
    def outputs_only(disp_in, vel_in, acc_in):
        _, disp_out, vel_out, acc_out, *_ = conf.mGPU_halo_moving(
            ptcl.pmid,
            ptcl.disp,
            disp_in,
            vel_in,
            acc_in,
            conf.halo_start,
            conf.halo_end,
            ptcl.unused_index,
        )
        return disp_out, vel_out, acc_out

    forward = conf.mGPU_halo_moving(
        ptcl.pmid,
        ptcl.disp,
        disp,
        vel,
        acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.unused_index,
    )
    pmid_out, disp_out, vel_out, acc_out, halo_out, unused_out, forward_failed, max_ptcl_moved = forward

    _, vjp_fn = jax.vjp(outputs_only, disp, vel, acc)
    vjp_disp, vjp_vel, vjp_acc = vjp_fn((cot_disp, cot_vel, cot_acc))

    fixed_disp, fixed_vel, fixed_acc = _halo_move_vjp(
        ptcl,
        disp,
        vel,
        acc,
        cot_disp,
        cot_vel,
        cot_acc,
        conf.num_devices == 2,
        conf,
    )

    valid_before = ~to_numpy(ptcl.unused_index)
    valid_after = ~to_numpy(unused_out)
    crossing_mask_np = to_numpy(crossing_mask)

    mesh_pos_before = (to_numpy(ptcl.pmid).astype(np.float64) + to_numpy(disp).astype(np.float64) * conf.disp_size) % float(conf.nMesh)
    mesh_pos_after = (to_numpy(pmid_out).astype(np.float64) + to_numpy(disp_out).astype(np.float64) * conf.disp_size) % float(conf.nMesh)

    reference = {
        "disp": to_numpy(vjp_disp),
        "vel": to_numpy(vjp_vel),
        "acc": to_numpy(vjp_acc),
    }
    helper_np = {
        "disp": to_numpy(fixed_disp),
        "vel": to_numpy(fixed_vel),
        "acc": to_numpy(fixed_acc),
    }

    helper_metrics = {name: compare_fields(reference[name], helper_np[name]) for name in reference}

    residual_norm = np.sqrt(
        np.sum((helper_np["disp"] - reference["disp"]) ** 2, axis=1)
        + np.sum((helper_np["vel"] - reference["vel"]) ** 2, axis=1)
        + np.sum((helper_np["acc"] - reference["acc"]) ** 2, axis=1)
    )

    save_slot_count_plot(
        output_dir / "halo_moving_slot_counts.png",
        conf,
        mesh_pos_before[:, 0],
        mesh_pos_after[:, 0],
        crossing_mask_np,
        valid_before,
        valid_after,
    )
    save_parity_plot(
        output_dir / "halo_moving_helper_grad_parity.png",
        reference,
        helper_np,
        valid_before,
        crossing_mask_np,
        helper_metrics,
    )
    save_residual_plot(
        output_dir / "halo_moving_helper_residual_projections.png",
        mesh_pos_before,
        valid_before,
        residual_norm,
        conf,
    )

    metrics = {
        "config": {
            "num_ptcl": int(args.num_ptcl),
            "mesh_shape": int(args.mesh_shape),
            "num_devices": int(args.num_devices),
            "max_ptcl_per_slice": int(conf.max_ptcl_per_slice),
            "max_share_ptcl": int(conf.max_share_ptcl),
            "max_share_gather_ptcl": int(conf.max_share_gather_ptcl),
        },
        "forward": {
            "forward_failed": bool(np.asarray(jax.device_get(forward_failed))),
            "max_ptcl_moved": int(np.asarray(jax.device_get(max_ptcl_moved))),
            "valid_slots_before": int(valid_before.sum()),
            "valid_slots_after": int(valid_after.sum()),
            "forced_crossing_slots": int((crossing_mask_np & valid_before).sum()),
        },
        "helper_vs_true": helper_metrics,
        "helper_combined_residual": {
            "max_norm_diff": float(np.max(residual_norm)),
            "mean_norm_diff": float(np.mean(residual_norm)),
        },
        "runtime_seconds": perf_counter() - start,
    }

    with (output_dir / "halo_moving_metrics.json").open("w", encoding="ascii") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
