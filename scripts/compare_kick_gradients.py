#!/usr/bin/env python
"""Compare PMPP kick forward/adjoint parity against PMWD."""

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
from jax.tree_util import tree_leaves, tree_map

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PMWD
from pmwd.nbody import kick as kick_pmwd, kick_adj as kick_adj_pmwd
from pmwd.particles import Particles as ParticlesPMWD

from pmpp.boltzmann import boltzmann as boltzmann_pmpp
from pmpp.cosmo import SimpleLCDM as SimpleLCDM_PMPP
from pmpp.particles import Particles
from pmpp.steps import kick as kick_pmpp, kick_adj as kick_adj_pmpp

from tests.test_utils import init_conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--num-ptcl", type=int, default=16)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--max-ptcl-factor", type=float, default=1.25)
    parser.add_argument("--max-share-ptcl", type=int, default=32)
    parser.add_argument("--max-share-gather-ptcl", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/tests/output"),
    )
    return parser.parse_args()


def to_numpy(array) -> np.ndarray:
    return np.asarray(jax.device_get(array))


def build_state(args: argparse.Namespace):
    conf = init_conf(
        num_ptcl=args.num_ptcl,
        mesh_shape=args.mesh_shape,
        box_size=args.box_size,
        num_devices=args.num_devices,
        max_ptcl_per_slice=args.max_ptcl_factor,
        max_share_ptcl=args.max_share_ptcl,
        max_share_gather_ptcl=args.max_share_gather_ptcl,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf.ptcl_spacing,
        ptcl_grid_shape=conf.ptcl_grid_shape,
        mesh_shape=conf.mesh_shape,
        a_start=conf.a_start,
        a_nbody_maxstep=conf.a_nbody_maxstep,
    )
    cosmo_pmwd = boltzmann_pmwd(SimpleLCDM_PMWD(conf_pmwd), conf_pmwd)
    cosmo_pmpp = boltzmann_pmpp(SimpleLCDM_PMPP(conf), conf)

    ptcl_pmwd = ParticlesPMWD.gen_grid(conf_pmwd, vel=True, acc=True)
    key = jax.random.PRNGKey(0)
    key_disp, key_vel, key_acc, key_cot = jax.random.split(key, 4)
    ptcl_pmwd = ptcl_pmwd.replace(
        disp=jax.random.uniform(
            key_disp,
            ptcl_pmwd.disp.shape,
            minval=-0.25 * conf.cell_size,
            maxval=0.25 * conf.cell_size,
        ).astype(conf.float_dtype),
        vel=(jax.random.normal(key_vel, ptcl_pmwd.vel.shape) * 0.15).astype(conf.float_dtype),
        acc=(jax.random.normal(key_acc, ptcl_pmwd.acc.shape) * 0.2).astype(conf.float_dtype),
    )
    ptcl_pmpp = Particles.from_ptcl(ptcl_pmwd, conf)
    return conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd, key_cot


def slot_mapping(ptcl_pmwd, conf):
    pid_payload = jnp.repeat(jnp.arange(conf.ptcl_num, dtype=jnp.int32)[:, None], 3, axis=1)

    pid_slots = []
    unused_slots = []
    for i in range(conf.num_devices):
        gpu_id = conf.devices_index[i]
        _, _, pid_vel, _, unused_index, _ = Particles.distribute_ptcl_pos(
            ptcl_pmwd.pmid,
            ptcl_pmwd.disp,
            pid_payload,
            None,
            conf,
            gpu_id,
        )
        pid_slots.append(np.asarray(pid_vel[:, 0]))
        unused_slots.append(np.asarray(unused_index))

    pid_slots = np.concatenate(pid_slots, axis=0)
    unused_slots = np.concatenate(unused_slots, axis=0)
    valid_slots = ~unused_slots

    first_slot = np.full(conf.ptcl_num, -1, dtype=np.int32)
    for slot, pid in enumerate(pid_slots):
        if valid_slots[slot] and first_slot[pid] < 0:
            first_slot[pid] = slot

    missing = np.flatnonzero(first_slot < 0)
    if missing.size:
        raise RuntimeError(f"Missing particle ids: {missing[:10].tolist()}")

    return pid_slots, valid_slots, first_slot


def reduce_input_slots(slot_values, pid_slots, valid_slots, conf):
    reduced = np.zeros((conf.ptcl_num, slot_values.shape[-1]), dtype=np.float64)
    for slot, pid in enumerate(pid_slots):
        if valid_slots[slot]:
            reduced[pid] += slot_values[slot]
    return reduced


def tree_leaf_metrics(ref_tree, got_tree):
    diffs = {}
    max_abs = 0.0
    mean_abs = 0.0
    leaves = 0
    for idx, (ref, got) in enumerate(zip(tree_leaves(ref_tree), tree_leaves(got_tree))):
        if ref is None or got is None:
            continue
        ref_np = to_numpy(ref)
        got_np = to_numpy(got)
        if ref_np.dtype.kind not in "fc":
            continue
        diff = np.abs(got_np - ref_np)
        leaf_max = float(diff.max(initial=0.0))
        leaf_mean = float(diff.mean())
        diffs[f"leaf_{idx}"] = {"max_abs_diff": leaf_max, "mean_abs_diff": leaf_mean}
        max_abs = max(max_abs, leaf_max)
        mean_abs += leaf_mean
        leaves += 1
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": 0.0 if leaves == 0 else mean_abs / leaves,
        "leaf_count": leaves,
        "leaves": diffs,
    }


def field_metrics(ref: np.ndarray, got: np.ndarray):
    diff = got - ref
    abs_err = np.abs(diff)
    return {
        "max_abs_diff": float(abs_err.max()),
        "mean_abs_diff": float(abs_err.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
    }, diff


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


def save_parity_plot(path: Path, forward_ref, forward_got, adj_ref, adj_got, cosmo_metrics):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), constrained_layout=True)
    panels = [
        ("Forward disp", forward_ref["disp"], forward_got["disp"]),
        ("Forward vel", forward_ref["vel"], forward_got["vel"]),
        ("Adjoint disp", adj_ref["disp"], adj_got["disp"]),
        ("Adjoint vel", adj_ref["vel"], adj_got["vel"]),
        ("Adjoint acc", adj_ref["acc"], adj_got["acc"]),
    ]

    for ax, (title, ref, got) in zip(axes.flat[:5], panels):
        ref_flat = ref.reshape(-1)
        got_flat = got.reshape(-1)
        lo = min(ref_flat.min(), got_flat.min())
        hi = max(ref_flat.max(), got_flat.max())
        diff = np.abs(got_flat - ref_flat)
        ax.scatter(ref_flat, got_flat, s=5, alpha=0.35, color="#457b9d", linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("PMWD")
        ax.set_ylabel("PMPP")
        ax.text(
            0.03,
            0.94,
            f"max |d| = {diff.max():.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.80, "pad": 2},
        )

    ax = axes.flat[5]
    ax.axis("off")
    ax.text(0.0, 0.88, "Cosmology Cotangent", fontsize=11, fontweight="bold")
    ax.text(0.0, 0.68, f"max |d| = {cosmo_metrics['max_abs_diff']:.2e}", fontsize=10)
    ax.text(0.0, 0.52, f"mean |d| = {cosmo_metrics['mean_abs_diff']:.2e}", fontsize=10)
    ax.text(0.0, 0.36, f"leaf count = {cosmo_metrics['leaf_count']}", fontsize=10)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_residual_plot(path: Path, mesh_pos, residual_norm, conf):
    gpu_layout = gpu_layout_from_conf(conf)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)

    for ax, projection in zip(axes, ("x", "y", "z")):
        image, xlabel, ylabel = residual_projection_image(
            mesh_pos,
            residual_norm,
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
        ax.set_title(f"Adjoint Residual Projection: {projection}")
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
    conf, conf_pmwd, cosmo_pmpp, cosmo_pmwd, ptcl_pmpp, ptcl_pmwd, key_cot = build_state(args)
    pid_slots, valid_slots, first_slot = slot_mapping(ptcl_pmwd, conf)
    first_slot_j = jnp.asarray(first_slot)

    a_acc = conf.a_start
    a_prev = conf.a_start
    a_next = conf.a_start * 1.4

    out_pmwd = kick_pmwd(a_acc, a_prev, a_next, ptcl_pmwd, cosmo_pmwd, conf_pmwd)
    out_pmpp = kick_pmpp(a_acc, a_prev, a_next, ptcl_pmpp, cosmo_pmpp, conf)

    forward_ref = {
        "disp": to_numpy(out_pmwd.disp),
        "vel": to_numpy(out_pmwd.vel),
        "acc": to_numpy(out_pmwd.acc),
    }
    forward_got = {
        "disp": to_numpy(out_pmpp.disp)[first_slot],
        "vel": to_numpy(out_pmpp.vel)[first_slot],
        "acc": to_numpy(out_pmpp.acc)[first_slot],
    }

    key_disp_cot, key_vel_cot, key_acc_cot = jax.random.split(key_cot, 3)
    cot_disp_unique = jax.random.normal(key_disp_cot, out_pmwd.disp.shape, dtype=out_pmwd.disp.dtype)
    cot_vel_unique = jax.random.normal(key_vel_cot, out_pmwd.vel.shape, dtype=out_pmwd.vel.dtype)
    cot_acc_unique = jax.random.normal(key_acc_cot, out_pmwd.acc.shape, dtype=out_pmwd.acc.dtype)
    ptcl_cot_pmwd = out_pmwd.replace(disp=cot_disp_unique, vel=cot_vel_unique, acc=cot_acc_unique)
    ptcl_cot_pmpp = out_pmpp.replace(
        disp=jnp.zeros(out_pmpp.disp.shape, dtype=out_pmpp.disp.dtype).at[first_slot_j].set(cot_disp_unique),
        vel=jnp.zeros(out_pmpp.vel.shape, dtype=out_pmpp.vel.dtype).at[first_slot_j].set(cot_vel_unique),
        acc=jnp.zeros(out_pmpp.acc.shape, dtype=out_pmpp.acc.dtype).at[first_slot_j].set(cot_acc_unique),
    )

    zero_cosmo_pmwd = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmwd)
    zero_cosmo_pmpp = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmpp)
    zero_force_pmwd = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmwd)
    zero_force_pmpp = tree_map(lambda x: jnp.zeros_like(x) if x is not None else None, cosmo_pmpp)

    _, in_cot_pmwd, cosmo_cot_pmwd = kick_adj_pmwd(
        a_acc, a_prev, a_next, ptcl_pmwd, ptcl_cot_pmwd, cosmo_pmwd, zero_cosmo_pmwd, zero_force_pmwd, conf_pmwd
    )
    _, in_cot_pmpp, cosmo_cot_pmpp = kick_adj_pmpp(
        a_acc, a_prev, a_next, ptcl_pmpp, ptcl_cot_pmpp, cosmo_pmpp, zero_cosmo_pmpp, zero_force_pmpp, conf
    )

    adj_ref = {
        "disp": to_numpy(in_cot_pmwd.disp),
        "vel": to_numpy(in_cot_pmwd.vel),
        "acc": to_numpy(in_cot_pmwd.acc),
    }
    adj_got = {
        "disp": reduce_input_slots(to_numpy(in_cot_pmpp.disp), pid_slots, valid_slots, conf),
        "vel": reduce_input_slots(to_numpy(in_cot_pmpp.vel), pid_slots, valid_slots, conf),
        "acc": reduce_input_slots(to_numpy(in_cot_pmpp.acc), pid_slots, valid_slots, conf),
    }

    forward_metrics = {}
    for name in ("disp", "vel", "acc"):
        forward_metrics[name], _ = field_metrics(forward_ref[name], forward_got[name])
    adjoint_metrics = {}
    adjoint_diffs = {}
    for name in adj_ref:
        adjoint_metrics[name], adjoint_diffs[name] = field_metrics(adj_ref[name], adj_got[name])
    cosmo_metrics = tree_leaf_metrics(cosmo_cot_pmwd, cosmo_cot_pmpp)

    mesh_pos = (to_numpy(ptcl_pmwd.pmid).astype(np.float64) + to_numpy(ptcl_pmwd.disp).astype(np.float64) * conf.disp_size) % float(conf.nMesh)
    residual_norm = np.sqrt(
        np.sum(adjoint_diffs["disp"] ** 2, axis=1)
        + np.sum(adjoint_diffs["vel"] ** 2, axis=1)
        + np.sum(adjoint_diffs["acc"] ** 2, axis=1)
    )

    save_parity_plot(
        output_dir / "kick_forward_and_adj_parity.png",
        forward_ref,
        forward_got,
        adj_ref,
        adj_got,
        cosmo_metrics,
    )
    save_residual_plot(
        output_dir / "kick_adj_residual_projections.png",
        mesh_pos,
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
        "forward": forward_metrics,
        "adjoint": adjoint_metrics,
        "cosmo_cot": cosmo_metrics,
        "runtime_seconds": perf_counter() - start,
    }

    with (output_dir / "kick_metrics.json").open("w", encoding="ascii") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
