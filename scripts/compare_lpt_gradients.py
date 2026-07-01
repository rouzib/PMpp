#!/usr/bin/env python
"""Compare notebook-style real-input modes/LPT forward values and mode gradients against PMWD."""

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
from pmwd.scatter import scatter as scatter_pmwd

from pmpp.boltzmann import boltzmann as boltzmann_pmpp
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM as SimpleLCDM_PP
from pmpp.lpt import lpt as lpt_pmpp
from pmpp.modes import linear_modes as linear_modes_pmpp
from pmpp.modes import white_noise as white_noise_pmpp
from pmpp.scatter import scatter as scatter_pmpp
from pmpp.utils import create_compute_mesh, pmid_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-seed", type=int, default=1)
    parser.add_argument("--num-ptcl", type=int, default=8)
    parser.add_argument("--mesh-shape", type=int, default=1)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--lpt-order", type=int, default=1)
    parser.add_argument("--max-ptcl-factor", type=float, default=2.2)
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
        a_start=1 / 60,
        a_nbody_maxstep=1 / 60,
        lpt_order=args.lpt_order,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        lpt_order=conf_pmpp.lpt_order,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def forward_pmwd(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmwd(base_cosmo, conf)
    modes = linear_modes_pmwd(modes_real, cosmo, conf)
    ptcl, _ = lpt_pmwd(modes, cosmo, conf)
    dens = scatter_pmwd(ptcl, conf)
    return modes, ptcl, dens


def forward_pmpp(modes_real, base_cosmo, conf):
    cosmo = boltzmann_pmpp(base_cosmo, conf)
    modes = linear_modes_pmpp(modes_real, cosmo, conf)
    ptcl = lpt_pmpp(modes, cosmo, conf.replace(max_share_ptcl=conf.max_share_ptcl * 2))
    dens = scatter_pmpp(ptcl, conf)
    return modes, ptcl, dens


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


def complex_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    diff = candidate - reference
    abs_diff = np.abs(diff)
    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(abs_diff**2))),
        "allclose_atol_1e-8_rtol_1e-6": bool(np.allclose(candidate, reference, atol=1e-8, rtol=1e-6)),
    }
    return metrics, diff


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    diff = candidate - reference
    abs_diff = np.abs(diff)
    abs_norm = np.linalg.norm(diff, axis=1)
    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "max_abs_norm_diff": float(abs_norm.max()),
        "mean_abs_norm_diff": float(abs_norm.mean()),
        "allclose_atol_1e-8_rtol_1e-6": bool(np.allclose(candidate, reference, atol=1e-8, rtol=1e-6)),
    }
    return metrics, diff, abs_norm


def scalar_field_metrics(reference: np.ndarray, candidate: np.ndarray, atol=1e-8, rtol=1e-6) -> tuple[dict[str, float], np.ndarray]:
    diff = candidate - reference
    abs_diff = np.abs(diff)
    metrics = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "allclose": bool(np.allclose(candidate, reference, atol=atol, rtol=rtol)),
    }
    return metrics, diff


def save_modes_plot(reference: np.ndarray, candidate: np.ndarray, metrics: dict[str, float], output_path: Path) -> None:
    ref_real = np.real(reference).ravel()
    cand_real = np.real(candidate).ravel()
    ref_imag = np.imag(reference).ravel()
    cand_imag = np.imag(candidate).ravel()
    abs_resid = np.abs((candidate - reference).ravel())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    lim_real = float(max(np.max(np.abs(ref_real)), np.max(np.abs(cand_real)), 1e-14))
    axes[0].scatter(ref_real, cand_real, s=8, alpha=0.2, linewidths=0, color="#1f77b4")
    axes[0].plot([-lim_real, lim_real], [-lim_real, lim_real], linestyle="--", color="black", linewidth=1.0)
    axes[0].set_xlim(-lim_real, lim_real)
    axes[0].set_ylim(-lim_real, lim_real)
    axes[0].set_title("real parity")
    axes[0].set_xlabel("PMWD")
    axes[0].set_ylabel("PMPP")

    lim_imag = float(max(np.max(np.abs(ref_imag)), np.max(np.abs(cand_imag)), 1e-14))
    axes[1].scatter(ref_imag, cand_imag, s=8, alpha=0.2, linewidths=0, color="#d55e00")
    axes[1].plot([-lim_imag, lim_imag], [-lim_imag, lim_imag], linestyle="--", color="black", linewidth=1.0)
    axes[1].set_xlim(-lim_imag, lim_imag)
    axes[1].set_ylim(-lim_imag, lim_imag)
    axes[1].set_title("imag parity")
    axes[1].set_xlabel("PMWD")
    axes[1].set_ylabel("PMPP")

    axes[2].hist(abs_resid, bins=50, color="#009e73", alpha=0.85, log=True)
    axes[2].set_title("residual histogram")
    axes[2].set_xlabel("|delta modes|")
    axes[2].set_ylabel("count")
    axes[2].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"max |delta| = {metrics['max_abs_diff']:.3e}",
                f"mean |delta| = {metrics['mean_abs_diff']:.3e}",
                f"rms |delta| = {metrics['rms_diff']:.3e}",
                f"allclose = {metrics['allclose_atol_1e-8_rtol_1e-6']}",
            ]
        ),
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3},
    )

    fig.suptitle("Linear modes parity for the notebook real-input path", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_lpt_plot(
    disp_pmwd: np.ndarray,
    disp_pmpp: np.ndarray,
    vel_pmwd: np.ndarray,
    vel_pmpp: np.ndarray,
    density_metrics: dict[str, float],
    disp_metrics: dict[str, float],
    vel_metrics: dict[str, float],
    density_diff: np.ndarray,
    output_path: Path,
) -> None:
    labels = ["x", "y", "z"]
    disp_abs_norm = np.linalg.norm(disp_pmpp - disp_pmwd, axis=1)
    vel_abs_norm = np.linalg.norm(vel_pmpp - vel_pmwd, axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for comp in range(3):
        x = disp_pmwd[:, comp]
        y = disp_pmpp[:, comp]
        lim = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-14))
        axes[0, comp].scatter(x, y, s=8, alpha=0.2, linewidths=0, color="#1f77b4")
        axes[0, comp].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
        axes[0, comp].set_xlim(-lim, lim)
        axes[0, comp].set_ylim(-lim, lim)
        axes[0, comp].set_title(f"disp {labels[comp]} parity")
        axes[0, comp].set_xlabel(f"PMWD {labels[comp]}")
        axes[0, comp].set_ylabel(f"PMPP {labels[comp]}")

        x = vel_pmwd[:, comp]
        y = vel_pmpp[:, comp]
        lim = float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-14))
        axes[1, comp].scatter(x, y, s=8, alpha=0.2, linewidths=0, color="#d55e00")
        axes[1, comp].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
        axes[1, comp].set_xlim(-lim, lim)
        axes[1, comp].set_ylim(-lim, lim)
        axes[1, comp].set_title(f"vel {labels[comp]} parity")
        axes[1, comp].set_xlabel(f"PMWD {labels[comp]}")
        axes[1, comp].set_ylabel(f"PMPP {labels[comp]}")

    axes[0, 3].hist(disp_abs_norm, bins=50, color="#009e73", alpha=0.85, log=True)
    axes[0, 3].set_title("disp residual norms")
    axes[0, 3].set_xlabel("||delta disp||")
    axes[0, 3].set_ylabel("count")
    axes[0, 3].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"max |delta| = {disp_metrics['max_abs_diff']:.3e}",
                f"mean |delta| = {disp_metrics['mean_abs_diff']:.3e}",
                f"max ||delta|| = {disp_metrics['max_abs_norm_diff']:.3e}",
                f"allclose = {disp_metrics['allclose_atol_1e-8_rtol_1e-6']}",
            ]
        ),
        transform=axes[0, 3].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3},
    )

    axes[1, 3].hist(np.abs(density_diff).ravel(), bins=50, color="#7f7f7f", alpha=0.9, log=True)
    axes[1, 3].set_title("vel + density residuals")
    axes[1, 3].set_xlabel("|delta|")
    axes[1, 3].set_ylabel("count")
    axes[1, 3].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"vel max |delta| = {vel_metrics['max_abs_diff']:.3e}",
                f"vel mean |delta| = {vel_metrics['mean_abs_diff']:.3e}",
                f"vel max ||delta|| = {vel_metrics['max_abs_norm_diff']:.3e}",
                f"density max |delta| = {density_metrics['max_abs_diff']:.3e}",
                f"density mean |delta| = {density_metrics['mean_abs_diff']:.3e}",
                f"density allclose = {density_metrics['allclose']}",
            ]
        ),
        transform=axes[1, 3].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3},
    )

    fig.suptitle("LPT particle parity for the notebook real-input path", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gradient_plot(
    grad_pmwd: np.ndarray,
    grad_pmpp: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    ref = grad_pmwd.ravel()
    cand = grad_pmpp.ravel()
    sample = min(ref.size, 20000)
    if sample < ref.size:
        idx = np.linspace(0, ref.size - 1, sample, dtype=np.int64)
        ref_scatter = ref[idx]
        cand_scatter = cand[idx]
    else:
        ref_scatter = ref
        cand_scatter = cand

    ref_proj = grad_pmwd.sum(axis=0)
    cand_proj = grad_pmpp.sum(axis=0)
    resid_proj = np.abs(cand_proj - ref_proj)

    finite = resid_proj[np.isfinite(resid_proj)]
    vmin = float(np.log10(max(finite.min(initial=1e-16), 1e-16)))
    vmax = float(np.log10(max(finite.max(initial=1e-16), 1e-16)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    lim = float(max(np.max(np.abs(ref_scatter)), np.max(np.abs(cand_scatter)), 1e-14))
    axes[0, 0].scatter(ref_scatter, cand_scatter, s=8, alpha=0.2, linewidths=0, color="#1f77b4")
    axes[0, 0].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
    axes[0, 0].set_xlim(-lim, lim)
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].set_title("flattened gradient parity")
    axes[0, 0].set_xlabel("PMWD")
    axes[0, 0].set_ylabel("PMPP")

    vmax_modes = float(max(np.max(np.abs(ref_proj)), np.max(np.abs(cand_proj)), 1e-14))
    axes[0, 1].imshow(ref_proj, cmap="RdBu_r", origin="lower", vmin=-vmax_modes, vmax=vmax_modes)
    axes[0, 1].set_title("PMWD grad sum over axis 0")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("z")

    axes[1, 0].imshow(cand_proj, cmap="RdBu_r", origin="lower", vmin=-vmax_modes, vmax=vmax_modes)
    axes[1, 0].set_title("PMPP grad sum over axis 0")
    axes[1, 0].set_xlabel("y")
    axes[1, 0].set_ylabel("z")

    image = axes[1, 1].imshow(
        np.log10(np.clip(resid_proj, 1e-16, None)),
        cmap="magma",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 1].set_title("log10 residual projection")
    axes[1, 1].set_xlabel("y")
    axes[1, 1].set_ylabel("z")
    axes[1, 1].text(
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
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
    )

    fig.colorbar(image, ax=axes[1, 1], shrink=0.82, label="log10 |delta grad|")
    fig.suptitle("Mode-gradient parity for the notebook LPT loss", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modes_plot_path = args.output_dir / "lpt_modes_forward_parity.png"
    lpt_plot_path = args.output_dir / "lpt_particle_forward_parity.png"
    grad_plot_path = args.output_dir / "lpt_mode_gradient_parity.png"
    metrics_path = args.output_dir / "lpt_gradient_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf_pmpp, conf_pmwd = init_confs(args, gpu_devices)

    base_cosmo_pmpp = SimpleLCDM_PP(conf_pmpp)
    base_cosmo_pmwd = SimpleLCDM_PM(conf_pmwd)

    modes_real_pmwd = white_noise_pmwd(args.seed, conf_pmwd, real=True)
    modes_real_pmpp = white_noise_pmpp(args.seed, conf_pmpp, real=True)
    target_modes_real = white_noise_pmwd(args.target_seed, conf_pmwd, real=True)

    start = perf_counter()
    modes_pmwd, ptcl_pmwd, dens_pmwd = forward_pmwd(modes_real_pmwd, base_cosmo_pmwd, conf_pmwd)
    modes_pmpp, ptcl_pmpp, dens_pmpp = forward_pmpp(modes_real_pmpp, base_cosmo_pmpp, conf_pmpp)
    target_dens = forward_pmwd(target_modes_real, base_cosmo_pmwd, conf_pmwd)[2]
    forward_elapsed = perf_counter() - start

    first_slot = first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp)

    disp_pmwd = to_numpy(ptcl_pmwd.disp)
    vel_pmwd = to_numpy(ptcl_pmwd.vel)
    disp_pmpp = to_numpy(ptcl_pmpp.disp)[first_slot]
    vel_pmpp = to_numpy(ptcl_pmpp.vel)[first_slot]

    modes_metrics, _ = complex_metrics(to_numpy(modes_pmwd), to_numpy(modes_pmpp))
    disp_metrics, _, _ = vector_metrics(disp_pmwd, disp_pmpp)
    vel_metrics, _, _ = vector_metrics(vel_pmwd, vel_pmpp)
    density_metrics, density_diff = scalar_field_metrics(to_numpy(dens_pmwd), to_numpy(dens_pmpp), atol=1e-12, rtol=1e-12)

    def loss_pmwd(modes_real):
        dens = forward_pmwd(modes_real, base_cosmo_pmwd, conf_pmwd)[2]
        return jnp.mean((dens - target_dens) ** 2)

    def loss_pmpp(modes_real):
        dens = forward_pmpp(modes_real, base_cosmo_pmpp, conf_pmpp)[2]
        return jnp.mean((dens - target_dens) ** 2)

    start = perf_counter()
    grad_pmwd = to_numpy(jax.jit(jax.grad(loss_pmwd))(modes_real_pmwd))
    grad_pmpp = to_numpy(jax.jit(jax.grad(loss_pmpp))(modes_real_pmpp))
    grad_elapsed = perf_counter() - start

    gradient_metrics, gradient_diff = scalar_field_metrics(grad_pmwd, grad_pmpp, atol=5e-6, rtol=1e-5)

    save_modes_plot(to_numpy(modes_pmwd), to_numpy(modes_pmpp), modes_metrics, modes_plot_path)
    save_lpt_plot(
        disp_pmwd,
        disp_pmpp,
        vel_pmwd,
        vel_pmpp,
        density_metrics,
        disp_metrics,
        vel_metrics,
        density_diff,
        lpt_plot_path,
    )
    save_gradient_plot(grad_pmwd, grad_pmpp, gradient_metrics, grad_plot_path)

    worst_grad_idx = np.unravel_index(np.argmax(np.abs(gradient_diff)), gradient_diff.shape)
    report = {
        "config": {
            "box_size": args.box_size,
            "seed": args.seed,
            "target_seed": args.target_seed,
            "num_ptcl": args.num_ptcl,
            "mesh_shape": args.mesh_shape,
            "num_devices": args.num_devices,
            "lpt_order": args.lpt_order,
            "max_ptcl_per_slice": int(conf_pmpp.max_ptcl_per_slice),
            "max_share_ptcl": int(conf_pmpp.max_share_ptcl),
            "max_share_gather_ptcl": int(conf_pmpp.max_share_gather_ptcl),
        },
        "runtime_seconds": {
            "forward": forward_elapsed,
            "mode_gradient": grad_elapsed,
        },
        "white_noise_real_match": bool(
            np.allclose(
                to_numpy(modes_real_pmpp),
                to_numpy(modes_real_pmwd),
                atol=1e-12,
                rtol=1e-12,
            )
        ),
        "modes_metrics": modes_metrics,
        "lpt_disp_metrics": disp_metrics,
        "lpt_vel_metrics": vel_metrics,
        "lpt_density_metrics": density_metrics,
        "mode_gradient_metrics": gradient_metrics,
        "loss_values": {
            "pmwd": float(loss_pmwd(modes_real_pmwd)),
            "pmpp": float(loss_pmpp(modes_real_pmpp)),
        },
        "worst_gradient_entry": {
            "index": [int(i) for i in worst_grad_idx],
            "pmwd": float(grad_pmwd[worst_grad_idx]),
            "pmpp": float(grad_pmpp[worst_grad_idx]),
            "abs_diff": float(np.abs(gradient_diff[worst_grad_idx])),
        },
        "artifacts": {
            "modes_plot": str(modes_plot_path),
            "lpt_plot": str(lpt_plot_path),
            "gradient_plot": str(grad_plot_path),
            "metrics_json": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved modes plot to {modes_plot_path}")
    print(f"Saved LPT plot to {lpt_plot_path}")
    print(f"Saved gradient plot to {grad_plot_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(report["mode_gradient_metrics"], indent=2))


if __name__ == "__main__":
    main()
