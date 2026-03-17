#!/usr/bin/env python
"""Investigate gradients through from_sigma8 -> boltzmann -> white_noise -> linear_modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
from pmwd.cosmology import Cosmology as CosmologyPMWD
from pmwd.modes import linear_modes as linear_modes_pmwd
from pmwd.modes import white_noise as white_noise_pmwd

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import Cosmology as CosmologyPMPP
from src.modes import linear_modes as linear_modes_pmpp
from src.modes import white_noise as white_noise_pmpp
from src.utils import create_compute_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=100.0)
    parser.add_argument("--num-ptcl", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma8", type=float, default=0.8)
    parser.add_argument("--omega-m", type=float, default=0.31)
    parser.add_argument("--n-s", type=float, default=0.9652)
    parser.add_argument("--omega-b", type=float, default=0.02233)
    parser.add_argument("--h", type=float, default=0.6737)
    parser.add_argument("--num-devices", type=int, default=2)
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
    ptcl_spacing = args.box_size / args.num_ptcl
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        ptcl_grid_shape,
        mesh_shape=1,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(args.num_ptcl**3 / len(gpu_devices) * 1.7),
        max_share_ptcl=2000,
        max_share_gather_ptcl=6000,
        to_save_z=[1, 2 / 3, 1 / 3, 0],
        a_start=1 / 60,
        a_nbody_maxstep=1 / 60,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def finite_difference(loss_fn, params: np.ndarray, eps: np.ndarray) -> np.ndarray:
    grads = []
    for i, step in enumerate(eps):
        delta = np.zeros_like(params)
        delta[i] = step
        plus = float(loss_fn(params + delta))
        minus = float(loss_fn(params - delta))
        grads.append((plus - minus) / (2 * step))
    return np.array(grads, dtype=np.float64)


def vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = candidate - reference
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(abs_diff**2))),
        "allclose_atol_1e-8_rtol_1e-6": bool(np.allclose(candidate, reference, atol=1e-8, rtol=1e-6)),
    }


def save_forward_plot(
    modes_pmwd: np.ndarray,
    modes_pmpp: np.ndarray,
    output_path: Path,
    metrics: dict[str, float],
) -> None:
    ref_real = np.real(modes_pmwd).ravel()
    cand_real = np.real(modes_pmpp).ravel()
    ref_imag = np.imag(modes_pmwd).ravel()
    cand_imag = np.imag(modes_pmpp).ravel()
    abs_resid = np.abs((modes_pmpp - modes_pmwd).ravel())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    lim = float(max(np.max(np.abs(ref_real)), np.max(np.abs(cand_real)), 1e-14))
    axes[0].scatter(ref_real, cand_real, s=6, alpha=0.18, linewidths=0, color="#1f77b4")
    axes[0].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_title("linear_modes real parity")
    axes[0].set_xlabel("PMWD")
    axes[0].set_ylabel("PMPP")

    lim = float(max(np.max(np.abs(ref_imag)), np.max(np.abs(cand_imag)), 1e-14))
    axes[1].scatter(ref_imag, cand_imag, s=6, alpha=0.18, linewidths=0, color="#d55e00")
    axes[1].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_title("linear_modes imag parity")
    axes[1].set_xlabel("PMWD")
    axes[1].set_ylabel("PMPP")

    axes[2].hist(abs_resid, bins=60, color="#009e73", alpha=0.85, log=True)
    axes[2].set_title("linear_modes residuals")
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

    fig.suptitle("white_noise -> linear_modes forward parity", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gradient_plot(
    stage_metrics: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    def grouped_bar(ax, title, categories, series):
        x = np.arange(len(categories))
        width = 0.24
        colors = ["#1f77b4", "#d55e00", "#009e73"]
        for i, (label, values) in enumerate(series):
            ax.bar(x + (i - 1) * width, values, width=width, label=label, color=colors[i], alpha=0.85)
            for xpos, value in zip(x + (i - 1) * width, values):
                if np.isnan(value):
                    ax.text(xpos, 0.0, "NaN", ha="center", va="bottom", rotation=90, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(title)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)

    from_sigma8 = stage_metrics["from_sigma8_A_s_1e9"]
    grouped_bar(
        axes[0],
        "from_sigma8 -> A_s_1e9 gradients",
        ["sigma8", "Omega_m"],
        [
            ("PMWD AD", [from_sigma8["pmwd_ad"]["sigma8"], from_sigma8["pmwd_ad"]["omega_m"]]),
            ("PMPP AD", [from_sigma8["pmpp_ad"]["sigma8"], from_sigma8["pmpp_ad"]["omega_m"]]),
            ("PMPP FD", [from_sigma8["pmpp_fd"]["sigma8"], from_sigma8["pmpp_fd"]["omega_m"]]),
        ],
    )

    sigma8_base = stage_metrics["sigma8_base"]
    grouped_bar(
        axes[1],
        "boltzmann(...).sigma8 base gradient",
        ["Omega_m"],
        [
            ("PMWD AD", [sigma8_base["pmwd_ad"]["omega_m"]]),
            ("PMPP AD", [sigma8_base["pmpp_ad"]["omega_m"]]),
            ("PMPP FD", [sigma8_base["pmpp_fd"]["omega_m"]]),
        ],
    )

    modes_loss = stage_metrics["linear_modes_loss"]
    grouped_bar(
        axes[2],
        "linear_modes loss gradients",
        ["sigma8", "Omega_m"],
        [
            ("PMWD AD", [modes_loss["pmwd_ad"]["sigma8"], modes_loss["pmwd_ad"]["omega_m"]]),
            ("PMPP AD", [modes_loss["pmpp_ad"]["sigma8"], modes_loss["pmpp_ad"]["omega_m"]]),
            ("PMPP FD", [modes_loss["pmpp_fd"]["sigma8"], modes_loss["pmpp_fd"]["omega_m"]]),
        ],
    )
    axes[2].text(
        0.03,
        0.03,
        "Observation: direct sigma8 normalization has a NaN reverse-mode\n"
        "gradient wrt Omega_m, but the tested linear_modes loss gradients\n"
        "still match PMWD and finite differences.",
        transform=axes[2].transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 2},
    )

    fig.suptitle("Gradient investigation for from_sigma8 -> boltzmann -> linear_modes", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forward_plot_path = args.output_dir / "modes_pipeline_forward_parity.png"
    gradient_plot_path = args.output_dir / "modes_pipeline_gradient_summary.png"
    metrics_path = args.output_dir / "modes_pipeline_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)
    conf_pmpp, conf_pmwd = init_confs(args, gpu_devices)

    params = np.array([args.sigma8, args.omega_m], dtype=np.float64)
    eps_modes = np.array([1e-5, 1e-4], dtype=np.float64)
    eps_as = np.array([1e-5, 1e-4], dtype=np.float64)
    eps_sigma8_base = np.array([1e-4], dtype=np.float64)

    white_pmwd = white_noise_pmwd(args.seed, conf_pmwd)
    white_pmpp = white_noise_pmpp(args.seed, conf_pmpp)
    white_pmwd_np = to_numpy(white_pmwd)
    white_pmpp_np = to_numpy(white_pmpp)

    probe = (
        jax.random.normal(jax.random.PRNGKey(123), white_pmwd.shape, dtype=conf_pmpp.float_dtype)
        + 1j * jax.random.normal(jax.random.PRNGKey(124), white_pmwd.shape, dtype=conf_pmpp.float_dtype)
    ).astype(jnp.complex128)

    def build_cosmo_pmwd_from_sigma8(sigma8_value, omega_m_value):
        return CosmologyPMWD.from_sigma8(
            conf_pmwd,
            sigma8=sigma8_value,
            n_s=args.n_s,
            Omega_m=omega_m_value,
            Omega_b=args.omega_b,
            h=args.h,
        )

    def build_cosmo_pmpp_from_sigma8(sigma8_value, omega_m_value):
        return CosmologyPMPP.from_sigma8(
            conf_pmpp,
            sigma8=sigma8_value,
            n_s=args.n_s,
            Omega_m=omega_m_value,
            Omega_b=args.omega_b,
            h=args.h,
        )

    def sigma8_base_pmwd(omega_m_value):
        cosmo = CosmologyPMWD(conf_pmwd, 1, n_s=args.n_s, Omega_m=omega_m_value, Omega_b=args.omega_b, h=args.h)
        cosmo = boltzmann_pmwd(cosmo, conf_pmwd)
        return cosmo.sigma8

    def sigma8_base_pmpp(omega_m_value):
        cosmo = CosmologyPMPP(conf_pmpp, 1, n_s=args.n_s, Omega_m=omega_m_value, Omega_b=args.omega_b, h=args.h)
        cosmo = boltzmann_pmpp(cosmo, conf_pmpp)
        return cosmo.sigma8

    def a_s_pmwd(vector):
        sigma8_value, omega_m_value = vector
        return build_cosmo_pmwd_from_sigma8(sigma8_value, omega_m_value).A_s_1e9

    def a_s_pmpp(vector):
        sigma8_value, omega_m_value = vector
        return build_cosmo_pmpp_from_sigma8(sigma8_value, omega_m_value).A_s_1e9

    def modes_loss_pmwd(vector):
        sigma8_value, omega_m_value = vector
        cosmo = build_cosmo_pmwd_from_sigma8(sigma8_value, omega_m_value)
        cosmo = boltzmann_pmwd(cosmo, conf_pmwd)
        modes = linear_modes_pmwd(white_pmwd, cosmo, conf_pmwd)
        return jnp.real(jnp.vdot(modes, probe)) / modes.size

    def modes_loss_pmpp(vector):
        sigma8_value, omega_m_value = vector
        cosmo = build_cosmo_pmpp_from_sigma8(sigma8_value, omega_m_value)
        cosmo = boltzmann_pmpp(cosmo, conf_pmpp)
        modes = linear_modes_pmpp(white_pmpp, cosmo, conf_pmpp)
        return jnp.real(jnp.vdot(modes, probe)) / modes.size

    cosmo_pmwd = build_cosmo_pmwd_from_sigma8(args.sigma8, args.omega_m)
    cosmo_pmwd = boltzmann_pmwd(cosmo_pmwd, conf_pmwd)
    cosmo_pmpp = build_cosmo_pmpp_from_sigma8(args.sigma8, args.omega_m)
    cosmo_pmpp = boltzmann_pmpp(cosmo_pmpp, conf_pmpp)
    linear_pmwd = to_numpy(linear_modes_pmwd(white_pmwd, cosmo_pmwd, conf_pmwd))
    linear_pmpp = to_numpy(linear_modes_pmpp(white_pmpp, cosmo_pmpp, conf_pmpp))

    white_metrics = vector_metrics(white_pmwd_np, white_pmpp_np)
    linear_metrics = vector_metrics(linear_pmwd, linear_pmpp)

    a_s_pmwd_ad = to_numpy(jax.grad(a_s_pmwd)(jnp.asarray(params, dtype=jnp.float64)))
    a_s_pmpp_ad = to_numpy(jax.grad(a_s_pmpp)(jnp.asarray(params, dtype=jnp.float64)))
    a_s_pmpp_fd = finite_difference(a_s_pmpp, params, eps_as)

    sigma8_base_pmwd_ad = float(jax.grad(sigma8_base_pmwd)(params[1]))
    sigma8_base_pmpp_ad = float(jax.grad(sigma8_base_pmpp)(params[1]))
    sigma8_base_pmpp_fd = float(
        (sigma8_base_pmpp(params[1] + eps_sigma8_base[0]) - sigma8_base_pmpp(params[1] - eps_sigma8_base[0]))
        / (2 * eps_sigma8_base[0])
    )

    modes_loss_pmwd_ad = to_numpy(jax.jit(jax.grad(modes_loss_pmwd))(jnp.asarray(params, dtype=jnp.float64)))
    modes_loss_pmpp_ad = to_numpy(jax.jit(jax.grad(modes_loss_pmpp))(jnp.asarray(params, dtype=jnp.float64)))
    modes_loss_pmpp_fd = finite_difference(modes_loss_pmpp, params, eps_modes)

    gradient_summary = {
        "from_sigma8_A_s_1e9": {
            "pmwd_ad": {"sigma8": float(a_s_pmwd_ad[0]), "omega_m": float(a_s_pmwd_ad[1])},
            "pmpp_ad": {"sigma8": float(a_s_pmpp_ad[0]), "omega_m": float(a_s_pmpp_ad[1])},
            "pmpp_fd": {"sigma8": float(a_s_pmpp_fd[0]), "omega_m": float(a_s_pmpp_fd[1])},
        },
        "sigma8_base": {
            "pmwd_ad": {"omega_m": sigma8_base_pmwd_ad},
            "pmpp_ad": {"omega_m": sigma8_base_pmpp_ad},
            "pmpp_fd": {"omega_m": sigma8_base_pmpp_fd},
        },
        "linear_modes_loss": {
            "pmwd_ad": {"sigma8": float(modes_loss_pmwd_ad[0]), "omega_m": float(modes_loss_pmwd_ad[1])},
            "pmpp_ad": {"sigma8": float(modes_loss_pmpp_ad[0]), "omega_m": float(modes_loss_pmpp_ad[1])},
            "pmpp_fd": {"sigma8": float(modes_loss_pmpp_fd[0]), "omega_m": float(modes_loss_pmpp_fd[1])},
        },
    }

    save_forward_plot(linear_pmwd, linear_pmpp, forward_plot_path, linear_metrics)
    save_gradient_plot(gradient_summary, gradient_plot_path)

    report = {
        "config": {
            "box_size": args.box_size,
            "num_ptcl": args.num_ptcl,
            "seed": args.seed,
            "sigma8": args.sigma8,
            "omega_m": args.omega_m,
            "n_s": args.n_s,
            "omega_b": args.omega_b,
            "h": args.h,
            "num_devices": args.num_devices,
        },
        "white_noise_metrics": white_metrics,
        "linear_modes_metrics": linear_metrics,
        "gradient_summary": gradient_summary,
        "artifacts": {
            "forward_plot": str(forward_plot_path),
            "gradient_plot": str(gradient_plot_path),
            "metrics_json": str(metrics_path),
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved forward plot to {forward_plot_path}")
    print(f"Saved gradient plot to {gradient_plot_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(report["gradient_summary"], indent=2))


if __name__ == "__main__":
    main()
