#!/usr/bin/env python
"""Compare distributed FFT forward passes and gradients against JAX references."""

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
from jax.sharding import NamedSharding, PartitionSpec as P

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmpp.FFT_distributed import create_ffts
from pmpp.utils import create_compute_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", type=int, default=64)
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


def metrics_from_diff(diff: np.ndarray) -> dict[str, float]:
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(abs_diff**2))),
    }


def legacy_buggy_mask(max_n: int, n: int, is_odd: int, dtype) -> jax.Array:
    mask = jnp.ones(max_n, dtype=dtype)
    mask = mask.at[1:max_n - 1].set(2.0)
    mask = mask.at[max_n - 1].set(1.0 - is_odd)
    return mask[:n]


def parity_limits(x: np.ndarray, y: np.ndarray) -> float:
    return float(max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-18))


def save_combined_plot(
    rfftn_forward_diff: np.ndarray,
    irfftn_forward_diff: np.ndarray,
    grad_rfftn_ref: np.ndarray,
    grad_rfftn_mgpu: np.ndarray,
    grad_irfftn_ref: np.ndarray,
    grad_irfftn_mgpu: np.ndarray,
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    spec_slice = np.log10(np.clip(np.abs(rfftn_forward_diff[:, :, rfftn_forward_diff.shape[2] // 2]), 1e-18, None))
    img = axes[0, 0].imshow(spec_slice.T, origin="lower", cmap="magma")
    axes[0, 0].set_title("rfftn forward residual")
    axes[0, 0].set_xlabel("kx")
    axes[0, 0].set_ylabel("ky")
    fig.colorbar(img, ax=axes[0, 0], shrink=0.85, label="log10 |delta|")

    real_slice = np.log10(np.clip(np.abs(irfftn_forward_diff[irfftn_forward_diff.shape[0] // 2]), 1e-18, None))
    img = axes[0, 1].imshow(real_slice, origin="lower", cmap="magma")
    axes[0, 1].set_title("irfftn forward residual")
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("z")
    fig.colorbar(img, ax=axes[0, 1], shrink=0.85, label="log10 |delta|")

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0,
        1.0,
        "\n".join(
            [
                "Current FFT parity metrics",
                "",
                f"rfftn forward max |delta| = {metrics['rfftn_forward']['max_abs_diff']:.3e}",
                f"irfftn forward max |delta| = {metrics['irfftn_forward']['max_abs_diff']:.3e}",
                f"rfftn grad max |delta| = {metrics['rfftn_grad']['max_abs_diff']:.3e}",
                f"irfftn grad max |delta| = {metrics['irfftn_grad']['max_abs_diff']:.3e}",
                "",
                f"legacy irfftn grad max |delta| = {metrics['legacy_irfftn_grad']['max_abs_diff']:.3e}",
                "",
                "The legacy bug doubled the even-size Nyquist plane in",
                "the irfftn backward mask.",
            ]
        ),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    ax = axes[1, 0]
    lim = parity_limits(grad_rfftn_ref, grad_rfftn_mgpu)
    ax.scatter(grad_rfftn_ref.ravel(), grad_rfftn_mgpu.ravel(), s=4, alpha=0.18, linewidths=0)
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("rfftn grad parity")
    ax.set_xlabel("reference")
    ax.set_ylabel("mGPU")

    ax = axes[1, 1]
    lim = parity_limits(np.real(grad_irfftn_ref), np.real(grad_irfftn_mgpu))
    ax.scatter(
        np.real(grad_irfftn_ref).ravel(),
        np.real(grad_irfftn_mgpu).ravel(),
        s=4,
        alpha=0.18,
        linewidths=0,
        color="#1f77b4",
    )
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("irfftn grad parity (real)")
    ax.set_xlabel("reference")
    ax.set_ylabel("mGPU")

    ax = axes[1, 2]
    lim = parity_limits(np.imag(grad_irfftn_ref), np.imag(grad_irfftn_mgpu))
    ax.scatter(
        np.imag(grad_irfftn_ref).ravel(),
        np.imag(grad_irfftn_mgpu).ravel(),
        s=4,
        alpha=0.18,
        linewidths=0,
        color="#d55e00",
    )
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("irfftn grad parity (imag)")
    ax.set_xlabel("reference")
    ax.set_ylabel("mGPU")

    fig.suptitle("Distributed FFT forward and gradient parity", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_irfftn_last_axis_plot(
    grad_irfftn_diff: np.ndarray,
    legacy_irfftn_diff: np.ndarray,
    output_path: Path,
) -> None:
    current_max = np.maximum(np.max(np.abs(grad_irfftn_diff), axis=(0, 1)), 1e-18)
    legacy_max = np.maximum(np.max(np.abs(legacy_irfftn_diff), axis=(0, 1)), 1e-18)
    current_mean = np.maximum(np.mean(np.abs(grad_irfftn_diff), axis=(0, 1)), 1e-18)
    legacy_mean = np.maximum(np.mean(np.abs(legacy_irfftn_diff), axis=(0, 1)), 1e-18)
    kz = np.arange(current_max.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True)

    axes[0].plot(kz, current_max, label="current", linewidth=2.0, color="#1f77b4")
    axes[0].plot(kz, legacy_max, label="legacy buggy mask", linewidth=2.0, color="#d55e00")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max |delta grad|")
    axes[0].set_title("irfftn gradient residual by compressed kz index")
    axes[0].legend()

    axes[1].plot(kz, current_mean, linewidth=2.0, color="#1f77b4")
    axes[1].plot(kz, legacy_mean, linewidth=2.0, color="#d55e00")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("compressed kz index")
    axes[1].set_ylabel("mean |delta grad|")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_plot_path = args.output_dir / "fft_forward_and_grad_parity.png"
    kz_plot_path = args.output_dir / "fft_irfftn_last_axis_residuals.png"
    metrics_path = args.output_dir / "fft_metrics.json"

    gpu_devices = resolve_gpu_devices(args.num_devices)
    compute_mesh = create_compute_mesh(gpu_devices)
    sharding = NamedSharding(compute_mesh, P("gpus", None, None))
    rfftn, irfftn, _, _ = create_ffts(compute_mesh)

    real_field = jax.random.normal(
        jax.random.PRNGKey(0),
        (args.mesh_size, args.mesh_size, args.mesh_size),
        dtype=jnp.float64,
    )
    real_field_sharded = jax.device_put(real_field, sharding)

    spectrum = (
        jax.random.normal(
            jax.random.PRNGKey(1),
            (args.mesh_size, args.mesh_size, args.mesh_size // 2 + 1),
            dtype=jnp.float64,
        )
        + 1j
        * jax.random.normal(
            jax.random.PRNGKey(2),
            (args.mesh_size, args.mesh_size, args.mesh_size // 2 + 1),
            dtype=jnp.float64,
        )
    ).astype(jnp.complex128)
    spectrum_sharded = jax.device_put(spectrum, sharding)

    rfftn_ref = jnp.fft.rfftn(real_field)
    rfftn_out = rfftn(real_field_sharded)
    irfftn_ref = jnp.fft.irfftn(spectrum)
    irfftn_out = irfftn(spectrum_sharded)

    rfftn_weights = (
        jax.random.normal(jax.random.PRNGKey(3), rfftn_ref.shape, dtype=jnp.float64)
        + 1j * jax.random.normal(jax.random.PRNGKey(4), rfftn_ref.shape, dtype=jnp.float64)
    ).astype(jnp.complex128)

    def loss_rfftn_ref(inp):
        return jnp.real(jnp.vdot(jnp.fft.rfftn(inp), rfftn_weights))

    def loss_rfftn_mgpu(inp):
        return jnp.real(jnp.vdot(rfftn(inp), rfftn_weights))

    grad_rfftn_ref = jax.grad(loss_rfftn_ref)(real_field)
    grad_rfftn_mgpu = jax.grad(loss_rfftn_mgpu)(real_field_sharded)

    irfftn_weights = jax.random.normal(jax.random.PRNGKey(5), irfftn_ref.shape, dtype=jnp.float64)

    def loss_irfftn_ref(inp):
        return jnp.sum(jnp.fft.irfftn(inp) * irfftn_weights)

    def loss_irfftn_mgpu(inp):
        return jnp.sum(irfftn(inp) * irfftn_weights)

    grad_irfftn_ref = jax.grad(loss_irfftn_ref)(spectrum)
    grad_irfftn_mgpu = jax.grad(loss_irfftn_mgpu)(spectrum_sharded)

    fft_lengths = irfftn_ref.shape
    compressed_n = spectrum.shape[-1]
    is_odd = fft_lengths[-1] % 2
    scale = 1 / np.prod(fft_lengths)
    legacy_mask = legacy_buggy_mask(args.mesh_size, compressed_n, is_odd, spectrum.dtype)
    legacy_grad_irfftn = scale * jnp.fft.rfftn(irfftn_weights).conj() * legacy_mask[None, None, :]

    rfftn_forward_diff = to_numpy(rfftn_out - rfftn_ref)
    irfftn_forward_diff = to_numpy(irfftn_out - irfftn_ref)
    grad_rfftn_ref_np = to_numpy(grad_rfftn_ref)
    grad_rfftn_mgpu_np = to_numpy(grad_rfftn_mgpu)
    grad_irfftn_ref_np = to_numpy(grad_irfftn_ref)
    grad_irfftn_mgpu_np = to_numpy(grad_irfftn_mgpu)
    legacy_grad_irfftn_np = to_numpy(legacy_grad_irfftn)

    grad_rfftn_diff = grad_rfftn_mgpu_np - grad_rfftn_ref_np
    grad_irfftn_diff = grad_irfftn_mgpu_np - grad_irfftn_ref_np
    legacy_irfftn_diff = legacy_grad_irfftn_np - grad_irfftn_ref_np

    metrics = {
        "config": {
            "mesh_size": args.mesh_size,
            "num_devices": args.num_devices,
        },
        "rfftn_forward": metrics_from_diff(rfftn_forward_diff),
        "irfftn_forward": metrics_from_diff(irfftn_forward_diff),
        "rfftn_grad": metrics_from_diff(grad_rfftn_diff),
        "irfftn_grad": metrics_from_diff(grad_irfftn_diff),
        "legacy_irfftn_grad": metrics_from_diff(legacy_irfftn_diff),
        "legacy_bug_peak_kz_index": int(np.argmax(np.max(np.abs(legacy_irfftn_diff), axis=(0, 1)))),
    }

    save_combined_plot(
        rfftn_forward_diff,
        irfftn_forward_diff,
        grad_rfftn_ref_np,
        grad_rfftn_mgpu_np,
        grad_irfftn_ref_np,
        grad_irfftn_mgpu_np,
        metrics,
        combined_plot_path,
    )
    save_irfftn_last_axis_plot(grad_irfftn_diff, legacy_irfftn_diff, kz_plot_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved parity plot to {combined_plot_path}")
    print(f"Saved irfftn kz residual plot to {kz_plot_path}")
    print(f"Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
