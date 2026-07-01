import sys
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmpp.FFT_distributed import create_batched_transposed_real_ffts, create_ffts
from pmpp.utils import create_compute_mesh

try:
    import pytest
except ImportError:
    pytest = None


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def test_distributed_fft_matches_reference_for_forward_and_gradients():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("distributed FFT test requires at least 1 GPU")
        raise SystemExit("distributed FFT test requires at least 1 GPU")

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)
    sharding = NamedSharding(compute_mesh, P("gpus", None, None))
    rfftn, irfftn, _, _, _, _ = create_ffts(compute_mesh)

    for real_shape in ((32, 32, 32), (32, 32, 33)):
        spectrum_shape = (real_shape[0], real_shape[1], real_shape[2] // 2 + 1)

        real_field = jax.random.normal(jax.random.PRNGKey(real_shape[2]), real_shape, dtype=jnp.float64)
        real_field_sharded = jax.device_put(real_field, sharding)

        spectrum = (
            jax.random.normal(jax.random.PRNGKey(100 + real_shape[2]), spectrum_shape, dtype=jnp.float64)
            + 1j
            * jax.random.normal(jax.random.PRNGKey(200 + real_shape[2]), spectrum_shape, dtype=jnp.float64)
        ).astype(jnp.complex128)
        spectrum_sharded = jax.device_put(spectrum, sharding)

        rfftn_ref = np.asarray(jax.device_get(jnp.fft.rfftn(real_field)))
        rfftn_out = np.asarray(jax.device_get(rfftn(real_field_sharded)))
        assert np.allclose(rfftn_out, rfftn_ref, atol=1e-12, rtol=1e-12)

        irfftn_ref = np.asarray(jax.device_get(jnp.fft.irfftn(spectrum)))
        irfftn_out = np.asarray(jax.device_get(irfftn(spectrum_sharded)))
        assert np.allclose(irfftn_out, irfftn_ref, atol=1e-12, rtol=1e-12)

        rfftn_weights = (
            jax.random.normal(jax.random.PRNGKey(300 + real_shape[2]), rfftn_ref.shape, dtype=jnp.float64)
            + 1j
            * jax.random.normal(jax.random.PRNGKey(400 + real_shape[2]), rfftn_ref.shape, dtype=jnp.float64)
        ).astype(jnp.complex128)

        def loss_rfftn_ref(inp):
            return jnp.real(jnp.vdot(jnp.fft.rfftn(inp), rfftn_weights))

        def loss_rfftn_mgpu(inp):
            return jnp.real(jnp.vdot(rfftn(inp), rfftn_weights))

        grad_rfftn_ref = np.asarray(jax.device_get(jax.grad(loss_rfftn_ref)(real_field)))
        grad_rfftn_out = np.asarray(jax.device_get(jax.grad(loss_rfftn_mgpu)(real_field_sharded)))
        assert np.allclose(grad_rfftn_out, grad_rfftn_ref, atol=1e-12, rtol=1e-12)

        irfftn_weights = jax.random.normal(jax.random.PRNGKey(500 + real_shape[2]), irfftn_ref.shape, dtype=jnp.float64)

        def loss_irfftn_ref(inp):
            return jnp.sum(jnp.fft.irfftn(inp) * irfftn_weights)

        def loss_irfftn_mgpu(inp):
            return jnp.sum(irfftn(inp) * irfftn_weights)

        grad_irfftn_ref = np.asarray(jax.device_get(jax.grad(loss_irfftn_ref)(spectrum)))
        grad_irfftn_out = np.asarray(jax.device_get(jax.grad(loss_irfftn_mgpu)(spectrum_sharded)))
        assert np.allclose(grad_irfftn_out, grad_irfftn_ref, atol=1e-12, rtol=1e-12)


def test_batched_transposed_irfftn_matches_scalar_transposed_path():
    if GPU_COUNT < 1:
        if pytest is not None:
            pytest.skip("distributed FFT test requires at least 1 GPU")
        raise SystemExit("distributed FFT test requires at least 1 GPU")

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)
    real_sharding = NamedSharding(compute_mesh, P(None, "gpus", None, None))
    spectrum_sharding = NamedSharding(compute_mesh, P(None, None, "gpus", None))
    _, _, _, _, rfftn_transposed, irfftn_transposed = create_ffts(compute_mesh)
    batched_rfftn_transposed, batched_irfftn_transposed = create_batched_transposed_real_ffts(compute_mesh)

    for real_shape in ((32, 32, 32), (32, 32, 33)):
        batch_size = 3
        spectrum_shape = (batch_size, real_shape[0], real_shape[1], real_shape[2] // 2 + 1)

        real_batch = jax.random.normal(
            jax.random.PRNGKey(600 + real_shape[2]),
            (batch_size,) + real_shape,
            dtype=jnp.float64,
        )
        real_batch_sharded = jax.device_put(real_batch, real_sharding)
        scalar_rfftn = jnp.stack(
            [rfftn_transposed(real_batch_sharded[i]) for i in range(batch_size)],
            axis=0,
        )
        batched_rfftn_out = np.asarray(jax.device_get(batched_rfftn_transposed(real_batch_sharded)))
        scalar_rfftn_ref = np.asarray(jax.device_get(scalar_rfftn))
        assert np.allclose(batched_rfftn_out, scalar_rfftn_ref, atol=1e-12, rtol=1e-12)

        spectrum_batch = (
            jax.random.normal(jax.random.PRNGKey(700 + real_shape[2]), spectrum_shape, dtype=jnp.float64)
            + 1j
            * jax.random.normal(jax.random.PRNGKey(800 + real_shape[2]), spectrum_shape, dtype=jnp.float64)
        ).astype(jnp.complex128)
        spectrum_batch_sharded = jax.device_put(spectrum_batch, spectrum_sharding)

        batched_irfftn_out = np.asarray(jax.device_get(batched_irfftn_transposed(spectrum_batch_sharded)))
        scalar_irfftn_out = np.asarray(
            jax.device_get(
                jnp.stack(
                    [irfftn_transposed(spectrum_batch_sharded[i]) for i in range(batch_size)],
                    axis=0,
                )
            )
        )
        assert np.allclose(batched_irfftn_out, scalar_irfftn_out, atol=1e-12, rtol=1e-12)

        output_shape = batched_irfftn_out.shape
        irfftn_weights = jax.random.normal(
            jax.random.PRNGKey(900 + real_shape[2]),
            output_shape,
            dtype=jnp.float64,
        )

        def loss_scalar(inp):
            outs = [irfftn_transposed(inp[i]) for i in range(batch_size)]
            return sum(jnp.sum(outs[i] * irfftn_weights[i]) for i in range(batch_size))

        def loss_batched(inp):
            return jnp.sum(batched_irfftn_transposed(inp) * irfftn_weights)

        grad_scalar = np.asarray(jax.device_get(jax.grad(loss_scalar)(spectrum_batch_sharded)))
        grad_batched = np.asarray(jax.device_get(jax.grad(loss_batched)(spectrum_batch_sharded)))
        assert np.allclose(grad_batched, grad_scalar, atol=1e-12, rtol=1e-12)


if pytest is not None:
    test_distributed_fft_matches_reference_for_forward_and_gradients = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="distributed FFT test requires at least 1 GPU",
    )(test_distributed_fft_matches_reference_for_forward_and_gradients)
    test_batched_transposed_irfftn_matches_scalar_transposed_path = pytest.mark.skipif(
        GPU_COUNT < 1,
        reason="distributed FFT test requires at least 1 GPU",
    )(test_batched_transposed_irfftn_matches_scalar_transposed_path)


if __name__ == "__main__":
    test_distributed_fft_matches_reference_for_forward_and_gradients()
    test_batched_transposed_irfftn_matches_scalar_transposed_path()
    print("distributed FFT regression passed")
