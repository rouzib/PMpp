import jax
import jax.numpy as jnp
import numpy as np

from pmpp.distributed import create_batched_transposed_real_ffts, create_compute_mesh, create_ffts
from pmpp.distributed.fft import (
    _batched_fftn_first_pass, _batched_ifftn_first_pass, _batched_irfftn_second_pass, _batched_rfftn_second_pass,
    _fftn_first_pass, _fftn_second_pass, _ifftn_first_pass, _ifftn_second_pass, _irfftn_second_pass, _rfftn_second_pass,
    distribute_array_on_gpus, distribute_array_on_gpus_old, split_array_for_gpus, test_functions as
    compare_fft_functions,
)
from pmpp.core import AXIS_NAME
from jax.sharding import PartitionSpec as P


def _one_device_mesh():
    return create_compute_mesh(jax.devices("gpu")[:1])


def test_local_fft_passes_transform_only_the_documented_axes():
    real = jnp.arange(3 * 4 * 6, dtype=jnp.float32).reshape(3, 4, 6)
    complex_values = real.astype(jnp.complex64) + 1j * (real[::-1] + 1)
    np.testing.assert_allclose(
        np.asarray(_fftn_first_pass(complex_values)), np.asarray(jnp.fft.fft(complex_values, axis=0))
    )
    np.testing.assert_allclose(
        np.asarray(_fftn_second_pass(complex_values)), np.asarray(jnp.fft.fftn(complex_values, axes=(1, 2)))
    )
    np.testing.assert_allclose(np.asarray(_rfftn_second_pass(real)), np.asarray(jnp.fft.rfftn(real, axes=(1, 2))))
    np.testing.assert_allclose(
        np.asarray(_ifftn_first_pass(complex_values)), np.asarray(jnp.fft.ifft(complex_values, axis=0))
    )
    np.testing.assert_allclose(
        np.asarray(_ifftn_second_pass(complex_values)), np.asarray(jnp.fft.ifftn(complex_values, axes=(1, 2)))
    )
    np.testing.assert_allclose(
        np.asarray(_irfftn_second_pass(jnp.fft.rfftn(real, axes=(1, 2)))), np.asarray(real), rtol=2e-6, atol=2e-6
    )

    batched = jnp.stack((real, 2 * real))
    np.testing.assert_allclose(np.asarray(_batched_fftn_first_pass(batched)), np.asarray(jnp.fft.fft(batched, axis=1)))
    np.testing.assert_allclose(
        np.asarray(_batched_rfftn_second_pass(batched)), np.asarray(jnp.fft.rfftn(batched, axes=(2, 3)))
    )
    batched_complex = jnp.fft.fft(batched.astype(jnp.complex64), axis=1)
    np.testing.assert_allclose(np.asarray(_batched_ifftn_first_pass(batched_complex)), np.asarray(batched))
    batched_real_hat = jnp.fft.rfftn(batched, axes=(2, 3))
    np.testing.assert_allclose(
        np.asarray(_batched_irfftn_second_pass(batched_real_hat)), np.asarray(batched), rtol=2e-6, atol=2e-6
    )


def test_host_split_and_both_distribution_helpers_preserve_global_values_and_sharding():
    devices = jax.devices("gpu")[:2]
    mesh = create_compute_mesh(devices)
    values = np.arange(8 * 3 * 2, dtype=np.float32).reshape(8, 3, 2)
    chunks = split_array_for_gpus(values, len(devices), axis=0)
    np.testing.assert_array_equal(np.asarray(chunks), np.stack(np.array_split(values, len(devices), axis=0)))

    partition = P(AXIS_NAME, None, None)
    modern = distribute_array_on_gpus(values, mesh, partition)
    legacy = distribute_array_on_gpus_old(values, mesh, partition, axis_name=AXIS_NAME)
    np.testing.assert_array_equal(np.asarray(modern), values)
    np.testing.assert_array_equal(np.asarray(legacy), values)
    assert modern.sharding.spec == partition
    assert legacy.sharding.spec == partition
    assert len(modern.addressable_shards) == len(devices)


def test_single_device_fft_family_matches_jax_forward_inverse_and_pullbacks():
    mesh = _one_device_mesh()
    rfftn, irfftn, fftn, ifftn, rfftn_t, irfftn_t = create_ffts(mesh)
    real = jax.random.normal(jax.random.PRNGKey(8), (4, 3, 6), dtype=jnp.float32)
    complex_values = real.astype(jnp.complex64) + 0.2j * real[::-1]

    expected_hat = jnp.fft.rfftn(real)
    np.testing.assert_allclose(np.asarray(rfftn(real)), np.asarray(expected_hat), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(np.asarray(rfftn_t(real)), np.asarray(expected_hat), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(np.asarray(irfftn(expected_hat)), np.asarray(real), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(np.asarray(irfftn_t(expected_hat)), np.asarray(real), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(
        np.asarray(fftn(complex_values)), np.asarray(jnp.fft.fftn(complex_values)), rtol=3e-6, atol=3e-6
    )
    np.testing.assert_allclose(
        np.asarray(ifftn(jnp.fft.fftn(complex_values))), np.asarray(complex_values), rtol=3e-6, atol=3e-6
    )

    weights = (
        jax.random.normal(jax.random.PRNGKey(9), expected_hat.shape) +
        1j * jax.random.normal(jax.random.PRNGKey(10), expected_hat.shape)
    ).astype(jnp.complex64)

    def forward_loss(function, value):
        transformed = function(value)
        return jnp.real(jnp.vdot(weights, transformed))

    reference_grad = jax.grad(lambda value: forward_loss(jnp.fft.rfftn, value))(real)
    for function in (rfftn, rfftn_t):
        actual_grad = jax.grad(lambda value: forward_loss(function, value))(real)
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(reference_grad), rtol=4e-6, atol=4e-6)

    real_weights = jax.random.normal(jax.random.PRNGKey(11), real.shape, dtype=jnp.float32)
    reference_vjp = jax.vjp(jnp.fft.irfftn, expected_hat)[1](real_weights)[0]
    for function in (irfftn, irfftn_t):
        actual_vjp = jax.vjp(function, expected_hat)[1](real_weights)[0]
        np.testing.assert_allclose(np.asarray(actual_vjp), np.asarray(reference_vjp), rtol=4e-6, atol=4e-6)


def test_single_device_batched_transposed_fft_matches_independent_transforms_and_vjp():
    mesh = _one_device_mesh()
    forward, inverse = create_batched_transposed_real_ffts(mesh)
    real = jax.random.normal(jax.random.PRNGKey(18), (3, 4, 3, 6), dtype=jnp.float32)
    expected_hat = jnp.fft.rfftn(real, axes=(1, 2, 3))

    np.testing.assert_allclose(np.asarray(forward(real)), np.asarray(expected_hat), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(np.asarray(inverse(expected_hat)), np.asarray(real), rtol=3e-6, atol=3e-6)

    weights = jax.random.normal(jax.random.PRNGKey(19), real.shape, dtype=jnp.float32)
    expected_vjp = jax.vjp(lambda value: jnp.fft.irfftn(value, axes=(1, 2, 3)), expected_hat)[1](weights)[0]
    actual_vjp = jax.vjp(inverse, expected_hat)[1](weights)[0]
    np.testing.assert_allclose(np.asarray(actual_vjp), np.asarray(expected_vjp), rtol=4e-6, atol=4e-6)


def test_fft_diagnostic_reports_both_exact_and_inexact_results(capsys):
    mesh = _one_device_mesh()
    values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    assert compare_fft_functions(lambda value: value, lambda value: value, values, mesh) is None
    exact_output = capsys.readouterr().out
    assert "Output close to reference" in exact_output
    assert "max_diff = np.float32(0.0)" in exact_output or "max_diff = Array(0." in exact_output

    assert compare_fft_functions(lambda value: value + 0.5, lambda value: value, values, mesh) is None
    inexact_output = capsys.readouterr().out
    assert "WARNING: Output not close to reference" in inexact_output
    assert "max_relative_diff" in inexact_output
