import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from pmpp.analysis import delta_to_cross_correlation, delta_to_pk
from pmpp.core import Configuration
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh


def test_two_gpu_power_and_cross_spectra_match_dense_values_and_gradients():
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("this distributed spectrum regression requires two GPUs")
    single = Configuration(1.25, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float32)
    multi = Configuration(
        1.25, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices[:2]), mode="mesh_halo"),
        max_ptcl_per_slice=320, max_share_ptcl=96, max_halo_share_ptcl=160, max_share_gather_ptcl=96,
    )
    rng = np.random.default_rng(20260807)
    field_a = rng.normal(size=single.mesh_shape).astype(np.float32)
    field_b = (0.73 * field_a + 0.27 * rng.normal(size=single.mesh_shape)).astype(np.float32)
    field_a -= field_a.mean(dtype=np.float64)
    field_b -= field_b.mean(dtype=np.float64)
    a_single = jnp.asarray(field_a)
    b_single = jnp.asarray(field_b)
    sharding = NamedSharding(multi.compute_mesh, P("gpus", None, None))
    a_multi = jax.device_put(a_single, sharding)
    b_multi = jax.device_put(b_single, sharding)

    single_pk = delta_to_pk(a_single, single, mas="CIC")
    multi_pk = delta_to_pk(a_multi, multi, mas="CIC")
    for actual, expected in zip(multi_pk, single_pk):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-6, atol=3e-6)

    single_cross = delta_to_cross_correlation(a_single, b_single, single, mas="TSC")
    multi_cross = delta_to_cross_correlation(a_multi, b_multi, multi, mas="TSC")
    for actual, expected in zip(multi_cross, single_cross):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=4e-6, atol=4e-6)

    weights = jnp.linspace(0.25, 1.75, single_pk[1].shape[0], dtype=jnp.float32)

    def loss(field, conf):
        return jnp.vdot(delta_to_pk(field, conf, mas="CIC")[1], weights)

    grad_single = jax.grad(loss)(a_single, single)
    grad_multi = jax.grad(loss)(a_multi, multi)
    np.testing.assert_allclose(np.asarray(grad_multi), np.asarray(grad_single), rtol=8e-6, atol=8e-7)


def test_odd_mesh_parseval_normalization_and_mas_contract_are_exact():
    conf = Configuration(0.7, (7, 7, 7), mesh_shape=1, float_dtype=jnp.float64, pallas_cic=False)
    axis = jnp.arange(7, dtype=jnp.float64)
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    delta = (
        0.3 * jnp.cos(2 * jnp.pi * x / 7) - 0.2 * jnp.sin(4 * jnp.pi * y / 7) +
        0.1 * jnp.cos(2 * jnp.pi * (x + y + z) / 7)
    )
    k, pk, nmodes = delta_to_pk(delta, conf, mas=None)
    assert np.all(np.diff(np.asarray(k)) > 0)
    assert np.all(np.asarray(nmodes) > 0)
    spectral_variance = jnp.sum(pk * nmodes)
    real_variance = jnp.prod(jnp.asarray(conf.box_size)) * jnp.mean(jnp.square(delta))
    np.testing.assert_allclose(spectral_variance, real_variance, rtol=3e-14, atol=3e-14)

    with pytest.raises(ValueError, match="Unsupported MAS"):
        delta_to_pk(delta, conf, mas="nearestish")
