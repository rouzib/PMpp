"""Correctness gates for the experimental Pallas CIC forward kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.configuration import Configuration
from pmpp.gather import _gather
from pmpp.pallas_cic import pallas_cic_supported
from pmpp.scatter import _scatter

pytestmark = pytest.mark.skipif(
    not pallas_cic_supported(jnp.float32), reason="Pallas CIC tests require a qualified float32 GPU JAX installation",
)


def _configuration(**kwargs):
    return Configuration(
        1.0, (4, 4, 4), mesh_shape=1, cosmo_dtype=jnp.float32, float_dtype=jnp.float32, pallas_cic=True, **kwargs,
    )


def _inputs(conf):
    key = jax.random.PRNGKey(17)
    pmid = jnp.array([[0, 0, 0], [1, 2, 3], [3, 0, 2], [2, 3, 1]], dtype=conf.pmid_dtype, )
    disp = jax.random.uniform(key, (4, 3), minval=-0.35, maxval=0.35, dtype=conf.float_dtype)
    mesh = jax.random.normal(key, (4, 4, 4, 2), dtype=conf.float_dtype)
    val = jnp.arange(1, 5, dtype=conf.float_dtype)
    return pmid, disp, mesh, val


def test_pallas_cic_matches_reference_and_gradients():
    conf = _configuration()
    reference = conf.replace(pallas_cic=False)
    pmid, disp, mesh, val = _inputs(conf)

    def gather_loss(d, m):
        return jnp.sum(_gather(pmid, d, conf, m, 0, 0, None)**2)

    def gather_reference_loss(d, m):
        return jnp.sum(_gather(pmid, d, reference, m, 0, 0, None)**2)

    gathered = _gather(pmid, disp, conf, mesh, 0, 0, None)
    gathered_reference = _gather(pmid, disp, reference, mesh, 0, 0, None)
    np.testing.assert_allclose(gathered, gathered_reference, rtol=2e-6, atol=2e-6)
    grad_gather = jax.grad(gather_loss, argnums=(0, 1))(disp, mesh)
    grad_gather_reference = jax.grad(gather_reference_loss, argnums=(0, 1))(disp, mesh)
    np.testing.assert_allclose(grad_gather[0], grad_gather_reference[0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(grad_gather[1], grad_gather_reference[1], rtol=2e-6, atol=2e-6)

    zero_mesh = jnp.zeros((4, 4, 4), dtype=conf.float_dtype)

    def scatter_loss(d, values):
        return jnp.sum(_scatter(pmid, d, conf, zero_mesh, values, 0, None)**2)

    def scatter_reference_loss(d, values):
        return jnp.sum(_scatter(pmid, d, reference, zero_mesh, values, 0, None)**2)

    scattered = _scatter(pmid, disp, conf, zero_mesh, val, 0, None)
    scattered_reference = _scatter(pmid, disp, reference, zero_mesh, val, 0, None)
    np.testing.assert_allclose(scattered, scattered_reference, rtol=2e-6, atol=2e-6)
    grad_scatter = jax.grad(scatter_loss, argnums=(0, 1))(disp, val)
    grad_scatter_reference = jax.grad(scatter_reference_loss, argnums=(0, 1))(disp, val)
    np.testing.assert_allclose(grad_scatter[0], grad_scatter_reference[0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(grad_scatter[1], grad_scatter_reference[1], rtol=2e-6, atol=2e-6)


def test_pallas_cic_explicit_mesh_cell_size():
    conf = _configuration()
    reference = conf.replace(pallas_cic=False)
    pmid, disp, _, val = _inputs(conf)
    mesh = jax.random.normal(jax.random.PRNGKey(9), (8, 8, 8), dtype=conf.float_dtype)
    offset = jnp.array([1.0, 0.0, 0.0], dtype=conf.float_dtype)

    actual = _gather(pmid, disp, conf, mesh, 0, offset, 0.5)
    expected = _gather(pmid, disp, reference, mesh, 0, offset, 0.5)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    actual_mesh = _scatter(pmid, disp, conf, jnp.zeros_like(mesh), val, offset, 0.5)
    expected_mesh = _scatter(pmid, disp, reference, jnp.zeros_like(mesh), val, offset, 0.5)
    np.testing.assert_allclose(actual_mesh, expected_mesh, rtol=3e-5, atol=3e-5)


def test_pallas_cic_masks_final_particle_tile():
    """The padded lanes in a non-full tile must not write or read output."""
    conf = _configuration()
    reference = conf.replace(pallas_cic=False)
    key = jax.random.PRNGKey(29)
    pmid = jax.random.randint(key, (5, 3), 0, 4, dtype=conf.pmid_dtype)
    disp = jax.random.uniform(key, (5, 3), minval=-0.45, maxval=0.45, dtype=conf.float_dtype)
    mesh = jax.random.normal(key, (4, 4, 4, 2), dtype=conf.float_dtype)
    values = jnp.arange(1, 6, dtype=conf.float_dtype)

    actual_gather = _gather(pmid, disp, conf, mesh, 0, 0, None)
    expected_gather = _gather(pmid, disp, reference, mesh, 0, 0, None)
    np.testing.assert_allclose(actual_gather, expected_gather, rtol=2e-6, atol=2e-6)

    base = jnp.ones((4, 4, 4), dtype=conf.float_dtype)
    actual_scatter = _scatter(pmid, disp, conf, base, values, 0, None)
    expected_scatter = _scatter(pmid, disp, reference, base, values, 0, None)
    np.testing.assert_allclose(actual_scatter, expected_scatter, rtol=2e-6, atol=2e-6)


def test_pallas_cic_validity_mask_and_three_channels():
    """Static-capacity padding must be invisible to forward and backward CIC."""
    conf = _configuration()
    reference = conf.replace(pallas_cic=False)
    key = jax.random.PRNGKey(41)
    pmid = jax.random.randint(key, (5, 3), 0, 4, dtype=conf.pmid_dtype)
    disp = jax.random.uniform(key, (5, 3), minval=-0.4, maxval=0.4, dtype=conf.float_dtype)
    valid = jnp.array([True, False, True, False, True])
    mesh = jax.random.normal(key, (4, 4, 4, 3), dtype=conf.float_dtype)
    values = jax.random.normal(key, (5, 3), dtype=conf.float_dtype)

    actual_gather = _gather(pmid, disp, conf, mesh, 0, 0, None, valid)
    expected_gather = _gather(pmid, disp, reference, mesh, 0, 0, None)
    expected_gather = jnp.where(valid[:, None], expected_gather, 0)
    np.testing.assert_allclose(actual_gather, expected_gather, rtol=3e-5, atol=3e-5)

    def gather_loss(d, m):
        return jnp.sum(_gather(pmid, d, conf, m, 0, 0, None, valid)**2)

    def gather_reference_loss(d, m):
        out = _gather(pmid, d, reference, m, 0, 0, None)
        return jnp.sum(jnp.where(valid[:, None], out, 0)**2)

    grad_actual = jax.grad(gather_loss, argnums=(0, 1))(disp, mesh)
    grad_expected = jax.grad(gather_reference_loss, argnums=(0, 1))(disp, mesh)
    np.testing.assert_allclose(grad_actual[0], grad_expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(grad_actual[1], grad_expected[1], rtol=3e-5, atol=3e-5)

    actual_scatter = _scatter(
        pmid, disp, conf, jnp.zeros((4, 4, 4, 3), dtype=conf.float_dtype), values, 0, None, valid,
    )
    expected_scatter = _scatter(
        pmid, disp, reference, jnp.zeros((4, 4, 4, 3), dtype=conf.float_dtype), jnp.where(valid[:, None], values, 0), 0,
        None,
    )
    np.testing.assert_allclose(actual_scatter, expected_scatter, rtol=3e-5, atol=3e-5)

    def scatter_loss(d, v):
        out = _scatter(pmid, d, conf, jnp.zeros((4, 4, 4, 3), dtype=conf.float_dtype), v, 0, None, valid, )
        return jnp.sum(out**2)

    def scatter_reference_loss(d, v):
        out = _scatter(
            pmid, d, reference, jnp.zeros((4, 4, 4, 3), dtype=conf.float_dtype), jnp.where(valid[:, None], v, 0), 0,
            None,
        )
        return jnp.sum(out**2)

    grad_actual = jax.grad(scatter_loss, argnums=(0, 1))(disp, values)
    grad_expected = jax.grad(scatter_reference_loss, argnums=(0, 1))(disp, values)
    np.testing.assert_allclose(grad_actual[0], grad_expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(grad_actual[1], grad_expected[1], rtol=3e-5, atol=3e-5)
