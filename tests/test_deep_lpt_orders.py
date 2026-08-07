import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.core import Configuration
from pmpp.cosmology import SimpleLCDM
from pmpp.initial_conditions import lpt
from pmpp.initial_conditions.lpt import _L, _strain


def _conf(**kwargs):
    return Configuration(
        1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float64, cosmo_dtype=jnp.float64, pallas_cic=False,
        a_start=1 / 32, a_nbody_maxstep=1 / 32, **kwargs,
    )


def test_lpt_quadratic_source_is_cache_independent_symmetric_and_bilinear():
    cached = _conf(lpt_cache_strains=True)
    recompute = cached.replace(lpt_cache_strains=False)
    axis = jnp.arange(4, dtype=jnp.float64)
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    real_a = (
        jnp.sin(2 * jnp.pi * x / 4) + 0.3 * jnp.cos(2 * jnp.pi * (y + z) / 4) +
        0.17 * jnp.sin(2 * jnp.pi * (x + y) / 4)
    )
    real_b = (0.4 * jnp.cos(2 * jnp.pi * y / 4) - 0.21 * jnp.sin(2 * jnp.pi * (x + z) / 4))
    pot_a = jnp.fft.rfftn(real_a)
    pot_b = jnp.fft.rfftn(real_b)

    quadratic_cached = _L(cached.kvec_spacing, pot_a, None, cached)
    quadratic_recomputed = _L(recompute.kvec_spacing, pot_a, None, recompute)
    np.testing.assert_allclose(np.asarray(quadratic_cached), np.asarray(quadratic_recomputed), rtol=2e-14, atol=2e-14)

    cross_ab = _L(cached.kvec_spacing, pot_a, pot_b, cached)
    cross_ba = _L(cached.kvec_spacing, pot_b, pot_a, cached)
    np.testing.assert_allclose(np.asarray(cross_ab), np.asarray(cross_ba), rtol=2e-14, atol=2e-14)
    cross_recomputed = _L(recompute.kvec_spacing, pot_a, pot_b, recompute)
    np.testing.assert_allclose(np.asarray(cross_ab), np.asarray(cross_recomputed), rtol=2e-14, atol=2e-14)

    eps = jnp.float64(1e-5)
    plus = _L(cached.kvec_spacing, pot_a + eps * pot_b, None, cached)
    minus = _L(cached.kvec_spacing, pot_a - eps * pot_b, None, cached)
    finite_difference = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(np.asarray(finite_difference), np.asarray(2 * cross_ab), rtol=2e-10, atol=2e-10)

    diagonal = _strain(cached.kvec_spacing, 0, 0, pot_a, cached)
    expected_diagonal = jnp.fft.irfftn(-cached.kvec_spacing[0]**2 * pot_a)
    np.testing.assert_allclose(np.asarray(diagonal), np.asarray(expected_diagonal), rtol=2e-14, atol=2e-14)
    assert np.all(np.isfinite(np.asarray(_strain(cached.kvec_spacing, 0, 1, pot_a, cached))))


def test_lpt_order_zero_is_exact_identity_and_third_order_fails_explicitly():
    conf_zero = _conf(lpt_order=0)
    modes = jnp.zeros((4, 4, 3), dtype=jnp.complex128)
    particles = lpt(modes, SimpleLCDM(conf_zero), conf_zero)
    np.testing.assert_array_equal(np.asarray(particles.disp), 0)
    np.testing.assert_array_equal(np.asarray(particles.vel), 0)
    assert particles.acc is None

    conf_third = conf_zero.replace(lpt_order=3)
    seed_modes = jnp.fft.rfftn(
        jnp.sin(2 * jnp.pi * jnp.arange(4, dtype=jnp.float64)[:, None, None] / 4) * jnp.ones((1, 4, 4)),
    )
    with pytest.raises(NotImplementedError, match="TODO"):
        lpt(seed_modes, SimpleLCDM(conf_third), conf_third)
