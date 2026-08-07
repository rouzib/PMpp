from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.core import Configuration
from pmpp.cosmology import SimpleLCDM
from pmpp.cosmology.boltzmann import (boltzmann, growth, linear_power, transfer, transfer_fit, transfer_integ, varlin, )


def _conf(**kwargs):
    return Configuration(
        10.0, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float64, cosmo_dtype=jnp.float64, pallas_cic=False,
        a_start=1 / 64, a_nbody_maxstep=1 / 64, **kwargs,
    )


def test_no_wiggle_transfer_is_finite_normalized_and_close_to_baryonic_fit_on_large_scales():
    full_conf = _conf(transfer_fit_nowiggle=False)
    smooth_conf = full_conf.replace(transfer_fit_nowiggle=True)
    cosmo = SimpleLCDM(full_conf)
    k = jnp.concatenate((jnp.asarray([0.0], dtype=jnp.float64), jnp.geomspace(1e-4, 0.5, 64)))
    baryonic = transfer_fit(k, cosmo, full_conf)
    smooth = transfer_fit(k, cosmo, smooth_conf)
    assert np.all(np.isfinite(np.asarray(smooth)))
    np.testing.assert_allclose(np.asarray(smooth[0]), 1, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(baryonic[0]), 1, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(smooth[1]), np.asarray(baryonic[1]), rtol=2e-3, atol=2e-6)
    assert np.all(np.diff(np.asarray(smooth)) <= 1e-12)


def test_missing_tables_disabled_components_and_scaled_variance_fail_or_scale_exactly():
    conf = _conf()
    empty = SimpleLCDM(conf)
    with pytest.raises(ValueError, match="Transfer table is empty"):
        transfer(jnp.asarray([0.1]), empty, conf)
    with pytest.raises(ValueError, match="Growth table is empty"):
        growth(jnp.asarray(0.5), empty, conf)
    with pytest.raises(ValueError, match="variance table is empty"):
        varlin(jnp.asarray(1.0), None, empty, conf)

    disabled = boltzmann(empty, conf, transfer=False, growth=False, varlin=False)
    assert disabled.transfer is None
    assert disabled.growth is None
    assert disabled.varlin is None

    complete = boltzmann(empty, conf)
    radius = jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float64)
    unscaled = varlin(radius, None, complete, conf)
    scaled = varlin(radius, jnp.float64(0.4), complete, conf)
    expected = unscaled * growth(jnp.float64(0.4), complete, conf)**2
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(expected), rtol=2e-14, atol=2e-14)


def test_unimplemented_transfer_backend_and_non_three_dimensional_power_fail_explicitly():
    conf = _conf().replace(transfer_fit=False)
    empty = SimpleLCDM(conf)
    with pytest.raises(NotImplementedError, match="TODO"):
        transfer_integ(empty, conf)
    with pytest.raises(NotImplementedError, match="TODO"):
        transfer(jnp.asarray([0.1]), empty.replace(transfer=jnp.ones_like(conf.transfer_k)), conf)
    with pytest.raises(ValueError, match="dim=2 not supported"):
        linear_power(jnp.asarray([0.1]), None, empty, SimpleNamespace(dim=2))
