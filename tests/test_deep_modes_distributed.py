import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from pmpp.core import Configuration
from pmpp.cosmology import Cosmology, boltzmann
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.initial_conditions import linear_modes, white_noise, white_noise_nested
from pmpp.initial_conditions.modes import get_k_magnitude, get_k_magnitude_transposed


def _confs():
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("this distributed initial-condition regression requires two GPUs")
    common = dict(
        ptcl_spacing=12.5, ptcl_grid_shape=(8, 8, 8), mesh_shape=1, float_dtype=jnp.float64, cosmo_dtype=jnp.float64,
        pallas_cic=False, a_start=1 / 64, a_nbody_maxstep=1 / 64,
    )
    single = Configuration(**common)
    multi = Configuration(
        **common, multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices[:2]), mode="mesh_halo"),
        max_ptcl_per_slice=320, max_share_ptcl=96, max_halo_share_ptcl=160, max_share_gather_ptcl=96,
    )
    return single, multi


def test_distributed_white_noise_variants_match_dense_seed_and_phase_contracts():
    single, multi = _confs()
    real_single = white_noise(9127, single, real=True)
    real_multi = white_noise(9127, multi, real=True)
    np.testing.assert_array_equal(np.asarray(real_multi), np.asarray(real_single))
    assert real_multi.sharding.spec == P("gpus")

    fixed_single = white_noise(9127, single, real=True, unit_abs=True)
    fixed_multi = white_noise(9127, multi, real=True, unit_abs=True)
    np.testing.assert_allclose(np.asarray(fixed_multi), np.asarray(fixed_single), rtol=2e-14, atol=2e-14)
    spectral_multi = white_noise(9127, multi, unit_abs=True)
    np.testing.assert_allclose(np.abs(np.asarray(spectral_multi)), 1, rtol=2e-15, atol=2e-15)

    nested_spectral = white_noise_nested(441, multi, unit_abs=True)
    nested_real = white_noise_nested(441, multi, real=True, unit_abs=True)
    np.testing.assert_allclose(np.abs(np.asarray(nested_spectral)), 1, rtol=2e-15, atol=2e-15)
    reconstructed = multi.mGPU_irfftn_transposed(nested_spectral)
    reconstructed *= jnp.sqrt(jnp.asarray(np.prod(multi.ptcl_grid_shape), dtype=reconstructed.dtype))
    np.testing.assert_allclose(np.asarray(nested_real), np.asarray(reconstructed), rtol=3e-14, atol=3e-14)


def test_distributed_mode_magnitudes_and_linear_realization_match_dense_forward_and_gradient():
    single, multi = _confs()
    k_single = get_k_magnitude(single.kvec_spacing, single)
    kt_single = get_k_magnitude_transposed(single.kvec_spacing, single)
    k_multi = get_k_magnitude(multi.kvec_spacing, multi)
    kt_multi = get_k_magnitude_transposed(multi.kvec_spacing, multi)
    np.testing.assert_allclose(np.asarray(k_multi), np.asarray(k_single), rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.asarray(kt_multi), np.asarray(kt_single), rtol=2e-15, atol=2e-15)
    assert k_multi.sharding.spec == P("gpus", None, None)
    assert kt_multi.sharding.spec == P(None, "gpus", None)

    cosmo_single = boltzmann(
        Cosmology.from_sigma8(single, sigma8=0.81, n_s=0.965, Omega_m=0.31, Omega_b=0.049, h=0.68), single
    )
    cosmo_multi = boltzmann(
        Cosmology.from_sigma8(multi, sigma8=0.81, n_s=0.965, Omega_m=0.31, Omega_b=0.049, h=0.68), multi
    )
    real_single = white_noise(71, single, real=True)
    real_multi = white_noise(71, multi, real=True)
    linear_single = linear_modes(real_single, cosmo_single, single, a=jnp.float64(0.4), real=True)
    linear_multi = linear_modes(real_multi, cosmo_multi, multi, a=jnp.float64(0.4), real=True)
    np.testing.assert_allclose(np.asarray(linear_multi), np.asarray(linear_single), rtol=2e-11, atol=2e-11)

    probe = jnp.cos(jnp.arange(linear_single.size, dtype=jnp.float64).reshape(linear_single.shape) / 17)
    single_loss = lambda modes: jnp.vdot(linear_modes(modes, cosmo_single, single, a=0.4, real=True), probe)
    multi_loss = lambda modes: jnp.vdot(linear_modes(modes, cosmo_multi, multi, a=0.4, real=True), probe)
    grad_single = jax.grad(single_loss)(real_single)
    grad_multi = jax.grad(multi_loss)(real_multi)
    np.testing.assert_allclose(np.asarray(grad_multi), np.asarray(grad_single), rtol=3e-11, atol=3e-11)
