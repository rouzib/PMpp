import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import pytest

from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.corrections import (
    apply_potential_correction,
    build_correction_optimizer,
    evaluate_high_k_softening,
    evaluate_mesh_potential_residual,
    evaluate_mesh_source_residual,
    evaluate_pm_window_compensation,
    evaluate_pgd_bandpass,
    evaluate_pgd_potential_transfer,
    evaluate_radial_potential_transfer,
    force_green_kernel,
    force_uses_interlacing,
    init_potential_correction,
    init_mesh_cnn_potential_correction,
    init_radial_potential_correction,
    sample_potential_transfer,
)
from pmpp.utils import create_compute_mesh


GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def test_radial_potential_correction_has_finite_transfer_at_init():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_radial_potential_correction(
        jax.random.PRNGKey(0),
        latent_size=8,
        n_knots=8,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    corrected = apply_potential_correction(pot, 1.0, cosmo, conf, correction)
    transfer = evaluate_radial_potential_transfer(correction, 1.0, cosmo, conf)

    assert np.all(np.isfinite(np.asarray(corrected)))
    assert np.all(np.isfinite(np.asarray(transfer)))
    np.testing.assert_allclose(np.asarray(transfer[0, 0, 0]), 1.0, atol=1e-6, rtol=1e-6)
    assert transfer.shape == pot.shape


def test_radial_potential_correction_changes_with_scale_factor():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_radial_potential_correction(
        jax.random.PRNGKey(0),
        latent_size=8,
        n_knots=8,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    transfer_early = evaluate_radial_potential_transfer(correction, 0.25, cosmo, conf)
    transfer_late = evaluate_radial_potential_transfer(correction, 1.0, cosmo, conf)

    np.testing.assert_allclose(np.asarray(transfer_early[0, 0, 0]), 1.0, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(transfer_late[0, 0, 0]), 1.0, atol=1e-6, rtol=1e-6)
    assert not np.allclose(np.asarray(transfer_early), np.asarray(transfer_late))


def test_sample_potential_transfer_uses_unit_radius_fraction():
    conf = Configuration(1.0, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_radial_potential_correction(
        jax.random.PRNGKey(0),
        latent_size=8,
        n_knots=8,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )
    radius = jnp.linspace(0.0, 1.0, 17, dtype=jnp.float32)
    transfer = sample_potential_transfer(correction, radius, 1.0, cosmo, conf)
    assert transfer.shape == radius.shape
    np.testing.assert_allclose(np.asarray(transfer[0]), 1.0, atol=1e-6, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(transfer)))


def test_build_correction_optimizer_updates_radial_params():
    correction = init_radial_potential_correction(
        jax.random.PRNGKey(0),
        latent_size=8,
        n_knots=8,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32),
    )
    optimizer = build_correction_optimizer(1e-3, optimizer_name="adamax")
    state = optimizer.init(correction)
    grads = jax.tree_util.tree_map(jnp.ones_like, correction)
    updates, state = optimizer.update(grads, state, correction)
    updated = jax.tree_util.tree_map(lambda p, u: p + u, correction, updates)
    leaves = jax.tree_util.tree_leaves(updated)
    assert any(hasattr(leaf, "shape") and leaf.size > 0 for leaf in leaves)


def test_mesh_cnn_potential_correction_is_identity_at_init():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_mesh_cnn_potential_correction(
        jax.random.PRNGKey(0),
        channels=4,
        depth=2,
        max_residual=0.2,
        output_init_scale=0.0,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    source = jnp.ones(conf.mesh_shape, dtype=conf.float_dtype)
    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    corrected = apply_potential_correction(pot, 1.0, cosmo, conf, correction, source_real=source)
    residual = evaluate_mesh_source_residual(correction, source, 1.0, cosmo, conf)

    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(corrected), np.asarray(pot), atol=1e-6, rtol=1e-6)


def test_mesh_cnn_init_starts_as_global_potential_rescaling():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_mesh_cnn_potential_correction(
        jax.random.PRNGKey(1),
        channels=4,
        depth=2,
        max_residual=0.2,
        output_init_scale=1e-2,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    source = jnp.linspace(0.1, 1.6, np.prod(conf.mesh_shape), dtype=conf.float_dtype).reshape(conf.mesh_shape)
    potential = jnp.linspace(-0.8, 0.7, np.prod(conf.mesh_shape), dtype=conf.float_dtype).reshape(conf.mesh_shape)
    residual = evaluate_mesh_potential_residual(correction, source, potential, 1.0, cosmo, conf)
    ratio = np.asarray(residual / potential)

    np.testing.assert_allclose(ratio, np.full_like(ratio, ratio.flat[0]), atol=1e-6, rtol=1e-6)


def test_build_correction_optimizer_updates_mesh_cnn_params():
    correction = init_mesh_cnn_potential_correction(
        jax.random.PRNGKey(0),
        channels=4,
        depth=2,
        max_residual=0.2,
        output_init_scale=0.0,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32),
    )
    optimizer = build_correction_optimizer(1e-3, optimizer_name="adamax")
    state = optimizer.init(correction)
    grads = jax.tree_util.tree_map(jnp.ones_like, correction)
    updates, state = optimizer.update(grads, state, correction)
    updated = jax.tree_util.tree_map(lambda p, u: p + u, correction, updates)
    leaves = jax.tree_util.tree_leaves(updated)
    assert any(hasattr(leaf, "shape") and leaf.size > 0 for leaf in leaves)


def test_combined_correction_matches_radial_when_cnn_is_identity():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    key = jax.random.PRNGKey(0)
    combined = init_potential_correction(
        key,
        model="combined",
        latent_size=8,
        n_knots=8,
        channels=4,
        depth=2,
        max_residual=0.2,
        output_init_scale=0.0,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    source = jnp.ones(conf.mesh_shape, dtype=conf.float_dtype)
    radial_only = apply_potential_correction(pot, 1.0, cosmo, conf, combined.radial)
    combined_out = apply_potential_correction(pot, 1.0, cosmo, conf, combined, source_real=source)

    np.testing.assert_allclose(np.asarray(combined_out), np.asarray(radial_only), atol=1e-6, rtol=1e-6)


def test_windowed_spline_combined_applies_window_and_radial_transfer():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_potential_correction(
        jax.random.PRNGKey(0),
        model="windowed_spline",
        latent_size=8,
        n_knots=8,
        window_alpha=0.25,
        window_max_gain=2.0,
        window_taper_start=0.5,
        window_taper_stop=1.0,
        interlacing=True,
        green_kernel="discrete_laplacian",
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    combined_out = apply_potential_correction(pot, 1.0, cosmo, conf, correction)
    manual = apply_potential_correction(pot, 1.0, cosmo, conf, correction.window)
    manual = apply_potential_correction(manual, 1.0, cosmo, conf, correction.radial)
    transfer = evaluate_radial_potential_transfer(correction, 1.0, cosmo, conf)
    expected_transfer = (
        evaluate_pm_window_compensation(correction.window, conf)
        * evaluate_radial_potential_transfer(correction.radial, 1.0, cosmo, conf)
    )

    np.testing.assert_allclose(np.asarray(combined_out), np.asarray(manual), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(transfer), np.asarray(expected_transfer), atol=1e-6, rtol=1e-6)
    assert force_uses_interlacing(correction)
    assert force_green_kernel(correction) == "discrete_laplacian"


def test_pgd_potential_correction_has_finite_band_limited_transfer():
    conf = Configuration(1.0, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_potential_correction(
        jax.random.PRNGKey(0),
        model="pgd",
        pgd_alpha0=0.25,
        pgd_A=0.0,
        pgd_B=0.0,
        pgd_kl=0.5,
        pgd_ks=2.0,
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((8, 8, 5), dtype=jnp.complex64)
    corrected = apply_potential_correction(pot, 1.0, cosmo, conf, correction)
    band = evaluate_pgd_bandpass(correction, 1.0, conf)
    transfer = evaluate_pgd_potential_transfer(correction, 1.0, conf)

    np.testing.assert_allclose(np.asarray(transfer[0, 0, 0]), 1.0, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(transfer), np.asarray(1.0 - 0.25 * band), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(corrected), np.asarray(pot * transfer), atol=1e-6, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(transfer)))


def test_windowed_pgd_combined_applies_product_transfer():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_potential_correction(
        jax.random.PRNGKey(0),
        model="windowed_pgd",
        window_alpha=0.25,
        window_max_gain=2.0,
        pgd_alpha0=-0.1,
        pgd_kl=0.5,
        pgd_ks=2.0,
        green_kernel="discrete_laplacian",
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    combined_out = apply_potential_correction(pot, 1.0, cosmo, conf, correction)
    expected_transfer = (
        evaluate_pm_window_compensation(correction.window, conf)
        * evaluate_pgd_potential_transfer(correction.pgd, 1.0, conf)
    )

    np.testing.assert_allclose(np.asarray(combined_out), np.asarray(pot * expected_transfer), atol=1e-6, rtol=1e-6)
    assert force_green_kernel(correction) == "discrete_laplacian"


def test_high_k_softening_is_identity_below_taper_and_damps_high_k():
    conf = Configuration(1.0, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_potential_correction(
        jax.random.PRNGKey(0),
        model="high_k_softening",
        softening_strength=0.2,
        softening_start=0.5,
        softening_stop=1.0,
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((8, 8, 5), dtype=jnp.complex64)
    transfer = evaluate_high_k_softening(correction, conf)
    corrected = apply_potential_correction(pot, 1.0, cosmo, conf, correction)

    np.testing.assert_allclose(np.asarray(transfer[0, 0, 0]), 1.0, atol=1e-6, rtol=1e-6)
    assert float(jnp.min(transfer)) < 1.0
    assert float(jnp.min(transfer)) >= 0.8 - 1e-6
    np.testing.assert_allclose(np.asarray(corrected), np.asarray(pot * transfer), atol=1e-6, rtol=1e-6)


def test_trainable_windowed_spline_pgd_has_trainable_scalar_leaves():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    cosmo = SimpleLCDM(conf)
    correction = init_potential_correction(
        jax.random.PRNGKey(0),
        model="trainable_windowed_spline_pgd",
        latent_size=8,
        n_knots=8,
        output_init_scale=0.0,
        window_alpha=0.48,
        window_max_gain=4.0,
        pgd_alpha0=0.0,
        pgd_kl=0.08,
        pgd_ks=0.4,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf,
    )

    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    transfer = evaluate_radial_potential_transfer(correction, 1.0, cosmo, conf)
    corrected = apply_potential_correction(pot, 1.0, cosmo, conf, correction)
    leaves = jax.tree_util.tree_leaves(correction)

    assert len([leaf for leaf in leaves if getattr(leaf, "shape", None) == ()]) >= 5
    assert np.all(np.isfinite(np.asarray(transfer)))
    assert np.all(np.isfinite(np.asarray(corrected)))


@pytest.mark.skipif(GPU_COUNT < 2, reason="requires 2 GPUs")
def test_mesh_cnn_potential_correction_matches_single_device_on_mesh_halo():
    ptcl_grid_shape = (8, 8, 8)
    conf_single = Configuration(1.0, ptcl_grid_shape, mesh_shape=1, float_dtype=jnp.float32)
    compute_mesh = create_compute_mesh([device for device in jax.devices() if device.platform == "gpu"][:2])
    conf_multi = Configuration(
        1.0,
        ptcl_grid_shape,
        mesh_shape=1,
        multigpu=MultiGPUConfiguration(compute_mesh=compute_mesh, mode="mesh_halo"),
        max_ptcl_per_slice=512,
        max_share_ptcl=256,
        max_halo_share_ptcl=256,
        max_share_gather_ptcl=256,
        float_dtype=jnp.float32,
    )
    cosmo_single = SimpleLCDM(conf_single)
    cosmo_multi = SimpleLCDM(conf_multi)
    correction = init_mesh_cnn_potential_correction(
        jax.random.PRNGKey(2),
        channels=4,
        depth=2,
        max_residual=0.2,
        output_init_scale=1e-2,
        allow_missing_sigma8=True,
        dtype=jnp.float32,
        conf=conf_single,
    )

    source = jnp.linspace(0.1, 1.6, np.prod(conf_single.mesh_shape), dtype=conf_single.float_dtype).reshape(conf_single.mesh_shape)
    potential = jnp.linspace(-0.8, 0.7, np.prod(conf_single.mesh_shape), dtype=conf_single.float_dtype).reshape(conf_single.mesh_shape)
    source_sharded = jax.device_put(source, NamedSharding(compute_mesh, P("gpus", None, None)))
    potential_sharded = jax.device_put(potential, NamedSharding(compute_mesh, P("gpus", None, None)))

    residual_single = evaluate_mesh_potential_residual(correction, source, potential, 1.0, cosmo_single, conf_single)
    residual_multi = evaluate_mesh_potential_residual(
        correction,
        source_sharded,
        potential_sharded,
        1.0,
        cosmo_multi,
        conf_multi,
    )

    np.testing.assert_allclose(
        np.asarray(jax.device_get(residual_multi)),
        np.asarray(residual_single),
        atol=1e-5,
        rtol=1e-5,
    )
