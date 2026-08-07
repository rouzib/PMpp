from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.core import Configuration
from pmpp.corrections import common
from pmpp.corrections import core as correction_core
from pmpp.corrections import local_pair as local_pair_module
from pmpp.corrections import nbody as nbody_correction
from pmpp.corrections.combined import CombinedPotentialCorrection
from pmpp.corrections.mesh_cnn import MeshCNNPotentialCorrection
from pmpp.corrections.pgd import PGDPotentialCorrection, TrainablePGDPotentialCorrection
from pmpp.corrections.radial import RadialPotentialCorrection
from pmpp.corrections.softening import HighKSofteningCorrection, evaluate_high_k_softening
from pmpp.corrections.window import PMWindowCompensationCorrection, TrainablePMWindowCompensationCorrection
from pmpp.cosmology import SimpleLCDM
from pmpp.nbody import Particles


def _conf(n=4):
    return Configuration(1.0, (n, n, n), mesh_shape=1, float_dtype=jnp.float32, cosmo_dtype=jnp.float32)


def _particles(conf):
    ptcl = Particles.gen_grid(conf)
    index = jnp.arange(ptcl.disp.shape[0], dtype=conf.float_dtype)
    disp = jnp.stack((0.02 * index, -0.01 * index, 0.03 * index), axis=-1)
    vel = jnp.stack((-0.03 * index, 0.01 * index, 0.02 * index), axis=-1)
    unused = jnp.zeros(index.shape, dtype=jnp.bool_).at[-1].set(True)
    halo = jnp.zeros(index.shape, dtype=jnp.bool_).at[-2].set(True)
    return ptcl.replace(disp=disp, vel=vel, acc=vel * 0.5, unused_index=unused, halo_mask=halo)


def test_optional_correction_dependencies_fail_with_targeted_messages(monkeypatch):
    monkeypatch.setattr(common, "hk", None)
    with pytest.raises(ImportError, match="haiku.*radial model.*dm-haiku"):
        common.require_haiku("radial model")
    monkeypatch.setattr(common, "hk", object())
    common.require_haiku("radial model")

    monkeypatch.setattr(common, "optax", None)
    with pytest.raises(ImportError, match="optax.*training"):
        common.require_optax("training")
    monkeypatch.setattr(common, "optax", object())
    common.require_optax("training")


def test_cosmology_conditioning_never_silently_accepts_missing_or_nonfinite_sigma8():
    dtype = jnp.float32
    np.testing.assert_allclose(common.resolve_sigma8(None, dtype), 0.8, atol=1e-7)
    cosmo = SimpleNamespace(Omega_m=0.31, sigma8=0.82)
    np.testing.assert_allclose(common.cosmo_features(cosmo, dtype), [0.31, 0.82], atol=1e-7)
    np.testing.assert_allclose(common.cosmo_features(jnp.asarray([0.29, 0.79]), dtype), [0.29, 0.79], atol=0)
    np.testing.assert_allclose(common.cosmo_features(None, dtype), [0.3, 0.8], atol=1e-7)

    missing = SimpleNamespace(Omega_m=0.31)
    with pytest.raises(ValueError, match="sigma8 is not initialized"):
        common.resolve_sigma8(missing, dtype)
    np.testing.assert_allclose(common.resolve_sigma8(missing, dtype, allow_missing_sigma8=True), 0.8, atol=1e-7)

    nonfinite = SimpleNamespace(Omega_m=0.31, sigma8=jnp.nan)
    with pytest.raises(ValueError, match="sigma8 is non-finite"):
        common.resolve_sigma8(nonfinite, dtype)
    np.testing.assert_allclose(common.resolve_sigma8(nonfinite, dtype, allow_missing_sigma8=True), 0.8, atol=1e-7)

    correction = SimpleNamespace(sigma8_value=0.91, allow_missing_sigma8=False)
    np.testing.assert_allclose(common.correction_cosmo_features(correction, None, dtype), [0.3, 0.91], atol=1e-7)
    np.testing.assert_allclose(
        common.correction_cosmo_features(correction, jnp.asarray([0.27, 0.77]), dtype), [0.27, 0.77], atol=0,
    )
    np.testing.assert_allclose(common.correction_cosmo_features(correction, cosmo, dtype), [0.31, 0.91], atol=1e-7)
    np.testing.assert_allclose(common.correction_cosmo_features(None, cosmo, dtype), [0.31, 0.82], atol=1e-7)


def test_optimizer_variants_clip_update_and_reject_unknown_algorithms(monkeypatch):
    optimizer = common.build_correction_optimizer(
        1e-2, gradient_clip_norm=None, optimizer_name="adam", apply_if_finite_steps=0
    )
    params = {"x": jnp.asarray([1.0, -2.0])}
    state = optimizer.init(params)
    updates, _ = optimizer.update({"x": jnp.asarray([0.5, -0.25])}, state, params)
    assert np.all(np.isfinite(np.asarray(updates["x"])))

    clipped = common.build_correction_optimizer(
        1e-2, gradient_clip_norm=0.1, optimizer_name="adamax", apply_if_finite_steps=2
    )
    state = clipped.init(params)
    updates, _ = clipped.update({"x": jnp.asarray([1e6, -1e6])}, state, params)
    assert np.max(np.abs(np.asarray(updates["x"]))) < 0.1

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        common.build_correction_optimizer(1e-3, optimizer_name="sgd")

    monkeypatch.setattr(common, "optax", None)
    with pytest.raises(ImportError, match="optimizer construction"):
        common.build_correction_optimizer(1e-3)


def test_correction_factory_exercises_omitted_model_families_with_valid_numerics():
    conf = _conf()
    key = jax.random.PRNGKey(4)
    small = {
        "conf": conf,
        "dtype": jnp.float32,
        "allow_missing_sigma8": True,
        "latent_size": 4,
        "n_knots": 4,
        "channels": 2,
        "depth": 1,
    }
    trainable_spline = correction_core.init_potential_correction(key, model="trainable_windowed_spline", **small)
    assert isinstance(trainable_spline.radial, RadialPotentialCorrection)
    assert isinstance(trainable_spline.window, TrainablePMWindowCompensationCorrection)

    trainable_pgd = correction_core.init_potential_correction(key, model="trainable_pgd", **small)
    assert isinstance(trainable_pgd, TrainablePGDPotentialCorrection)
    trainable_window_pgd = correction_core.init_potential_correction(key, model="trainable_windowed_pgd", **small)
    assert isinstance(trainable_window_pgd.window, TrainablePMWindowCompensationCorrection)
    assert isinstance(trainable_window_pgd.pgd, TrainablePGDPotentialCorrection)

    softened = correction_core.init_potential_correction(key, model="windowed_softening", strength=0.3, **small)
    assert isinstance(softened.window, PMWindowCompensationCorrection)
    assert isinstance(softened.softening, HighKSofteningCorrection)
    window = correction_core.init_potential_correction(key, model="pm_window", **small)
    assert isinstance(window, PMWindowCompensationCorrection)
    with pytest.raises(ValueError, match="Unsupported correction model"):
        correction_core.init_potential_correction(key, model="not-a-model", **small)

    mapped = correction_core._mesh_initializer_kwargs({"mesh_channels": 3, "mesh_depth": 2, "unchanged": 1})
    assert mapped["channels"] == 3 and mapped["depth"] == 2 and mapped["unchanged"] == 1


def test_transfer_dispatch_rejects_incompatible_correction_semantics():
    conf = _conf()
    cosmo = SimpleLCDM(conf)
    mesh = MeshCNNPotentialCorrection(params={}, dtype=jnp.float32)
    window = PMWindowCompensationCorrection(dtype=jnp.float32)
    pgd = PGDPotentialCorrection(dtype=jnp.float32)
    softening = HighKSofteningCorrection(dtype=jnp.float32)
    radius = jnp.linspace(0, 1, 5)

    for correction, message in ((CombinedPotentialCorrection(mesh_cnn=mesh), "does not define a radial"),
                                (mesh, "Mesh CNN"), (window, "anisotropic"), (pgd, "sampling is not implemented"),
                                (softening, "sampling is not implemented"), (object(), "Unsupported correction type"),
                                ):
        with pytest.raises(TypeError, match=message):
            correction_core.sample_potential_transfer(correction, radius, 1.0, cosmo, conf)

    for correction in (CombinedPotentialCorrection(mesh_cnn=mesh), mesh, object()):
        with pytest.raises(TypeError):
            correction_core.evaluate_radial_potential_transfer(correction, 1.0, cosmo, conf)

    np.testing.assert_allclose(
        correction_core.evaluate_radial_potential_transfer(window, 1.0, cosmo, conf),
        correction_core.evaluate_pm_window_compensation(window, conf), rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        correction_core.evaluate_radial_potential_transfer(pgd, 0.7, cosmo, conf),
        correction_core.evaluate_pgd_potential_transfer(pgd, 0.7, conf), rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        correction_core.evaluate_radial_potential_transfer(softening, 1.0, cosmo, conf),
        evaluate_high_k_softening(softening, conf), rtol=1e-6, atol=1e-6,
    )


def test_potential_application_requires_mesh_source_and_preserves_identity_and_aliases(monkeypatch):
    conf = _conf()
    mesh = MeshCNNPotentialCorrection(params={}, dtype=jnp.float32)
    pot = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    assert correction_core.apply_potential_correction(pot, None, None, conf, None) is pot
    with pytest.raises(ValueError, match="source_real is required"):
        correction_core.apply_potential_correction(pot, None, None, conf, mesh)

    source = jnp.arange(64, dtype=jnp.float32).reshape(4, 4, 4)
    fake_conf = SimpleNamespace(
        compute_mesh=object(), float_dtype=jnp.float32, mGPU_irfftn_transposed=lambda value: jnp.fft.irfftn(value),
        mGPU_rfftn_transposed=lambda value: jnp.fft.rfftn(value),
    )
    original_evaluate_mesh = correction_core.evaluate_mesh_potential_residual
    monkeypatch.setattr(
        correction_core, "evaluate_mesh_potential_residual",
        lambda correction, source_real, potential_real, a, cosmo, conf: jnp.zeros_like(potential_real),
    )
    corrected = correction_core.apply_potential_correction(pot, None, None, fake_conf, mesh, source_real=source)
    np.testing.assert_array_equal(corrected, pot)
    monkeypatch.setattr(correction_core, "evaluate_mesh_potential_residual", original_evaluate_mesh)

    combined = CombinedPotentialCorrection(mesh_cnn=mesh)
    monkeypatch.setattr(
        correction_core, "_evaluate_mesh_source_residual", lambda correction, source, a, cosmo, conf: source * 2,
    )
    monkeypatch.setattr(
        correction_core, "_evaluate_mesh_potential_residual",
        lambda correction, source, potential, a, cosmo, conf: potential * 3,
    )
    np.testing.assert_array_equal(
        correction_core.evaluate_mesh_source_residual(combined, source, 1, None, conf), source * 2,
    )
    np.testing.assert_array_equal(
        correction_core.evaluate_mesh_potential_residual(combined, source, source, 1, None, conf), source * 3,
    )


def _float0_leaf():
    return jax.grad(lambda value: jnp.sum(value.astype(jnp.float32)), allow_int=True)(jnp.asarray([1], jnp.int32))


def test_correction_cotangent_helpers_preserve_none_float0_and_sum_dynamic_leaves():
    float0 = _float0_leaf()
    tree = {"a": jnp.asarray([1.0, 2.0]), "b": None, "c": float0}
    zero = correction_core.zero_potential_correction_cotangent(tree)
    np.testing.assert_array_equal(zero["a"], 0)
    assert zero["b"] is None and zero["c"].dtype == jax.dtypes.float0
    assert correction_core.zero_potential_correction_cotangent(None) is None
    assert correction_core.add_potential_correction_cotangents(None, tree) is tree
    assert correction_core.add_potential_correction_cotangents(tree, None) is tree

    lhs = {"a": jnp.asarray([1.0]), "b": None, "c": float0, "d": jnp.asarray([4.0])}
    rhs = {"a": jnp.asarray([2.0]), "b": jnp.asarray([3.0]), "c": jnp.asarray([5.0]), "d": float0}
    added = correction_core.add_potential_correction_cotangents(lhs, rhs)
    np.testing.assert_array_equal(added["a"], [3.0])
    np.testing.assert_array_equal(added["b"], [3.0])
    np.testing.assert_array_equal(added["c"], [5.0])
    np.testing.assert_array_equal(added["d"], [4.0])

    nbody_zero = nbody_correction.zero_nbody_correction_cotangent(tree)
    np.testing.assert_array_equal(nbody_zero["a"], 0)
    assert nbody_correction.zero_nbody_correction_cotangent(None) is None
    assert nbody_correction.add_nbody_correction_cotangents(None, tree) is tree
    assert nbody_correction.add_nbody_correction_cotangents(tree, None) is tree
    nbody_added = nbody_correction.add_nbody_correction_cotangents(lhs, rhs)
    for name in added:
        if added[name] is None:
            assert nbody_added[name] is None
        else:
            np.testing.assert_array_equal(nbody_added[name], added[name])


def test_combined_force_kernel_dispatch_detects_incompatible_derivatives():
    spectral = SimpleNamespace(interlacing=False, green_kernel="continuum", gradient_kernel="spectral")
    discrete = SimpleNamespace(interlacing=True, green_kernel="discrete_laplacian", gradient_kernel="fastpm_4point")
    combined = CombinedPotentialCorrection(radial=spectral, window=discrete)
    assert correction_core.force_uses_interlacing(combined)
    assert correction_core.force_green_kernel(combined) == "discrete_laplacian"
    assert correction_core.force_gradient_kernel(combined) == "fastpm_4point"

    incompatible = CombinedPotentialCorrection(
        radial=SimpleNamespace(gradient_kernel="fastpm_4point"), window=SimpleNamespace(gradient_kernel="other"),
    )
    with pytest.raises(ValueError, match="incompatible force-gradient kernels"):
        correction_core.force_gradient_kernel(incompatible)


def test_phase_space_constructor_and_local_pair_protocol_reject_invalid_contracts():
    with pytest.raises(TypeError, match="apply_fn must be callable"):
        nbody_correction.BoundedPhaseSpaceCorrection(params={}, apply_fn=1)
    with pytest.raises(TypeError, match="context_fn must be callable"):
        nbody_correction.BoundedPhaseSpaceCorrection(params={}, context_fn=1)
    with pytest.raises(ValueError, match="non-negative"):
        nbody_correction.BoundedPhaseSpaceCorrection(params={}, max_displacement_cells=-1)
    with pytest.raises(ValueError, match="explicit inverse"):
        nbody_correction.BoundedPhaseSpaceCorrection(params={}, invertible=True)

    conf = _conf(2)
    ptcl = _particles(conf)
    zeros = nbody_correction.apply_local_pair_correction(None, 1, ptcl, None, conf)
    np.testing.assert_array_equal(zeros, np.zeros_like(ptcl.disp))

    class MethodCorrection:

        def acceleration_residual(self, a, ptcl, cosmo, conf):
            return jnp.ones_like(ptcl.disp) * a

    np.testing.assert_array_equal(
        nbody_correction.apply_local_pair_correction(MethodCorrection(), 2, ptcl, None, conf),
        np.ones_like(ptcl.disp) * 2,
    )
    np.testing.assert_array_equal(
        nbody_correction.apply_local_pair_correction(lambda a, ptcl, cosmo, conf: -ptcl.disp, 1, ptcl, None, conf),
        -ptcl.disp,
    )
    with pytest.raises(TypeError, match="must be callable"):
        nbody_correction.apply_local_pair_correction(object(), 1, ptcl, None, conf)
    with pytest.raises(ValueError, match="must match particle displacement"):
        nbody_correction.apply_local_pair_correction(lambda *args: jnp.zeros((1, 3)), 1, ptcl, None, conf)


def test_phase_space_separate_and_fused_reductions_match_with_masks_bounds_and_context():
    conf = _conf(2)
    ptcl = _particles(conf)
    cosmo = SimpleLCDM(conf)

    def head(params, a, ptcl, cosmo, conf, context):
        gain = params["gain"] + context["offset"]
        return gain * (ptcl.disp + ptcl.vel), gain * (ptcl.vel - ptcl.disp)

    context_calls = []

    def context_fn(a, ptcl, cosmo, conf, local_pair):
        context_calls.append(float(a))
        return {"offset": jnp.asarray(local_pair, dtype=ptcl.disp.dtype)}

    phase = nbody_correction.BoundedPhaseSpaceCorrection(
        params={"gain": 4.0}, apply_fn=head, context_fn=context_fn, max_displacement_cells=0.2, max_velocity_cells=0.3,
        mean_free=True,
    )
    context = nbody_correction.prepare_phase_space_context(phase, 0.6, ptcl, cosmo, conf, local_pair=1.0)
    assert context_calls == [0.6]
    assert nbody_correction.prepare_phase_space_context(None, 0.6, ptcl, cosmo, conf) is None
    phase_no_context = nbody_correction.init_bounded_phase_space_correction()
    assert nbody_correction.prepare_phase_space_context(phase_no_context, 0.6, ptcl, cosmo, conf) is None

    separate = nbody_correction.evaluate_phase_space_residual(
        phase, 0.6, ptcl, cosmo, conf, drift_scale=0.25, context=context,
    )
    fused = nbody_correction.evaluate_phase_space_residual(
        phase, 0.6, ptcl, cosmo, conf, drift_scale=0.25, context=context, reduction_backend="fused",
    )
    for expected, actual in zip(separate, fused):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(actual[-1], 0)
        np.testing.assert_allclose(np.asarray(actual[:-2].mean(axis=0)), 0, atol=2e-6)
    assert np.max(np.linalg.norm(np.asarray(fused[0]), axis=-1)) <= 0.2 * conf.ptcl_spacing + 2e-6
    assert np.max(np.linalg.norm(np.asarray(fused[1]) * 0.25, axis=-1)) <= 0.3 * conf.ptcl_spacing + 2e-6

    no_mean = phase.replace(mean_free=False)
    separate_no_mean = nbody_correction.evaluate_phase_space_residual(
        no_mean, 0.6, ptcl, cosmo, conf, drift_scale=0.0, context=context,
    )
    fused_no_mean = nbody_correction.evaluate_phase_space_residual_fused(
        no_mean, 0.6, ptcl, cosmo, conf, drift_scale=0.0, context=context,
    )
    for expected, actual in zip(separate_no_mean, fused_no_mean):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_phase_space_error_paths_accessors_and_identity_application():
    conf = _conf(2)
    ptcl = _particles(conf)
    phase = nbody_correction.init_bounded_phase_space_correction(dtype=jnp.float64)
    composite = nbody_correction.NBodyCorrection(long_range="long", local_pair="local", phase_space=phase)
    assert nbody_correction.long_range_correction(composite) == "long"
    assert nbody_correction.long_range_correction("legacy") == "legacy"
    assert nbody_correction.local_pair_correction(composite) == "local"
    assert nbody_correction.local_pair_correction("legacy") is None
    assert nbody_correction.phase_space_correction(composite) is phase
    assert nbody_correction.phase_space_correction("legacy") is None
    assert nbody_correction.has_phase_space_correction(composite)
    assert not nbody_correction.phase_space_is_invertible(composite)
    assert nbody_correction.phase_space_is_invertible(None)
    assert nbody_correction.apply_phase_space_correction(None, 1, ptcl, None, conf, 1) is ptcl

    zeros = nbody_correction.evaluate_phase_space_residual(None, 1, ptcl, None, conf, 1)
    np.testing.assert_array_equal(zeros[0], 0)
    np.testing.assert_array_equal(zeros[1], 0)
    fused_zeros = nbody_correction.evaluate_phase_space_residual_fused(None, 1, ptcl, None, conf, 1)
    np.testing.assert_array_equal(fused_zeros[0], 0)
    with pytest.raises(TypeError, match="BoundedPhaseSpaceCorrection"):
        nbody_correction.evaluate_phase_space_residual(object(), 1, ptcl, None, conf, 1)
    with pytest.raises(TypeError, match="BoundedPhaseSpaceCorrection"):
        nbody_correction.evaluate_phase_space_residual_fused(object(), 1, ptcl, None, conf, 1)
    with pytest.raises(ValueError, match="reduction_backend"):
        nbody_correction.evaluate_phase_space_residual(phase, 1, ptcl, None, conf, 1, reduction_backend="invalid")

    bad = phase.replace(apply_fn=lambda *args: (jnp.zeros((1, 3)), jnp.zeros((1, 3))))
    with pytest.raises(ValueError, match="must match particle displacement shape"):
        nbody_correction.evaluate_phase_space_residual(bad, 1, ptcl, None, conf, 1)
    with pytest.raises(ValueError, match="must match particle displacement shape"):
        nbody_correction.evaluate_phase_space_residual_fused(bad, 1, ptcl, None, conf, 1)


def test_local_pair_shape_refinement_and_phase_context_invariants():
    assert local_pair_module._mesh_refinement_factors(SimpleNamespace(mesh_shape=(4, 4, 4),
                                                                      ptcl_grid_shape=(4, 4, 4))) == (1, 1, 1)
    assert local_pair_module._mesh_refinement_factors(SimpleNamespace(mesh_shape=(8, 8, 8),
                                                                      ptcl_grid_shape=(4, 4, 4))) == (2, 2, 2)
    with pytest.raises(ValueError, match="integer force-mesh refinement"):
        local_pair_module._mesh_refinement_factors(SimpleNamespace(mesh_shape=(7, 8, 8), ptcl_grid_shape=(4, 4, 4)))
    with pytest.raises(ValueError, match="isotropic"):
        local_pair_module._mesh_refinement_factors(SimpleNamespace(mesh_shape=(4, 8, 4), ptcl_grid_shape=(4, 4, 4)))
    with pytest.raises(ValueError, match="ratios 1 and 2"):
        local_pair_module._mesh_refinement_factors(SimpleNamespace(mesh_shape=(12, 12, 12), ptcl_grid_shape=(4, 4, 4)))

    source = jnp.arange(8**3, dtype=jnp.float32).reshape(8, 8, 8)
    down = local_pair_module._downsample_force_source_to_particle_grid(source, (2, 2, 2))
    assert down.shape == (4, 4, 4)
    up = local_pair_module._upsample_particle_force_to_force_mesh(down[..., None].repeat(3, axis=-1), (2, 2, 2))
    assert up.shape == (8, 8, 8, 3)
    assert local_pair_module._downsample_force_source_to_particle_grid(source, (1, 1, 1)) is source

    conf = _conf(2)
    ptcl = _particles(conf)
    context = local_pair_module.prepare_local_pair_phase_context(1, ptcl, None, conf, None)
    np.testing.assert_array_equal(context.local_acceleration, 0)
    assert context.sigma8_value == pytest.approx(0.8)
    with pytest.raises(TypeError, match="acceleration_residual"):
        local_pair_module.prepare_local_pair_phase_context(1, ptcl, None, conf, object())
    with pytest.raises(ValueError, match="conf is required"):
        local_pair_module.init_local_pair_correction(jax.random.PRNGKey(0), conf=None)
    with pytest.raises(ValueError, match="channels must be positive"):
        local_pair_module.init_local_pair_phase_space_correction(jax.random.PRNGKey(0), channels=0)
    np.testing.assert_array_equal(
        local_pair_module.evaluate_local_pair_potential(None, jnp.ones((2, 2, 2)), 1, None, conf), 0
    )
    with pytest.raises(TypeError, match="requires LocalPairCorrection"):
        local_pair_module.evaluate_local_pair_potential(object(), jnp.ones((2, 2, 2)), 1, None, conf)


def test_softening_modes_are_bounded_and_unknown_mode_is_a_runtime_error():
    conf = _conf()
    linear = HighKSofteningCorrection(strength=0.7, start=0.2, stop=0.8, mode="linear", dtype=jnp.float32)
    transfer = evaluate_high_k_softening(linear, conf)
    assert transfer.shape == (4, 4, 3)
    assert np.min(np.asarray(transfer)) >= 0
    assert np.max(np.asarray(transfer)) <= 1
    invalid = HighKSofteningCorrection(mode="quadratic")
    with pytest.raises(ValueError, match="Unsupported high-k softening mode"):
        evaluate_high_k_softening(invalid, conf)


@pytest.mark.parametrize("correction_type", [PMWindowCompensationCorrection, TrainablePMWindowCompensationCorrection])
def test_window_corrections_reject_unknown_force_gradient_kernel(correction_type):
    kwargs = {"gradient_kernel": "bad"}
    if correction_type is TrainablePMWindowCompensationCorrection:
        kwargs.update(raw_alpha=jnp.asarray(0.0), raw_max_gain=jnp.asarray(0.0))
    with pytest.raises(ValueError, match="Unsupported force-gradient kernel"):
        correction_type(**kwargs)
