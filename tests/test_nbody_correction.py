import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.core import Configuration
from pmpp.corrections import (
    BoundedPhaseSpaceCorrection, NBodyCorrection, apply_phase_space_correction, evaluate_phase_space_residual,
    init_bounded_phase_space_correction,
)
from pmpp.cosmology import SimpleLCDM
from pmpp.nbody import Particles, integrator as steps_module, nbody, solver as nbody_module

GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


class _LinearLocalPair(NamedTuple):
    gain: jax.Array

    def acceleration_residual(self, a, ptcl, cosmo, conf):
        del a, cosmo, conf
        centered = ptcl.disp - jnp.mean(ptcl.disp, axis=0, keepdims=True)
        return self.gain * centered


def _configuration(*, symp_splits=((0, 0.5), (1, 0.5)), steps=1):
    return Configuration(
        1.0, (2, 2, 2), mesh_shape=1, a_start=0.5, a_stop=0.5 + 0.1 * steps, a_nbody_maxstep=0.1,
        symp_splits=symp_splits, float_dtype=jnp.float32, cosmo_dtype=jnp.float32,
    )


def _particles(conf):
    pmid = jnp.asarray(list(itertools.product(range(2), repeat=3)), dtype=conf.pmid_dtype)
    index = jnp.arange(pmid.shape[0], dtype=conf.float_dtype)
    disp = jnp.stack((0.02 * index - 0.07, 0.01 * index, -0.015 * index), axis=-1)
    vel = jnp.stack((0.03 * index, -0.02 * index + 0.04, 0.01 * index), axis=-1)
    return Particles(conf, pmid, disp, vel=vel, acc=jnp.zeros_like(vel))


def _feature_phase_head(params, a, ptcl, cosmo, conf, local_pair):
    del a, conf, local_pair
    gain = params["gain"]
    omega_m = jnp.asarray(cosmo.Omega_m, dtype=ptcl.disp.dtype)
    return gain * omega_m * (ptcl.disp + 0.5 * ptcl.vel), gain * omega_m * (ptcl.vel - ptcl.disp)


def test_nbody_correction_is_a_dynamic_pytree_and_zero_phase_is_identity():
    conf = _configuration()
    ptcl = _particles(conf)
    phase = init_bounded_phase_space_correction(dtype=conf.float_dtype)
    correction = NBodyCorrection(phase_space=phase)

    leaves = jax.tree_util.tree_leaves(correction)
    assert len(leaves) == 2
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)

    corrected = apply_phase_space_correction(
        phase, 0.6, ptcl, SimpleLCDM(conf), conf, jnp.asarray(0.2, dtype=conf.float_dtype),
    )
    np.testing.assert_array_equal(np.asarray(corrected.disp), np.asarray(ptcl.disp))
    np.testing.assert_array_equal(np.asarray(corrected.vel), np.asarray(ptcl.vel))


def test_zero_composite_preserves_full_nbody_forward(monkeypatch):
    conf = _configuration()
    ptcl = _particles(conf)
    cosmo = SimpleLCDM(conf)
    monkeypatch.setattr(
        steps_module, "drift_factor",
        lambda a_vel, a_prev, a_next, cosmo, conf: jnp.asarray(0.2, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "kick_factor",
        lambda a_acc, a_prev, a_next, cosmo, conf: jnp.asarray(0.0, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "gravity", lambda a, force_ptcl, cosmo, conf, correction=None: jnp.zeros_like(force_ptcl.disp),
    )

    baseline = nbody(ptcl, cosmo, conf)
    empty_composite = nbody(ptcl, cosmo, conf, correction=NBodyCorrection())
    zero_phase = nbody(
        ptcl, cosmo, conf,
        correction=NBodyCorrection(phase_space=init_bounded_phase_space_correction(dtype=conf.float_dtype)),
    )
    for candidate in (empty_composite, zero_phase):
        np.testing.assert_array_equal(np.asarray(candidate.pmid), np.asarray(baseline.pmid))
        np.testing.assert_array_equal(np.asarray(candidate.disp), np.asarray(baseline.disp))
        np.testing.assert_array_equal(np.asarray(candidate.vel), np.asarray(baseline.vel))
        np.testing.assert_array_equal(np.asarray(candidate.acc), np.asarray(baseline.acc))


def test_phase_space_outputs_are_mean_free_and_cell_bounded():
    conf = _configuration()
    ptcl = _particles(conf)
    phase = BoundedPhaseSpaceCorrection(
        params={"gain": jnp.asarray(100.0, dtype=conf.float_dtype)}, apply_fn=_feature_phase_head,
        max_displacement_cells=0.25, max_velocity_cells=0.25, dtype=conf.float_dtype,
    )
    drift_scale = jnp.asarray(0.2, dtype=conf.float_dtype)

    disp_delta, vel_delta = evaluate_phase_space_residual(phase, 0.6, ptcl, SimpleLCDM(conf), conf, drift_scale, )

    np.testing.assert_allclose(np.asarray(disp_delta.mean(axis=0)), 0.0, atol=2e-7)
    np.testing.assert_allclose(np.asarray(vel_delta.mean(axis=0)), 0.0, atol=2e-7)
    assert float(jnp.max(jnp.linalg.norm(disp_delta, axis=-1))) <= 0.25 * conf.ptcl_spacing + 2e-7
    assert float(jnp.max(jnp.linalg.norm(vel_delta * drift_scale, axis=-1))) <= 0.25 * conf.ptcl_spacing + 2e-7


def test_zero_initialized_phase_head_has_finite_training_gradient():
    conf = _configuration()
    ptcl = _particles(conf)
    cosmo = SimpleLCDM(conf)
    weights = jnp.arange(ptcl.disp.size, dtype=conf.float_dtype).reshape(ptcl.disp.shape)

    def loss(gain):
        phase = BoundedPhaseSpaceCorrection(
            params={"gain": gain}, apply_fn=_feature_phase_head, dtype=conf.float_dtype,
        )
        disp_delta, vel_delta = evaluate_phase_space_residual(
            phase, 0.6, ptcl, cosmo, conf, jnp.asarray(0.2, dtype=conf.float_dtype),
        )
        return jnp.sum(weights * (disp_delta + 0.2 * vel_delta))

    gradient = jax.grad(loss)(jnp.asarray(0.0, dtype=conf.float_dtype))
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.abs(gradient)) > 0.0


def test_phase_correction_runs_once_after_last_raw_drift_before_force(monkeypatch):
    conf = _configuration(symp_splits=((0.4, 0.0), (0.6, 0.0), (0.0, 1.0)))
    ptcl = _particles(conf)
    cosmo = SimpleLCDM(conf)
    phase_inputs = []
    force_inputs = []

    def phase_head(params, a, phase_ptcl, cosmo, conf, local_pair):
        del params, a, cosmo, conf, local_pair
        phase_inputs.append(np.asarray(phase_ptcl.disp))
        return phase_ptcl.disp, jnp.zeros_like(phase_ptcl.vel)

    def fake_force_acceleration(a, force_ptcl, cosmo, conf, correction=None, *, streamed_gravity=False):
        del a, cosmo, conf, correction, streamed_gravity
        force_inputs.append(np.asarray(force_ptcl.disp))
        return jnp.zeros_like(force_ptcl.disp)

    monkeypatch.setattr(
        steps_module, "drift_factor",
        lambda a_vel, a_prev, a_next, cosmo, conf: jnp.asarray(a_next - a_prev, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "kick_factor",
        lambda a_acc, a_prev, a_next, cosmo, conf: jnp.asarray(0.0, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(steps_module, "force_acceleration", fake_force_acceleration)
    phase = BoundedPhaseSpaceCorrection(
        params={}, apply_fn=phase_head, max_displacement_cells=0.1, max_velocity_cells=0.0, mean_free=False,
        dtype=conf.float_dtype,
    )

    out = steps_module.integrate(
        conf.a_start, conf.a_stop, ptcl, cosmo, conf, correction=NBodyCorrection(phase_space=phase),
    )

    assert len(phase_inputs) == 1
    assert len(force_inputs) == 2
    np.testing.assert_allclose(force_inputs[-1], np.asarray(out.disp), atol=0, rtol=0)
    assert not np.array_equal(phase_inputs[0], force_inputs[-1])


def test_phase_enabled_nbody_gradients_use_exact_rematerialized_path(monkeypatch, request):
    conf = _configuration()
    ptcl = _particles(conf)
    cosmo = SimpleLCDM(conf)

    monkeypatch.setattr(
        steps_module, "drift_factor",
        lambda a_vel, a_prev, a_next, cosmo, conf: jnp.asarray(0.2, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "kick_factor",
        lambda a_acc, a_prev, a_next, cosmo, conf: jnp.asarray(0.0, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "gravity", lambda a, force_ptcl, cosmo, conf, correction=None: jnp.zeros_like(force_ptcl.disp),
    )

    def forbidden_reversible_adjoint(*args, **kwargs):
        raise AssertionError("phase-space gradients must not use nbody_adj")

    monkeypatch.setattr(nbody_module, "nbody_adj", forbidden_reversible_adjoint)

    # Compile the transformed solver from the controls installed above. JAX
    # 0.9.1 can otherwise retain an earlier solver trace while coverage keeps
    # the associated Python frame alive. Clear again after the test so the
    # monkeypatched trace cannot affect later scientific regressions.
    jax.clear_caches()
    request.addfinalizer(jax.clear_caches)

    def loss(gain, disp, omega_m):
        phase = BoundedPhaseSpaceCorrection(
            params={"gain": gain}, apply_fn=_feature_phase_head, max_displacement_cells=0.25, max_velocity_cells=0.25,
            dtype=conf.float_dtype,
        )
        out = nbody(
            ptcl.replace(disp=disp), cosmo.replace(Omega_m=omega_m), conf,
            correction=NBodyCorrection(phase_space=phase),
        )
        weights = jnp.linspace(0.2, 1.1, out.disp.size, dtype=out.disp.dtype).reshape(out.disp.shape)
        return jnp.sum(weights * out.disp)

    gain = jnp.asarray(0.2, dtype=conf.float_dtype)
    omega_m = cosmo.Omega_m
    autodiff_gain, autodiff_disp, autodiff_omega = jax.grad(loss, argnums=(0, 1, 2), )(gain, ptcl.disp, omega_m)
    eps = jnp.asarray(2e-3, dtype=conf.float_dtype)
    finite_gain = (loss(gain + eps, ptcl.disp, omega_m) - loss(gain - eps, ptcl.disp, omega_m)) / (2 * eps)
    disp_plus = ptcl.disp.at[0, 0].add(eps)
    disp_minus = ptcl.disp.at[0, 0].add(-eps)
    finite_disp = (loss(gain, disp_plus, omega_m) - loss(gain, disp_minus, omega_m)) / (2 * eps)
    finite_omega = (loss(gain, ptcl.disp, omega_m + eps) - loss(gain, ptcl.disp, omega_m - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(autodiff_gain), np.asarray(finite_gain), rtol=3e-3, atol=3e-4)
    np.testing.assert_allclose(np.asarray(autodiff_disp[0, 0]), np.asarray(finite_disp), rtol=3e-3, atol=3e-4)
    np.testing.assert_allclose(np.asarray(autodiff_omega), np.asarray(finite_omega), rtol=3e-3, atol=3e-4)


def test_force_only_composite_keeps_fast_adjoint_and_matches_naive_grad(monkeypatch):
    conf = _configuration()
    ptcl = _particles(conf)
    cosmo = SimpleLCDM(conf)
    monkeypatch.setattr(
        steps_module, "drift_factor",
        lambda a_vel, a_prev, a_next, cosmo, conf: jnp.asarray(0.2, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "kick_factor",
        lambda a_acc, a_prev, a_next, cosmo, conf: jnp.asarray(0.15, dtype=conf.float_dtype),
    )
    monkeypatch.setattr(
        steps_module, "gravity", lambda a, force_ptcl, cosmo, conf, correction=None: jnp.zeros_like(force_ptcl.disp),
    )
    weights = jnp.linspace(0.3, 1.2, ptcl.disp.size, dtype=conf.float_dtype).reshape(ptcl.disp.shape)

    def correction_for(gain):
        return NBodyCorrection(local_pair=_LinearLocalPair(gain))

    def custom_loss(gain):
        return jnp.sum(weights * nbody(ptcl, cosmo, conf, correction=correction_for(gain)).disp)

    def naive_loss(gain):
        out = nbody_module._nbody_impl(ptcl, cosmo, conf, correction=correction_for(gain))
        return jnp.sum(weights * out.disp)

    gain = jnp.asarray(0.1, dtype=conf.float_dtype)
    custom_grad = jax.grad(custom_loss)(gain)
    naive_grad = jax.grad(naive_loss)(gain)
    np.testing.assert_allclose(np.asarray(custom_grad), np.asarray(naive_grad), rtol=2e-5, atol=2e-6)


def test_reverse_rejects_noninvertible_phase_space_correction():
    conf = _configuration()
    correction = NBodyCorrection(phase_space=init_bounded_phase_space_correction(dtype=conf.float_dtype))
    with pytest.raises(ValueError, match="non-invertible phase-space correction"):
        nbody(_particles(conf), SimpleLCDM(conf), conf, reverse=True, correction=correction)


def test_phase_space_cannot_claim_invertibility_without_an_inverse_protocol():
    with pytest.raises(ValueError, match="explicit inverse phase-map protocol"):
        init_bounded_phase_space_correction(invertible=True)


@pytest.mark.skipif(GPU_COUNT < 2, reason="requires 2 GPUs")
def test_zero_phase_preserves_cross_slab_drift_layout():
    from test_grad_drift import _build_crossing_state

    conf, _, cosmo, _, ptcl, _, a_vel, a_prev, a_next = _build_crossing_state()
    baseline = steps_module.drift_for_force(a_vel, a_prev, a_next, ptcl, cosmo, conf, )
    corrected = steps_module.drift_for_force(
        a_vel, a_prev, a_next, ptcl, cosmo, conf,
        correction=NBodyCorrection(phase_space=init_bounded_phase_space_correction(dtype=conf.float_dtype)),
        apply_phase=True,
    )
    for baseline_field, corrected_field in zip(
        (baseline.pmid, baseline.disp, baseline.vel, baseline.acc, baseline.unused_index, baseline.halo_mask),
        (corrected.pmid, corrected.disp, corrected.vel, corrected.acc, corrected.unused_index, corrected.halo_mask),
    ):
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(corrected_field)), np.asarray(jax.device_get(baseline_field)),
        )
