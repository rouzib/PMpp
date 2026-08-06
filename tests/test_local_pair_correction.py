import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from pmpp.core import Configuration
from pmpp.corrections import (
    NBodyCorrection, apply_local_pair_correction, evaluate_local_pair_potential, evaluate_phase_space_residual,
    init_local_pair_correction, init_local_pair_phase_space_correction,
)
from pmpp.cosmology import SimpleLCDM
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.extras.quijote import gather_authoritative_particles
from pmpp.nbody import Particles, force_acceleration, integrator as steps_module

GPU_COUNT = len([device for device in jax.devices() if device.platform == "gpu"])


def _grid_particles(conf, perturb=False):
    positions = jnp.asarray(
        list(itertools.product(*(range(size) for size in conf.ptcl_grid_shape))), dtype=conf.float_dtype,
    )
    positions = positions * jnp.asarray(conf.ptcl_spacing, dtype=conf.float_dtype)
    if perturb:
        positions = positions.at[0].add(jnp.asarray([0.2, 0.1, -0.1], dtype=conf.float_dtype) * conf.ptcl_spacing)
    return Particles.from_ordered_pos(conf, positions, vel=jnp.zeros_like(positions), acc=jnp.zeros_like(positions), )


def test_zero_initialized_local_pair_is_exact_identity_for_mesh_ratios():
    for mesh_ratio in (1, 2):
        conf = Configuration(1.0, (4, 4, 4), mesh_shape=mesh_ratio, float_dtype=jnp.float32, )
        correction = init_local_pair_correction(
            jax.random.PRNGKey(mesh_ratio), conf=conf, channels=4, allow_missing_sigma8=True,
        )
        residual = apply_local_pair_correction(
            correction, jnp.asarray(1.0, dtype=conf.float_dtype), _grid_particles(conf, perturb=True), SimpleLCDM(conf),
            conf,
        )
        np.testing.assert_array_equal(np.asarray(residual), np.zeros(residual.shape, np.float32))


def test_local_pair_parameter_gradient_is_finite_and_nonzero():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    correction = init_local_pair_correction(
        jax.random.PRNGKey(0), conf=conf, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    particles = _grid_particles(conf, perturb=True)
    cosmo = SimpleLCDM(conf)

    def loss(model):
        residual = apply_local_pair_correction(model, 1.0, particles, cosmo, conf)
        return jnp.sum(residual * residual)

    value, gradient = jax.value_and_grad(loss)(correction)
    gradient_leaves = jax.tree_util.tree_leaves(gradient)
    assert np.isfinite(float(value)) and float(value) > 0
    assert gradient_leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in gradient_leaves)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in gradient_leaves)


def test_nbody_composite_dispatches_local_pair_force_and_gradients():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    particles = _grid_particles(conf, perturb=True)
    cosmo = SimpleLCDM(conf)
    local = init_local_pair_correction(
        jax.random.PRNGKey(19), conf=conf, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )

    def loss(model):
        acceleration = force_acceleration(1.0, particles, cosmo, conf, correction=NBodyCorrection(local_pair=model), )
        return jnp.sum(acceleration * acceleration)

    value, gradient = jax.value_and_grad(loss)(local)
    assert np.isfinite(float(value))
    leaves = jax.tree_util.tree_leaves(gradient)
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_one_local_model_runs_on_particle_grid_for_mesh_ratios_one_and_two():
    conf_one = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    conf_two = Configuration(1.0, (4, 4, 4), mesh_shape=2, float_dtype=jnp.float32)
    correction = init_local_pair_correction(
        jax.random.PRNGKey(23), conf=conf_one, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    outputs = []
    for conf in (conf_one, conf_two):
        residual = apply_local_pair_correction(
            correction, 0.8, _grid_particles(conf, perturb=True), SimpleLCDM(conf), conf,
        )
        assert residual.shape == (conf.ptcl_num, 3)
        assert np.all(np.isfinite(np.asarray(residual)))
        np.testing.assert_allclose(np.asarray(residual).sum(axis=0), 0.0, atol=2e-8)
        outputs.append(residual)
    assert float(jnp.max(jnp.abs(outputs[0]))) > 0
    assert float(jnp.max(jnp.abs(outputs[1]))) > 0


def test_local_pair_residual_is_integer_cell_translation_equivariant_and_mean_free():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    correction = init_local_pair_correction(
        jax.random.PRNGKey(3), conf=conf, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    particles = _grid_particles(conf, perturb=True)
    translated = Particles.from_ordered_pos(
        conf, (particles.pos(dtype=conf.float_dtype) + 1.0) % conf.box_size[0], vel=jnp.zeros_like(particles.disp),
        acc=jnp.zeros_like(particles.disp),
    )
    cosmo = SimpleLCDM(conf)
    residual = apply_local_pair_correction(correction, 1.0, particles, cosmo, conf)
    translated_residual = apply_local_pair_correction(correction, 1.0, translated, cosmo, conf)
    np.testing.assert_allclose(np.asarray(residual).sum(axis=0), 0.0, atol=2e-9)
    np.testing.assert_allclose(np.asarray(translated_residual), np.asarray(residual), atol=2e-6, rtol=2e-5, )


def test_local_pair_phase_head_is_mean_free_and_bounded():
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    particles = _grid_particles(conf, perturb=True)
    cosmo = SimpleLCDM(conf)
    local = init_local_pair_correction(
        jax.random.PRNGKey(11), conf=conf, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    phase = init_local_pair_phase_space_correction(jax.random.PRNGKey(12), channels=8, dtype=jnp.float32)
    params = dict(phase.params)
    params["output_bias"] = params["output_bias"].at[0].set(0.4).at[3].set(-0.2)
    phase = phase.replace(params=params)
    displacement, velocity = evaluate_phase_space_residual(
        phase, 0.8, particles, cosmo, conf, drift_scale=jnp.asarray(0.5, dtype=jnp.float32), local_pair=local,
    )
    np.testing.assert_allclose(np.asarray(displacement).mean(axis=0), 0.0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(velocity).mean(axis=0), 0.0, atol=2e-6)
    assert float(jnp.max(jnp.linalg.norm(displacement, axis=-1))) <= 0.25 + 1e-6
    # Velocity is bounded such that its displacement over this drift is at
    # most one quarter of a particle cell.
    assert float(jnp.max(jnp.linalg.norm(velocity * 0.5, axis=-1))) <= 0.25 + 1e-6


def test_owner_aligned_phase_context_keeps_local_parameter_gradients(monkeypatch):
    conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    particles = _grid_particles(conf, perturb=True)
    cosmo = SimpleLCDM(conf)
    local = init_local_pair_correction(
        jax.random.PRNGKey(41), conf=conf, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    phase = init_local_pair_phase_space_correction(jax.random.PRNGKey(42), channels=8, dtype=jnp.float32)
    phase_params = dict(phase.params)
    phase_params["output_bias"] = phase_params["output_bias"].at[0].set(0.5)
    phase = phase.replace(params=phase_params)
    monkeypatch.setattr(
        steps_module, "drift_factor",
        lambda a_vel, a_prev, a_next, cosmo, conf: jnp.asarray(0.2, dtype=conf.float_dtype),
    )
    weights = jnp.linspace(0.2, 1.1, particles.disp.size, dtype=conf.float_dtype).reshape(particles.disp.shape)

    def loss(model):
        out = steps_module.drift_for_force(
            0.5, 0.5, 0.6, particles, cosmo, conf, correction=NBodyCorrection(local_pair=model, phase_space=phase),
            apply_phase=True,
        )
        return jnp.sum(weights * out.disp)

    value, gradient = jax.value_and_grad(loss)(local)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert np.isfinite(float(value))
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)


@pytest.mark.skipif(GPU_COUNT < 2, reason="requires 2 GPUs")
def test_local_pair_potential_matches_two_gpu_mesh_halo_partition():
    shape = (8, 8, 8)
    conf_single = Configuration(1.0, shape, mesh_shape=1, float_dtype=jnp.float32)
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)
    conf_multi = Configuration(
        1.0, shape, mesh_shape=1, multigpu=MultiGPUConfiguration(compute_mesh=compute_mesh,
                                                                 mode="mesh_halo"), max_ptcl_per_slice=512,
        max_share_ptcl=256, max_halo_share_ptcl=256, max_share_gather_ptcl=256, float_dtype=jnp.float32,
    )
    correction = init_local_pair_correction(
        jax.random.PRNGKey(7), conf=conf_single, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    source = jnp.linspace(-0.7, 1.1, np.prod(shape), dtype=jnp.float32).reshape(shape)
    source_sharded = jax.device_put(source, NamedSharding(compute_mesh, P("gpus", None, None)))
    expected = evaluate_local_pair_potential(correction, source, 0.75, SimpleLCDM(conf_single), conf_single)
    actual = evaluate_local_pair_potential(correction, source_sharded, 0.75, SimpleLCDM(conf_multi), conf_multi)
    np.testing.assert_allclose(np.asarray(jax.device_get(actual)), np.asarray(expected), atol=2e-5, rtol=2e-5, )


@pytest.mark.skipif(GPU_COUNT < 2, reason="requires 2 GPUs")
def test_mesh_two_local_acceleration_matches_two_gpu_mesh_halo():
    shape = (8, 8, 8)
    conf_single = Configuration(1.0, shape, mesh_shape=2, float_dtype=jnp.float32)
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)
    conf_multi = Configuration(
        1.0, shape, mesh_shape=2, multigpu=MultiGPUConfiguration(compute_mesh=compute_mesh,
                                                                 mode="mesh_halo"), max_ptcl_per_slice=320,
        max_share_ptcl=256, max_halo_share_ptcl=256, max_share_gather_ptcl=512, float_dtype=jnp.float32,
    )
    positions = jnp.asarray(list(itertools.product(range(8), repeat=3)), dtype=jnp.float32)
    positions = positions + 0.08 * jax.random.normal(jax.random.PRNGKey(31), positions.shape)
    zeros = jnp.zeros_like(positions)
    particles_single = Particles.from_ordered_pos(conf_single, positions, vel=zeros, acc=zeros)
    particles_multi = Particles.from_ordered_pos(conf_multi, positions, vel=zeros, acc=zeros)
    correction = init_local_pair_correction(
        jax.random.PRNGKey(32), conf=conf_single, channels=4, output_init_scale=1e-2, allow_missing_sigma8=True,
    )
    residual_single = apply_local_pair_correction(
        correction, 0.8, particles_single, SimpleLCDM(conf_single), conf_single
    )
    residual_multi = apply_local_pair_correction(correction, 0.8, particles_multi, SimpleLCDM(conf_multi), conf_multi)
    dense_single = gather_authoritative_particles(
        particles_single.replace(acc=residual_single), conf_single
    ).acceleration
    dense_multi = gather_authoritative_particles(particles_multi.replace(acc=residual_multi), conf_multi).acceleration
    np.testing.assert_allclose(
        np.asarray(jax.device_get(dense_multi)), np.asarray(jax.device_get(dense_single)), atol=3e-5, rtol=3e-4,
    )


@pytest.mark.parametrize("mesh_ratio", [1, 2])
@pytest.mark.skipif(GPU_COUNT < 2, reason="requires 2 GPUs")
def test_nonzero_local_phase_cross_slab_matches_single_gpu(mesh_ratio, monkeypatch):
    """Phase features are cached before raw-drift particles leave their owner."""
    shape = (8, 8, 8)
    conf_single = Configuration(1.0, shape, mesh_shape=mesh_ratio, float_dtype=jnp.float32, )
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    compute_mesh = create_compute_mesh(gpu_devices)
    conf_multi = Configuration(
        1.0, shape, mesh_shape=mesh_ratio, multigpu=MultiGPUConfiguration(compute_mesh=compute_mesh,
                                                                          mode="mesh_halo"), max_ptcl_per_slice=384,
        max_share_ptcl=256, max_halo_share_ptcl=256, max_share_gather_ptcl=512, float_dtype=jnp.float32,
    )

    q = jnp.asarray(list(itertools.product(range(8), repeat=3)), dtype=jnp.float32)
    particle_id = jnp.arange(q.shape[0], dtype=jnp.float32)
    perturbation = jnp.stack(
        (jnp.zeros_like(particle_id), 0.06 * jnp.sin(0.37 * particle_id), 0.05 * jnp.cos(0.23 * particle_id),
         ), axis=-1,
    )
    positions = jnp.mod(q + perturbation, 8.0)
    velocity_x = jnp.where(q[:, 0] == 3, 1.2, 0.0)
    velocity_x = jnp.where(q[:, 0] == 4, -1.2, velocity_x)
    velocity = jnp.stack((velocity_x, jnp.zeros_like(velocity_x), jnp.zeros_like(velocity_x)), axis=-1, )
    acceleration = jnp.zeros_like(velocity)
    ptcl_single = Particles.from_ordered_pos(conf_single, positions, vel=velocity, acc=acceleration)
    ptcl_multi = Particles.from_ordered_pos(conf_multi, positions, vel=velocity, acc=acceleration)

    local = init_local_pair_correction(
        jax.random.PRNGKey(70 + mesh_ratio), conf=conf_single, channels=4, output_init_scale=1e-2,
        allow_missing_sigma8=True,
    )
    phase = init_local_pair_phase_space_correction(jax.random.PRNGKey(80 + mesh_ratio), channels=8, dtype=jnp.float32)
    phase_params = dict(phase.params)
    phase_params["output_bias"] = (phase_params["output_bias"].at[0].set(0.7).at[3].set(-0.4))
    correction = NBodyCorrection(local_pair=local, phase_space=phase.replace(params=phase_params), )
    monkeypatch.setattr(
        steps_module, "drift_factor",
        lambda a_vel, a_prev, a_next, cosmo, conf: jnp.asarray(1.0, dtype=conf.float_dtype),
    )

    out_single = steps_module.drift_for_force(
        0.5, 0.5, 0.6, ptcl_single, SimpleLCDM(conf_single), conf_single, correction=correction, apply_phase=True,
    )
    out_multi = steps_module.drift_for_force(
        0.5, 0.5, 0.6, ptcl_multi, SimpleLCDM(conf_multi), conf_multi, correction=correction, apply_phase=True,
    )
    dense_single = gather_authoritative_particles(out_single, conf_single)
    dense_multi = gather_authoritative_particles(out_multi, conf_multi)

    raw_position = jnp.mod(positions + velocity, conf_single.box_size[0])
    phase_shift = jnp.mod(
        dense_single.position - raw_position + 0.5 * conf_single.box_size[0], conf_single.box_size[0],
    ) - 0.5 * conf_single.box_size[0]
    assert float(jnp.max(jnp.linalg.norm(phase_shift, axis=-1))) > 1e-4
    np.testing.assert_array_equal(np.asarray(dense_multi.counts), 1)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(dense_multi.position)), np.asarray(jax.device_get(dense_single.position)), atol=2e-4,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(dense_multi.velocity)), np.asarray(jax.device_get(dense_single.velocity)), atol=2e-4,
        rtol=2e-5,
    )
