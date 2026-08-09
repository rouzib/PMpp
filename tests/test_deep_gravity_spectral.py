import importlib
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from pmpp.core import Configuration
from pmpp.corrections import PMWindowCompensationCorrection
from pmpp.cosmology import SimpleLCDM
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.nbody import Particles, gravity
from pmpp.nbody.gravity import (
    _acceleration_from_density_hat, _can_use_batched_gradient_fft, _density_hat_from_real,
    _gradient_meshes_from_potential, _gradient_meshes_from_spectral_components, _gravity_from_density,
    _gravity_mesh_fields_from_density, _gravity_potential_from_density, _laplace_replicated, _reduce_gather_disp_cot,
    _spectral_gradient_components, _spectral_gradient_components_from_density_hat,
    _spectral_gradient_components_from_potential, apply_particle_nyquist_filter, duplicate_slot_counts,
    get_discrete_k_squared_transposed, get_k_squared, get_k_squared_transposed, laplace, laplace_bwd, laplace_fwd,
    laplace_transposed, laplace_transposed_bwd, laplace_transposed_fwd, laplace_transposed_with_kernel, neg_grad,
    reduce_duplicate_slot_cot,
)


def _conf(shape=(4, 4, 4), **kwargs):
    return Configuration(1.25, shape, mesh_shape=1, float_dtype=jnp.float32, **kwargs)


def test_poisson_symbols_and_custom_vjps_match_closed_form_without_nan_zero_mode():
    conf = _conf()
    k2 = get_k_squared(conf.kvec, conf)
    k2_t = get_k_squared_transposed(conf.kvec, conf)
    k2_d = get_discrete_k_squared_transposed(conf.kvec, conf)
    np.testing.assert_allclose(np.asarray(k2), np.asarray(k2_t), atol=1e-7)
    assert float(k2[0, 0, 0]) == 0
    assert float(k2_d[0, 0, 0]) == 0
    assert np.all(np.asarray(k2_d) <= np.asarray(k2) + 2e-6)
    assert np.any(np.asarray(k2_d) < np.asarray(k2) - 1e-5)

    src = (jnp.arange(k2.size, dtype=jnp.float32).reshape(k2.shape) + 1).astype(jnp.complex64)
    expected = jnp.where(k2 != 0, -src / jnp.where(k2 == 0, 1, k2), 0)
    np.testing.assert_allclose(np.asarray(laplace(conf.kvec, src, conf)), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(np.asarray(laplace_transposed(conf.kvec, src, conf)), np.asarray(expected), rtol=2e-6)

    forward, residual = laplace_fwd(conf.kvec, src, conf, None)
    backward = laplace_bwd(residual, jnp.ones_like(src))[1]
    np.testing.assert_allclose(np.asarray(forward), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(np.asarray(backward), np.asarray(jnp.where(k2 != 0, -1 / k2, 0)), rtol=2e-6)
    forward_t, residual_t = laplace_transposed_fwd(conf.kvec, src, conf, None)
    backward_t = laplace_transposed_bwd(residual_t, jnp.ones_like(src))[1]
    np.testing.assert_allclose(np.asarray(forward_t), np.asarray(expected), rtol=2e-6)
    np.testing.assert_allclose(np.asarray(backward_t), np.asarray(backward), rtol=2e-6)

    def loss(real_src):
        potential = laplace(conf.kvec, real_src.astype(jnp.complex64), conf)
        return jnp.sum(jnp.square(jnp.abs(potential)))

    grad = jax.grad(loss)(src.real)
    expected_grad = jnp.where(k2 != 0, 2 * src.real / jnp.square(k2), 0)
    assert np.all(np.isfinite(np.asarray(grad)))
    np.testing.assert_allclose(np.asarray(grad), np.asarray(expected_grad), rtol=3e-5, atol=2e-6)


def test_poisson_and_gradient_kernel_variants_are_exact_and_reject_unknown_names():
    conf = _conf()
    k = jnp.asarray([0, jnp.pi / (2 * conf.cell_size), jnp.pi / conf.cell_size], dtype=conf.float_dtype)
    pot = jnp.asarray([1 + 2j, 2 - 1j, 3 + 4j], dtype=jnp.complex64)

    spectral = neg_grad(k, pot, conf.cell_size, "spectral")
    np.testing.assert_allclose(np.asarray(spectral[:2]), np.asarray(-1j * k[:2] * pot[:2]), rtol=2e-6)
    np.testing.assert_allclose(np.asarray(spectral[2]), 0, atol=1e-6)
    fastpm = neg_grad(k, pot, conf.cell_size, "fastpm_4point")
    symbol = (8 * jnp.sin(k * conf.cell_size) - jnp.sin(2 * k * conf.cell_size)) / (6 * conf.cell_size)
    np.testing.assert_allclose(np.asarray(fastpm), np.asarray(-1j * symbol * pot), rtol=2e-6, atol=1e-6)
    with pytest.raises(ValueError, match="force-gradient kernel"):
        neg_grad(k, pot, conf.cell_size, "not-a-kernel")

    src = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    continuum = laplace_transposed_with_kernel(conf.kvec, src, conf, "continuum")
    discrete = laplace_transposed_with_kernel(conf.kvec, src, conf, "discrete_laplacian")
    assert continuum[0, 0, 0] == 0
    assert discrete[0, 0, 0] == 0
    assert np.all(np.isfinite(np.asarray(discrete)))
    assert not np.allclose(np.asarray(continuum), np.asarray(discrete))
    with pytest.raises(ValueError, match="Green's function"):
        laplace_transposed_with_kernel(conf.kvec, src, conf, "not-a-kernel")


def test_particle_nyquist_filter_and_cached_spectral_operators_have_exact_support():
    conf = _conf(shape=(4, 4, 4))
    src = jnp.ones((4, 4, 3), dtype=jnp.complex64)
    masks = (
        jnp.asarray([1, 1, 0, 0], dtype=jnp.float32)[:, None, None], jnp.asarray([1, 0, 1, 0],
                                                                                 dtype=jnp.float32)[None, :, None],
        jnp.asarray([1, 1, 0], dtype=jnp.float32)[None, None, :],
    )
    filtered = apply_particle_nyquist_filter(src, masks)
    expected = np.zeros(src.shape, dtype=np.complex64)
    expected[:2, [0, 2], :2] = 1
    np.testing.assert_array_equal(np.asarray(filtered), expected)

    pot = (jnp.arange(src.size, dtype=jnp.float32).reshape(src.shape) + 1).astype(jnp.complex64)
    cached = _spectral_gradient_components_from_potential(pot, conf)
    fallback_conf = types.SimpleNamespace(kvec=conf.kvec, cell_size=conf.cell_size, neg_ik=None)
    fallback = _spectral_gradient_components_from_potential(pot, fallback_conf)
    np.testing.assert_allclose(np.asarray(cached), np.asarray(fallback), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(_spectral_gradient_components(pot, conf)), np.asarray(cached))

    fastpm = _spectral_gradient_components_from_potential(pot, conf, "fastpm_4point")
    manual = jnp.stack([neg_grad(k, pot, conf.cell_size, "fastpm_4point") for k in conf.kvec])
    np.testing.assert_allclose(np.asarray(fastpm), np.asarray(manual), rtol=2e-6, atol=2e-6)


def test_real_and_spectral_gravity_helpers_match_direct_fft_oracles():
    conf = _conf()
    axis = jnp.arange(4, dtype=conf.float_dtype)
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    density = 1 + jnp.float32(0.03) * (jnp.sin(2 * jnp.pi * x / 4) + jnp.cos(2 * jnp.pi * y / 4))
    omega_m = jnp.asarray(0.31, dtype=conf.float_dtype)

    dens_hat = _density_hat_from_real((density - 1) * (1.5 * omega_m), conf)
    np.testing.assert_allclose(
        np.asarray(dens_hat), np.asarray(jnp.fft.rfftn((density - 1) * (1.5 * omega_m))), rtol=2e-6, atol=2e-6,
    )
    potential = _gravity_potential_from_density(density, omega_m, conf)
    expected = laplace_transposed_with_kernel(conf.kvec, dens_hat, conf)
    np.testing.assert_allclose(np.asarray(potential), np.asarray(expected), rtol=2e-6, atol=2e-6)

    spectral_grads = _spectral_gradient_components_from_density_hat(dens_hat, conf)
    expected_grads = _spectral_gradient_components_from_potential(_laplace_replicated(conf.kvec, dens_hat, conf), conf)
    np.testing.assert_allclose(np.asarray(spectral_grads), np.asarray(expected_grads), rtol=2e-6, atol=2e-6)

    from_components = _gradient_meshes_from_spectral_components(spectral_grads, conf, use_batched=False)
    from_potential = _gradient_meshes_from_potential(potential, conf, use_batched=False)
    for component, oracle in zip(from_components, jnp.fft.irfftn(spectral_grads, axes=(1, 2, 3))):
        np.testing.assert_allclose(np.asarray(component), np.asarray(oracle), rtol=3e-6, atol=2e-6)
    for component, oracle in zip(from_potential, from_components):
        np.testing.assert_allclose(np.asarray(component), np.asarray(oracle), rtol=3e-6, atol=2e-6)

    mesh_fields = _gravity_mesh_fields_from_density(density, omega_m, conf)
    for component, oracle in zip(mesh_fields, from_components):
        np.testing.assert_allclose(np.asarray(component), np.asarray(oracle), rtol=3e-6, atol=2e-6)


def test_batched_and_distributed_fft_dispatches_preserve_component_order_and_dtype():
    spectral = jnp.stack([
        jnp.ones((2, 2, 2), dtype=jnp.complex64), 2 * jnp.ones((2, 2, 2), dtype=jnp.complex64),
        3 * jnp.ones((2, 2, 2), dtype=jnp.complex64),
    ])

    calls = []

    def batched(values):
        calls.append(("batched", values.shape))
        return values.real + jnp.arange(3, dtype=jnp.float32)[:, None, None, None]

    def scalar(value):
        calls.append(("scalar", value.shape))
        return value.real

    fake = types.SimpleNamespace(
        compute_mesh=object(), mGPU_irfftn_transposed_batched=batched, mGPU_irfftn_transposed=scalar,
        float_dtype=jnp.float32,
    )
    assert _can_use_batched_gradient_fft(fake)
    batched_out = _gradient_meshes_from_spectral_components(spectral, fake, use_batched=True)
    assert calls == [("batched", spectral.shape)]
    for index, component in enumerate(batched_out):
        np.testing.assert_allclose(np.asarray(component), 2 * index + 1)

    calls.clear()
    scalar_out = _gradient_meshes_from_spectral_components(spectral, fake, use_batched=False)
    assert calls == [("scalar", (2, 2, 2))] * 3
    for index, component in enumerate(scalar_out):
        np.testing.assert_allclose(np.asarray(component), index + 1)

    fake_no_batch = types.SimpleNamespace(compute_mesh=object(), mGPU_irfftn_transposed_batched=None)
    assert not _can_use_batched_gradient_fft(fake_no_batch)


def test_large_local_gradient_mesh_avoids_signed_32bit_batched_fft_overflow():
    calls = []

    def batched(values):
        calls.append(("batched", values.shape))
        return values.real

    def scalar(value):
        calls.append(("scalar", value.shape))
        return value.real

    # This is the per-GPU owned real-mesh shape for a 2048^3 force mesh on
    # eight devices. The three-component result has 3,221,225,472 elements,
    # exceeding INT32_MAX even before mesh halos are attached.
    large = types.SimpleNamespace(
        compute_mesh=object(), mGPU_irfftn_transposed_batched=batched, mGPU_irfftn_transposed=scalar,
        local_mesh_shape=(256, 2048, 2048), local_mesh_with_halo_shape=(258, 2048, 2048), dim=3,
        float_dtype=jnp.float32,
    )
    assert not _can_use_batched_gradient_fft(large)

    spectral = jnp.ones((3, 2, 2, 2), dtype=jnp.complex64)
    components = _gradient_meshes_from_spectral_components(spectral, large, use_batched=True)
    assert calls == [("scalar", (2, 2, 2))] * 3
    assert len(components) == 3

    # The guard is based on the halo-expanded array that is consumed by the
    # stacked gather, not only on the smaller owned FFT output.
    boundary_safe = types.SimpleNamespace(
        compute_mesh=object(), mGPU_irfftn_transposed_batched=batched, local_mesh_shape=(170, 2048, 2048),
        local_mesh_with_halo_shape=(170, 2048, 2048), dim=3,
    )
    boundary_unsafe = types.SimpleNamespace(
        compute_mesh=object(), mGPU_irfftn_transposed_batched=batched, local_mesh_shape=(171, 2048, 2048),
        local_mesh_with_halo_shape=(171, 2048, 2048), dim=3,
    )
    assert _can_use_batched_gradient_fft(boundary_safe)
    assert not _can_use_batched_gradient_fft(boundary_unsafe)


def test_large_density_fft_streams_before_constructing_component_stack(monkeypatch):
    gravity_module = importlib.import_module("pmpp.nbody.gravity")
    calls = []
    potential = jnp.asarray([3 + 4j], dtype=jnp.complex64)
    expected = jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float32)

    def laplace_stub(kvec, density, conf, kernel="continuum"):
        calls.append(("laplace", density.shape, kernel))
        return potential

    def streamed_stub(actual_potential, particles, conf):
        calls.append(("streamed", actual_potential is potential))
        return expected

    def forbidden_stack(*_args, **_kwargs):
        raise AssertionError("large-mesh gravity constructed a component stack")

    monkeypatch.setattr(gravity_module, "laplace_transposed_with_kernel", laplace_stub)
    monkeypatch.setattr(gravity_module, "_streamed_acceleration_from_potential", streamed_stub)
    monkeypatch.setattr(gravity_module, "_spectral_gradient_components_from_density_hat", forbidden_stack)

    large = types.SimpleNamespace(
        replicated_mesh=False, compute_mesh=object(), mGPU_irfftn_transposed_batched=lambda value: value,
        local_mesh_shape=(320, 2560, 2560), local_mesh_with_halo_shape=(322, 2560, 2560), dim=3, kvec=(),
    )
    actual = _acceleration_from_density_hat(jnp.ones((1, ), dtype=jnp.complex64), object(), large)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert calls == [("laplace", (1, ), "continuum"), ("streamed", True)]


def test_single_device_gravity_is_zero_for_uniform_lattice_and_sensitive_to_tiny_perturbations():
    conf = _conf()
    cosmo = SimpleLCDM(conf)
    grid = Particles.gen_grid(conf, vel=True)
    uniform = gravity(jnp.asarray(0.8, dtype=conf.float_dtype), grid, cosmo, conf)
    np.testing.assert_allclose(np.asarray(uniform), 0, atol=3e-7)

    displaced = grid.replace(disp=grid.disp.at[0, 0].set(jnp.float32(1e-3)))
    perturbed = gravity(jnp.asarray(0.8, dtype=conf.float_dtype), displaced, cosmo, conf)
    assert np.linalg.norm(np.asarray(perturbed)) > 1e-7
    assert np.linalg.norm(np.asarray(perturbed).sum(axis=0)) < 2e-5
    assert not np.array_equal(np.asarray(perturbed), np.asarray(uniform))

    density = jnp.ones(conf.mesh_shape, dtype=conf.float_dtype)
    from_density = _gravity_from_density(density, grid, cosmo, conf)
    np.testing.assert_allclose(np.asarray(from_density), 0, atol=3e-7)
    acceleration = _acceleration_from_density_hat(jnp.fft.rfftn(density - 1), grid, conf)
    np.testing.assert_allclose(np.asarray(acceleration), 0, atol=3e-7)

    np.testing.assert_allclose(np.asarray(duplicate_slot_counts(grid, conf)), 1)
    cot = jnp.arange(grid.disp.size, dtype=conf.float_dtype).reshape(grid.disp.shape)
    np.testing.assert_array_equal(np.asarray(_reduce_gather_disp_cot(grid.pmid, grid.disp, None, cot, conf)), cot)
    np.testing.assert_array_equal(np.asarray(reduce_duplicate_slot_cot(grid, cot, conf)), cot)


def test_interlaced_gravity_and_its_position_gradient_are_finite_and_nontrivial():
    conf = _conf()
    cosmo = SimpleLCDM(conf)
    grid = Particles.gen_grid(conf, vel=True)
    correction = PMWindowCompensationCorrection(
        alpha=0.0, max_gain=1.0, interlacing=True, green_kernel="continuum", gradient_kernel="spectral",
    )
    displaced = grid.replace(disp=grid.disp.at[0, 0].set(jnp.float32(0.07)))

    def loss(disp):
        ptcl = displaced.replace(disp=disp)
        acc = gravity(jnp.asarray(0.9, dtype=conf.float_dtype), ptcl, cosmo, conf, correction=correction)
        return jnp.sum(jnp.square(acc))

    value, grad = jax.value_and_grad(loss)(displaced.disp)
    assert float(value) > 0
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.linalg.norm(np.asarray(grad)) > 1e-8


def test_two_gpu_mesh_halo_gravity_matches_single_device_forward_and_gradient():
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("this distributed gravity regression requires two GPUs")
    single = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    multi = Configuration(
        1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices[:2]), mode="mesh_halo"),
        max_ptcl_per_slice=48, max_share_ptcl=16, max_halo_share_ptcl=24, max_share_gather_ptcl=24,
    )
    base = Particles.gen_grid(single, vel=True)
    disp = jax.random.uniform(
        jax.random.PRNGKey(317), base.disp.shape, minval=-0.12, maxval=0.12, dtype=single.float_dtype,
    )
    single_ptcl = base.replace(disp=disp)
    multi_ptcl = Particles.from_pmid(multi, base.pmid, disp, vel=base.vel)
    single_cosmo = SimpleLCDM(single)
    multi_cosmo = SimpleLCDM(multi)

    single_acc = gravity(jnp.asarray(0.7, dtype=single.float_dtype), single_ptcl, single_cosmo, single)
    multi_acc = gravity(jnp.asarray(0.7, dtype=multi.float_dtype), multi_ptcl, multi_cosmo, multi)
    active = ~np.asarray(multi_ptcl.unused_index)
    active_pos = np.asarray(multi_ptcl.pos(dtype=jnp.float32))[active]
    order = np.lexsort((active_pos[:, 2], active_pos[:, 1], active_pos[:, 0]))
    source_pos = np.asarray(single_ptcl.pos(dtype=jnp.float32))
    source_order = np.lexsort((source_pos[:, 2], source_pos[:, 1], source_pos[:, 0]))
    np.testing.assert_allclose(
        np.asarray(multi_acc)[active][order],
        np.asarray(single_acc)[source_order], rtol=3e-5, atol=3e-5
    )

    def single_loss(disp_value):
        return jnp.sum(jnp.square(gravity(0.7, single_ptcl.replace(disp=disp_value), single_cosmo, single)))

    def multi_loss(disp_value):
        candidate = multi_ptcl.replace(disp=disp_value)
        return jnp.sum(jnp.square(gravity(0.7, candidate, multi_cosmo, multi)[active]))

    grad_single = jax.grad(single_loss)(single_ptcl.disp)
    grad_multi = jax.grad(multi_loss)(multi_ptcl.disp)
    np.testing.assert_allclose(
        np.asarray(grad_multi)[active][order],
        np.asarray(grad_single)[source_order], rtol=2e-4, atol=2e-4
    )


def test_two_gpu_distributed_poisson_symbols_match_dense_float64_oracles():
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("this distributed spectral-symbol test requires two GPUs")
    single = Configuration(1.0, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float64, pallas_cic=False)
    multi = Configuration(
        1.0, (8, 8, 8), mesh_shape=1, float_dtype=jnp.float64, pallas_cic=False,
        multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices[:2]), mode="mesh_halo"),
        max_ptcl_per_slice=320, max_share_ptcl=96, max_halo_share_ptcl=160, max_share_gather_ptcl=96,
    )
    expected_k2 = get_k_squared(single.kvec, single)
    expected_transposed = get_k_squared_transposed(single.kvec, single)
    expected_discrete = get_discrete_k_squared_transposed(single.kvec, single)
    actual_k2 = get_k_squared(multi.kvec, multi)
    actual_transposed = get_k_squared_transposed(multi.kvec, multi)
    actual_discrete = get_discrete_k_squared_transposed(multi.kvec, multi)
    np.testing.assert_allclose(np.asarray(actual_k2), np.asarray(expected_k2), rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.asarray(actual_transposed), np.asarray(expected_transposed), rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.asarray(actual_discrete), np.asarray(expected_discrete), rtol=2e-15, atol=2e-15)
    assert actual_k2.sharding.spec == P("gpus", None, None)
    assert actual_transposed.sharding.spec == P(None, "gpus", None)
    assert actual_discrete.sharding.spec == P(None, "gpus", None)


def test_particle_halo_duplicate_reduction_matches_exact_host_grouped_sums():
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("this particle-halo transpose test requires two GPUs")
    conf = Configuration(
        1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(compute_mesh=create_compute_mesh(devices[:2]), mode="particle_halo"),
        max_ptcl_per_slice=64, max_share_ptcl=24, max_halo_share_ptcl=40, max_share_gather_ptcl=32,
    )
    base_conf = Configuration(1.0, (4, 4, 4), mesh_shape=1, float_dtype=jnp.float32)
    base = Particles.gen_grid(base_conf)
    phase = jnp.arange(base.disp.size, dtype=jnp.float32).reshape(base.disp.shape)
    displaced = jnp.sin(phase + 0.3) * jnp.float32(0.12)
    ptcl = Particles.from_pmid(conf, base.pmid, displaced)
    active = ~np.asarray(ptcl.unused_index)
    keys = np.asarray(((ptcl.pmid[:, 0] * 4 + ptcl.pmid[:, 1]) * 4 + ptcl.pmid[:, 2]))
    slot_cot = jnp.stack((keys.astype(np.float32) + 1, 2 * keys.astype(np.float32) - 3, -keys.astype(np.float32) / 7,
                          ), axis=1)
    slot_cot = jnp.asarray(slot_cot, dtype=jnp.float32)
    reduced = np.asarray(reduce_duplicate_slot_cot(ptcl, slot_cot, conf))
    counts = np.asarray(duplicate_slot_counts(ptcl, conf))

    for key in np.unique(keys[active]):
        slots = active & (keys == key)
        expected = np.asarray(slot_cot)[slots].sum(axis=0)
        np.testing.assert_allclose(reduced[slots], np.broadcast_to(expected, reduced[slots].shape), rtol=0, atol=0)
        np.testing.assert_array_equal(counts[slots], len(np.flatnonzero(slots)))
    np.testing.assert_array_equal(reduced[~active], 0)
    assert np.any(counts[active] > 1)
