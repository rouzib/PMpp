import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from pmpp.core import AXIS_NAME, Configuration
from pmpp.cosmology import SimpleLCDM
from pmpp.distributed import MultiGPUConfiguration, create_compute_mesh
from pmpp.distributed.fft import _host_shape_product
from pmpp.initial_conditions import lpt
from pmpp.initial_conditions.lpt import (
    _L, _L_streaming_2lpt, _accumulate_low_memory_lpt_order, _low_memory_particle_grid,
    _low_memory_second_order_potential, lpt_low_memory,
)
from pmpp.nbody import Particles

try:
    GPU_DEVICES = tuple(jax.devices("gpu"))
except RuntimeError:
    GPU_DEVICES = ()


def _configuration(**kwargs):
    return Configuration(
        1.0, (4, 4, 4), mesh_shape=1, lpt_order=2, pallas_cic=False, float_dtype=jnp.float32, **kwargs,
    )


def _cosmology_with_growth(conf):
    growth = jnp.ones((2, 3, conf.growth_a.shape[0]), dtype=conf.cosmo_dtype)
    return SimpleLCDM(conf).replace(growth=growth)


def _test_modes(conf):
    axis = jnp.arange(conf.ptcl_grid_shape[0], dtype=conf.float_dtype)
    x, y, z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    real = jnp.sin(2 * jnp.pi *
                   (x + y) / conf.ptcl_grid_shape[0]) + 0.3 * jnp.cos(2 * jnp.pi * z / conf.ptcl_grid_shape[2], )
    if conf.compute_mesh is None:
        return jnp.fft.rfftn(real)
    real = jax.device_put(real, NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None)))
    return conf.mGPU_rfftn_transposed(real)


def test_large_shape_products_remain_host_width_with_x64_disabled():
    with jax.enable_x64(False):
        conf = Configuration(1.0, (2048, 2048, 2048), mesh_shape=1, pallas_cic=False)
    assert conf.ptcl_num == 8_589_934_592
    assert _host_shape_product((2048, 2048, 2048)) == 8_589_934_592


def test_public_single_word_raveled_ids_fail_closed_without_x64():
    with jax.enable_x64(False):
        conf = Configuration(1.0, (2048, 2048, 2048), mesh_shape=1, pallas_cic=False, max_ptcl_per_slice=1, )
        particles = Particles(
            conf, jnp.asarray([[0, 0, 0]], dtype=jnp.int16), jnp.zeros((1, 3), dtype=jnp.float32),
            unused_index=jnp.asarray([False]), halo_mask=jnp.asarray([False]),
        )
        with pytest.raises(OverflowError, match="two-limb"):
            particles.raveled_id()


def test_streaming_2lpt_and_forward_entrypoint_match_standard_path():
    conf = _configuration()
    cosmo = _cosmology_with_growth(conf)
    modes = _test_modes(conf)
    pot = -modes / jnp.where(sum(k**2 for k in conf.kvec_spacing) != 0, sum(k**2 for k in conf.kvec_spacing), 1)

    expected_source = _L(conf.kvec_spacing, pot, None, conf)
    actual_source = _L_streaming_2lpt(conf.kvec_spacing, pot, conf)
    np.testing.assert_allclose(np.asarray(actual_source), np.asarray(expected_source), rtol=1e-6, atol=1e-6)

    expected_grid = Particles.gen_grid(conf, vel=True)
    actual_grid = _low_memory_particle_grid(conf)
    for name in ("pmid", "disp", "vel", "unused_index", "halo_mask"):
        np.testing.assert_array_equal(np.asarray(getattr(actual_grid, name)), np.asarray(getattr(expected_grid, name)))

    expected = lpt(modes, cosmo, conf)
    actual = lpt_low_memory(modes, cosmo, conf)
    np.testing.assert_array_equal(np.asarray(actual.pmid), np.asarray(expected.pmid))
    np.testing.assert_array_equal(np.asarray(actual.unused_index), np.asarray(expected.unused_index))
    np.testing.assert_allclose(np.asarray(actual.disp), np.asarray(expected.disp), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actual.vel), np.asarray(expected.vel), rtol=1e-6, atol=1e-6)


def test_low_memory_lpt_donation_is_confirmed_by_compiled_alias_bytes():
    conf = _configuration()
    cosmo = _cosmology_with_growth(conf)
    modes = _test_modes(conf)
    particles = _low_memory_particle_grid(conf)
    potential = -modes / jnp.where(sum(k**2 for k in conf.kvec_spacing) != 0, sum(k**2 for k in conf.kvec_spacing), 1, )
    second = jax.jit(lambda value: _low_memory_second_order_potential(value, cosmo, conf), donate_argnums=(0, ),
                     ).lower(potential).compile().memory_analysis()
    assert second.alias_size_in_bytes >= potential.size * potential.dtype.itemsize

    accumulated = _accumulate_low_memory_lpt_order.lower(
        particles.disp, particles.vel, particles.unused_index, potential, jnp.float32(1), jnp.float32(1), conf,
    ).compile().memory_analysis()
    state_bytes = particles.disp.size * particles.disp.dtype.itemsize + particles.vel.size * particles.vel.dtype.itemsize
    assert accumulated.alias_size_in_bytes >= state_bytes


def test_low_memory_lpt_rejects_outer_jax_transformations():
    conf = _configuration()
    cosmo = _cosmology_with_growth(conf)
    modes = _test_modes(conf)
    transformed = jax.jit(lambda value: lpt_low_memory(value, cosmo, conf))
    with pytest.raises(NotImplementedError, match="forward-only"):
        transformed(modes)


@pytest.mark.skipif(len(GPU_DEVICES) < 2, reason="distributed LPT comparison requires exactly two available GPUs")
def test_two_gpu_low_memory_lpt_matches_standard_owned_slab_path():
    mesh = create_compute_mesh(GPU_DEVICES[:2])
    conf = Configuration(
        1.0, (8, 8, 8), mesh_shape=1, multigpu=MultiGPUConfiguration(compute_mesh=mesh,
                                                                     mode="mesh_halo"), max_ptcl_per_slice=320,
        max_share_ptcl=300, max_share_gather_ptcl=300, lpt_order=2, pallas_cic=False, float_dtype=jnp.float32,
    )
    cosmo = _cosmology_with_growth(conf)
    modes = _test_modes(conf)
    expected = lpt(modes, cosmo, conf)
    actual = lpt_low_memory(modes, cosmo, conf)

    for name in ("pmid", "unused_index", "halo_mask"):
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(getattr(actual, name))), np.asarray(jax.device_get(getattr(expected, name))),
        )
    for name in ("disp", "vel"):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(getattr(actual, name))), np.asarray(jax.device_get(getattr(expected, name))),
            rtol=1e-6, atol=1e-6,
        )
