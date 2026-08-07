import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmpp.cic.enmesh import enmesh
from pmpp.cic.scatter import _scatter
from pmpp.core import Configuration
from pmpp.nbody import Particles


def _conf(chunk_size):
    return Configuration(
        1.0, (2, 2, 2), mesh_shape=1, float_dtype=jnp.float64, chunk_size=chunk_size, pallas_cic=False,
    )


def test_non_pallas_chunked_scatter_matches_unchunked_forward_vjp_and_finite_difference():
    unchunked = _conf(100)
    chunked = _conf(3)
    ptcl = Particles.gen_grid(unchunked)
    sequence = jnp.arange(ptcl.disp.size, dtype=jnp.float64).reshape(ptcl.disp.shape)
    disp = jnp.sin(sequence + 0.37) * 0.17
    values = jnp.stack(
        (jnp.linspace(-0.4, 1.2, ptcl.disp.shape[0]), jnp.cos(jnp.arange(ptcl.disp.shape[0], dtype=jnp.float64))),
        axis=1
    )
    values = values.at[2].set(0)
    valid = jnp.asarray([True, True, True, False, True, True, True, True])
    offset = jnp.asarray([0.13, -0.21, 0.07], dtype=jnp.float64)
    weights = jnp.sin(jnp.arange(16, dtype=jnp.float64).reshape(2, 2, 2, 2) + 0.2)

    def loss(displacement, particle_values, conf):
        mesh = _scatter(ptcl.pmid, displacement, conf, None, particle_values, offset, jnp.float64(0.75), valid, )
        return jnp.sum(mesh * weights)

    value_full, grads_full = jax.value_and_grad(loss, argnums=(0, 1))(disp, values, unchunked)
    value_chunk, grads_chunk = jax.value_and_grad(loss, argnums=(0, 1))(disp, values, chunked)
    np.testing.assert_allclose(value_chunk, value_full, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(grads_chunk[0]), np.asarray(grads_full[0]), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(np.asarray(grads_chunk[1]), np.asarray(grads_full[1]), rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(np.asarray(grads_chunk[0][2]), 0)
    np.testing.assert_array_equal(np.asarray(grads_chunk[0][3]), 0)
    np.testing.assert_array_equal(np.asarray(grads_chunk[1][3]), 0)

    eps = jnp.float64(2e-6)
    disp_plus = disp.at[5, 1].add(eps)
    disp_minus = disp.at[5, 1].add(-eps)
    fd_disp = (loss(disp_plus, values, chunked) - loss(disp_minus, values, chunked)) / (2 * eps)
    np.testing.assert_allclose(fd_disp, grads_chunk[0][5, 1], rtol=2e-8, atol=2e-9)
    val_plus = values.at[6, 0].add(eps)
    val_minus = values.at[6, 0].add(-eps)
    fd_val = (loss(disp, val_plus, chunked) - loss(disp, val_minus, chunked)) / (2 * eps)
    np.testing.assert_allclose(fd_val, grads_chunk[1][6, 0], rtol=2e-8, atol=2e-9)


def test_non_pallas_scalar_density_path_and_shape_contracts_cover_chunk_edges():
    conf = _conf(4)
    ptcl = Particles.gen_grid(conf)
    density = _scatter(ptcl.pmid, ptcl.disp, conf, None, None, 0, None)
    np.testing.assert_allclose(np.asarray(density), 1, rtol=0, atol=0)
    assert float(density.sum()) == pytest.approx(conf.mesh_size)

    scalar_loss = lambda displacement: jnp.sum(
        _scatter(ptcl.pmid, displacement, conf, None, jnp.float64(0), 0, None)**2
    )
    np.testing.assert_array_equal(np.asarray(jax.grad(scalar_loss)(ptcl.disp)), 0)

    channel_values = jnp.ones((ptcl.pmid.shape[0], 2), dtype=jnp.float64)
    bad_mesh = jnp.zeros(conf.mesh_shape + (3, ), dtype=jnp.float64)
    with pytest.raises(ValueError, match="channel shape mismatch"):
        _scatter(ptcl.pmid, ptcl.disp, conf, bad_mesh, channel_values, 0, None)


def test_enmesh_general_resampling_branches_preserve_partition_and_gradient():
    pmid = jnp.asarray([[0, 0], [1, 1], [-1, 0]], dtype=jnp.int32)
    disp = jnp.asarray([[0.2, -0.3], [-0.15, 0.25], [0.1, -0.2]], dtype=jnp.float64)
    indices, fractions, gradients = enmesh(pmid, disp, 1.0, (2, 2), jnp.asarray([0.1, -0.2]), 0.5, (4, 4), True, )
    assert indices.shape == (3, 4, 2)
    np.testing.assert_allclose(np.asarray(fractions.sum(axis=1)), 1, rtol=0, atol=2e-15)
    np.testing.assert_allclose(np.asarray(gradients[:2].sum(axis=1)), 0, rtol=0, atol=2e-15)

    no_wrap_indices, no_wrap_fractions = enmesh(
        jnp.asarray([[0, 0]], dtype=jnp.int32), jnp.asarray([[-0.3, 0.2]], dtype=jnp.float64), 1.0, None,
        jnp.asarray([0.0, 0.0]), None, (2, 2), False,
    )
    assert np.any(np.asarray(no_wrap_indices) == 2)
    np.testing.assert_allclose(np.asarray(no_wrap_fractions.sum(axis=1)), 1, rtol=0, atol=2e-15)

    unbounded_indices, unbounded_fractions, unbounded_gradients = enmesh(
        pmid[:2], disp[:2], 1.0, None, jnp.asarray([0.13, -0.07]), 0.6, None, True,
    )
    assert np.all(np.isfinite(np.asarray(unbounded_indices)))
    np.testing.assert_allclose(np.asarray(unbounded_fractions.sum(axis=1)), 1, rtol=0, atol=2e-15)
    np.testing.assert_allclose(np.asarray(unbounded_gradients.sum(axis=1)), 0, rtol=0, atol=2e-15)
