import numpy as np
import jax
import jax.numpy as jnp
import Pk_library as PKL
import MAS_library as MASL

from src.configuration import Configuration
from src.particles import Particles
from src.power_spectrum import delta_to_cross_correlation, delta_to_pk, particles_to_pk


def _reference_pk(delta, box_size, mas="CIC"):
    pk = PKL.Pk(np.array(delta, dtype=np.float32, copy=True, order="C"), float(box_size), 0, mas, 1, False)
    return (
        np.asarray(pk.k3D, dtype=np.float64),
        np.asarray(pk.Pk[:, 0], dtype=np.float64),
        np.asarray(pk.Nmodes3D, dtype=np.int32),
    )


def _assert_matches_reference(k, pk, k_ref, pk_ref, nmodes_ref):
    np.testing.assert_allclose(np.asarray(k), k_ref, rtol=5e-3, atol=2e-3)

    pk = np.asarray(pk)
    dense = nmodes_ref > 200
    medium = (nmodes_ref > 20) & ~dense
    sparse = ~dense & ~medium

    if np.any(dense):
        np.testing.assert_allclose(pk[dense], pk_ref[dense], rtol=3e-3, atol=5e-6)
    if np.any(medium):
        np.testing.assert_allclose(pk[medium], pk_ref[medium], rtol=3e-2, atol=1e-4)
    if np.any(sparse):
        np.testing.assert_allclose(pk[sparse], pk_ref[sparse], rtol=0.2, atol=1e-4)


def _directional_fd_check(loss_fn, x0, direction, eps, *, rtol, atol):
    direction = jnp.asarray(direction, dtype=x0.dtype)
    direction /= jnp.linalg.norm(direction)

    grad = jax.grad(loss_fn)(x0)
    analytic = jnp.vdot(grad, direction)
    finite_diff = (loss_fn(x0 + eps * direction) - loss_fn(x0 - eps * direction)) / (2 * eps)

    np.testing.assert_allclose(np.asarray(analytic), np.asarray(finite_diff), rtol=rtol, atol=atol)


def test_delta_to_pk_matches_pkl_on_small_random_field():
    box_size = 25.0
    nmesh = 32
    conf = Configuration(
        ptcl_spacing=box_size / nmesh,
        ptcl_grid_shape=(nmesh, nmesh, nmesh),
        mesh_shape=1,
        float_dtype=jnp.float32,
    )

    rng = np.random.default_rng(0)
    delta = rng.normal(size=(nmesh, nmesh, nmesh)).astype(np.float32)
    delta -= delta.mean(dtype=np.float64)

    k, pk, _ = delta_to_pk(jnp.asarray(delta), conf, mas="CIC")
    k_ref, pk_ref, nmodes_ref = _reference_pk(delta, box_size, mas="CIC")

    _assert_matches_reference(k, pk, k_ref, pk_ref, nmodes_ref)


def test_delta_to_cross_correlation_identity_and_sign():
    box_size = 25.0
    nmesh = 16
    conf = Configuration(
        ptcl_spacing=box_size / nmesh,
        ptcl_grid_shape=(nmesh, nmesh, nmesh),
        mesh_shape=1,
        float_dtype=jnp.float32,
    )

    rng = np.random.default_rng(4)
    delta = rng.normal(size=(nmesh, nmesh, nmesh)).astype(np.float32)
    delta -= delta.mean(dtype=np.float64)

    _, r_same, pk_cross_same, pk_a_same, pk_b_same, _ = delta_to_cross_correlation(
        jnp.asarray(delta),
        jnp.asarray(delta),
        conf,
        mas="CIC",
    )
    _, r_opposite, pk_cross_opposite, pk_a_opposite, pk_b_opposite, _ = delta_to_cross_correlation(
        jnp.asarray(delta),
        -jnp.asarray(delta),
        conf,
        mas="CIC",
    )

    np.testing.assert_allclose(np.asarray(r_same), 1.0, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(pk_cross_same), np.asarray(pk_a_same), atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pk_a_same), np.asarray(pk_b_same), atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(r_opposite), -1.0, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(pk_cross_opposite), -np.asarray(pk_a_opposite), atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pk_a_opposite), np.asarray(pk_b_opposite), atol=1e-5, rtol=1e-6)


def test_particles_to_pk_matches_scatter_plus_pkl():
    box_size = 25.0
    nmesh = 16
    conf = Configuration(
        ptcl_spacing=box_size / nmesh,
        ptcl_grid_shape=(nmesh, nmesh, nmesh),
        mesh_shape=1,
        float_dtype=jnp.float32,
    )

    rng = np.random.default_rng(1)
    positions = rng.uniform(0.0, box_size, size=(nmesh ** 3, 3)).astype(np.float32)
    ptcl = Particles.from_pos(conf, jnp.asarray(positions))

    k, pk, _ = particles_to_pk(ptcl, conf, mas="CIC")

    density = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    MASL.MA(np.array(positions, dtype=np.float32, copy=True, order="C"), density, box_size, "CIC", verbose=False)
    density /= density.mean(dtype=np.float64)
    delta = density - 1.0
    k_ref, pk_ref, nmodes_ref = _reference_pk(delta, box_size, mas="CIC")

    _assert_matches_reference(k, pk, k_ref, pk_ref, nmodes_ref)


def test_delta_to_pk_directional_derivative_matches_finite_difference():
    box_size = 25.0
    nmesh = 8
    conf = Configuration(
        ptcl_spacing=box_size / nmesh,
        ptcl_grid_shape=(nmesh, nmesh, nmesh),
        mesh_shape=1,
        float_dtype=jnp.float32,
    )

    rng = np.random.default_rng(2)
    delta0 = rng.normal(size=(nmesh, nmesh, nmesh)).astype(np.float32)
    delta0 -= delta0.mean(dtype=np.float64)
    direction = rng.normal(size=delta0.shape).astype(np.float32)
    direction -= direction.mean(dtype=np.float64)

    def loss_fn(delta):
        _, pk, _ = delta_to_pk(delta, conf, mas="CIC")
        weights = jnp.linspace(1.0, 2.0, pk.shape[0], dtype=pk.dtype)
        return jnp.sum(weights * pk)

    _directional_fd_check(
        loss_fn,
        jnp.asarray(delta0),
        jnp.asarray(direction),
        eps=jnp.asarray(3e-3, dtype=jnp.float32),
        rtol=2e-2,
        atol=2e-4,
    )


def test_particles_to_pk_directional_derivative_matches_finite_difference():
    box_size = 25.0
    nmesh = 6
    conf = Configuration(
        ptcl_spacing=box_size / nmesh,
        ptcl_grid_shape=(nmesh, nmesh, nmesh),
        mesh_shape=1,
        float_dtype=jnp.float32,
    )

    grid = np.stack(np.meshgrid(
        np.arange(nmesh, dtype=np.int16),
        np.arange(nmesh, dtype=np.int16),
        np.arange(nmesh, dtype=np.int16),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)
    base_pmid = jnp.asarray(grid, dtype=conf.pmid_dtype)
    rng = np.random.default_rng(3)
    max_disp = 0.1 * float(conf.cell_size)
    disp0 = rng.uniform(-max_disp, max_disp, size=grid.shape).astype(np.float32)
    direction = rng.normal(size=disp0.shape).astype(np.float32)

    def loss_fn(disp):
        ptcl = Particles.from_pmid(conf, base_pmid, disp)
        _, pk, _ = particles_to_pk(ptcl, conf, mas="CIC")
        weights = jnp.linspace(1.0, 2.0, pk.shape[0], dtype=pk.dtype)
        return jnp.sum(weights * pk)

    _directional_fd_check(
        loss_fn,
        jnp.asarray(disp0),
        jnp.asarray(direction),
        eps=jnp.asarray(5e-3, dtype=jnp.float32),
        rtol=5e-2,
        atol=2e-4,
    )
