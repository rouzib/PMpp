from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from .configuration import Configuration
from .scatter import scatter, reduce_grad_across_gpus
from .gather_old import gather
from .utils import AXIS_NAME


def get_k_squared(kvec, conf):
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]

    @partial(jax.jit,
             in_shardings=(
                     NamedSharding(conf.compute_mesh, P(AXIS_NAME)),
                     NamedSharding(conf.compute_mesh, P(None)),
                     NamedSharding(conf.compute_mesh, P(None)),
             ),
             out_shardings=NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None))
             )
    def create_k_magnitude_sharded(kx_sharded, ky_replicated, kz_replicated):
        """
        Creates the magnitude of the k-vector in a JIT-compatible and
        memory-efficient, sharded manner.

        Each device runs this same code, but on its own piece of the data.
        """
        kx_b = kx_sharded[:, None, None]
        ky_b = ky_replicated[None, :, None]
        kz_b = kz_replicated[None, None, :]

        local_shard = jnp.array(kx_b ** 2 + ky_b ** 2 + kz_b ** 2)
        return local_shard.astype(conf.float_dtype)

    return create_k_magnitude_sharded(kx, ky, kz)


@custom_vjp
def laplace(kvec, src, conf, cosmo=None):
    """Laplace kernel in Fourier space."""
    k2 = get_k_squared(kvec, conf)

    pot = jnp.where(k2 != 0, - src / k2, 0)

    return pot


def laplace_fwd(kvec, src, conf, cosmo):
    pot = laplace(kvec, src, conf, cosmo)
    return pot, (kvec, conf, cosmo)


def laplace_bwd(res, pot_cot):
    """Custom vjp to avoid NaN when using where, as well as to save memory.

    .. _JAX FAQ:
        https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    """
    kvec, conf, cosmo = res
    src_cot = laplace(kvec, pot_cot, conf, cosmo)
    return None, src_cot, None, None


laplace.defvjp(laplace_fwd, laplace_bwd)


def neg_grad(k, pot, spacing):
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = jnp.multiply(neg_ik, pot)

    return grad


def _gravity_from_density(dens, ptcl, cosmo, conf: Configuration):
    grad_meshes = _gravity_mesh_fields_from_density(dens, cosmo.Omega_m, conf)
    acc = [gather(ptcl, conf, grad) for grad in grad_meshes]
    return jnp.stack(acc, axis=-1)


def _gravity_mesh_fields_from_density(dens, omega_m, conf: Configuration):
    kvec = conf.kvec

    dens = dens - 1
    dens = dens * (1.5 * omega_m.astype(conf.float_dtype))
    dens = conf.mGPU_rfftn(dens)

    pot = laplace(kvec, dens, conf, None)

    grad_meshes = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)
        grad = conf.mGPU_irfftn(grad)
        grad_meshes.append(grad.astype(conf.float_dtype))

    return tuple(grad_meshes)


def _reduce_gather_disp_cot(ptcl, disp_cot, conf: Configuration):
    if not conf.use_mGPU:
        return disp_cot

    @partial(
        shard_map,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME, None),
            None,
        ),
        out_specs=P(AXIS_NAME, None),
        check_rep=False,
    )
    def reduce_local(disp_cot_local, pmid_local, unused_local, disp_local, conf_local):
        valid_mask = ~unused_local
        return reduce_grad_across_gpus(disp_cot_local, pmid_local, disp_local, valid_mask, conf_local)

    unused_index = (
        jnp.zeros(disp_cot.shape[0], dtype=jnp.bool_)
        if ptcl.unused_index is None
        else jax.lax.stop_gradient(ptcl.unused_index)
    )
    pmid = jax.lax.stop_gradient(ptcl.pmid)
    return reduce_local(disp_cot, pmid, unused_index, ptcl.disp, conf)


def duplicate_slot_counts(ptcl, conf: Configuration):
    if not conf.use_mGPU:
        return jnp.ones_like(ptcl.disp)

    @partial(
        shard_map,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME, None),
            None,
        ),
        out_specs=P(AXIS_NAME, None),
        check_rep=False,
    )
    def count_local(counts_local, pmid_local, unused_local, disp_local, conf_local):
        valid_mask = ~unused_local
        return reduce_grad_across_gpus(counts_local, pmid_local, disp_local, valid_mask, conf_local)

    unused_index = (
        jnp.zeros(ptcl.disp.shape[0], dtype=jnp.bool_)
        if ptcl.unused_index is None
        else jax.lax.stop_gradient(ptcl.unused_index)
    )
    pmid = jax.lax.stop_gradient(ptcl.pmid)
    counts = count_local(jnp.ones_like(ptcl.disp), pmid, unused_index, ptcl.disp, conf)
    return jnp.where(counts != 0, counts, 1)


def _float0_zeros_like(x):
    return jnp.zeros(x.shape, dtype=jax.dtypes.float0)


def _particle_cotangent(disp_cot, ptcl):
    return ptcl.replace(
        pmid=_float0_zeros_like(ptcl.pmid),
        disp=disp_cot,
        vel=None if ptcl.vel is None else jnp.zeros_like(ptcl.vel),
        acc=None if ptcl.acc is None else jnp.zeros_like(ptcl.acc),
        unused_index=None if ptcl.unused_index is None else _float0_zeros_like(ptcl.unused_index),
        halo_mask=None if ptcl.halo_mask is None else _float0_zeros_like(ptcl.halo_mask),
        attr=None if ptcl.attr is None else jax.tree_util.tree_map(jnp.zeros_like, ptcl.attr),
    )


def _cosmo_cotangent(omega_m_cot, cosmo):
    cosmo_cot = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.zeros_like(x),
        cosmo,
    )
    return cosmo_cot.replace(Omega_m=omega_m_cot)


@custom_vjp
def gravity(a, ptcl, cosmo, conf: Configuration):
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""
    dens = scatter(ptcl, conf)
    return _gravity_from_density(dens, ptcl, cosmo, conf)


def gravity_fwd(a, ptcl, cosmo, conf):
    dens = scatter(ptcl, conf)
    acc = _gravity_from_density(dens, ptcl, cosmo, conf)
    return acc, (a, ptcl, cosmo, conf, dens)


def gravity_bwd(res, acc_cot):
    a, ptcl, cosmo, conf, dens = res

    grad_meshes = _gravity_mesh_fields_from_density(dens, cosmo.Omega_m, conf)

    mesh_cots = []
    gather_disp_cot_raw = jnp.zeros_like(ptcl.disp)
    for axis, grad_mesh in enumerate(grad_meshes):
        _, gather_vjp = jax.vjp(
            lambda mesh_input, disp_input: gather(ptcl.replace(disp=disp_input), conf, mesh_input),
            grad_mesh,
            ptcl.disp,
        )
        mesh_cot_axis, disp_cot_axis = gather_vjp(acc_cot[..., axis])
        mesh_cots.append(mesh_cot_axis)
        gather_disp_cot_raw = gather_disp_cot_raw + disp_cot_axis

    _, density_to_mesh_vjp = jax.vjp(
        lambda dens_input, omega_m_input: _gravity_mesh_fields_from_density(
            dens_input, omega_m_input, conf
        ),
        dens,
        cosmo.Omega_m,
    )
    dens_cot, omega_m_cot = density_to_mesh_vjp(tuple(mesh_cots))

    _, scatter_vjp = jax.vjp(lambda disp_input: scatter(ptcl.replace(disp=disp_input), conf), ptcl.disp)
    scatter_disp_cot, = scatter_vjp(dens_cot)

    gather_disp_cot = _reduce_gather_disp_cot(ptcl, gather_disp_cot_raw, conf)
    ptcl_cot = _particle_cotangent(scatter_disp_cot + gather_disp_cot, ptcl)
    cosmo_cot = _cosmo_cotangent(omega_m_cot, cosmo)

    return jnp.zeros_like(a), ptcl_cot, cosmo_cot, None


gravity.defvjp(gravity_fwd, gravity_bwd)
