from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from .configuration import Configuration
from .gather import gather, gather_stacked_mesh_halo
from .potential_correction import apply_potential_correction
from .scatter import scatter, reduce_grad_across_gpus
from .utils import AXIS_NAME


def get_k_squared(kvec, conf):
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]
    if conf.compute_mesh is None:
        return (kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(conf.float_dtype)

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


def get_k_squared_transposed(kvec, conf):
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]
    if conf.compute_mesh is None:
        return (kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(conf.float_dtype)

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(conf.compute_mesh, P(None)),
            NamedSharding(conf.compute_mesh, P(AXIS_NAME)),
            NamedSharding(conf.compute_mesh, P(None)),
        ),
        out_shardings=NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)),
    )
    def create_k_magnitude_transposed(kx_replicated, ky_sharded, kz_replicated):
        kx_b = kx_replicated[:, None, None]
        ky_b = ky_sharded[None, :, None]
        kz_b = kz_replicated[None, None, :]

        local_shard = jnp.array(kx_b ** 2 + ky_b ** 2 + kz_b ** 2)
        return local_shard.astype(conf.float_dtype)

    return create_k_magnitude_transposed(kx, ky, kz)


def apply_particle_nyquist_filter(src, masks):
    """Apply broadcastable per-axis particle-Nyquist masks."""
    for mask in masks:
        src = src * mask
    return src


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


@custom_vjp
def laplace_transposed(kvec, src, conf, cosmo=None):
    """Laplace kernel in Fourier space for the transposed spectral layout."""
    k2 = get_k_squared_transposed(kvec, conf)

    pot = jnp.where(k2 != 0, - src / k2, 0)

    return pot


def laplace_transposed_fwd(kvec, src, conf, cosmo):
    pot = laplace_transposed(kvec, src, conf, cosmo)
    return pot, (kvec, conf, cosmo)


def laplace_transposed_bwd(res, pot_cot):
    kvec, conf, cosmo = res
    src_cot = laplace_transposed(kvec, pot_cot, conf, cosmo)
    return None, src_cot, None, None


laplace_transposed.defvjp(laplace_transposed_fwd, laplace_transposed_bwd)


def neg_grad(k, pot, spacing):
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = jnp.multiply(neg_ik, pot)

    return grad


def _gravity_potential_from_density(dens, omega_m, conf: Configuration, a=None, cosmo=None, correction=None):
    dens = dens - 1
    dens = dens * (1.5 * omega_m.astype(conf.float_dtype))
    source_real = dens
    if conf.compute_mesh is None:
        dens = jnp.fft.rfftn(dens)
    else:
        dens = conf.mGPU_rfftn_transposed(dens)
    dens = apply_particle_nyquist_filter(dens, conf.particle_nyquist_masks)
    pot = laplace_transposed(conf.kvec, dens, conf, None)
    return apply_potential_correction(pot, a, cosmo, conf, correction, source_real=source_real)


def _spectral_gradient_components(pot, conf: Configuration):
    return jnp.stack([neg_grad(k, pot, conf.cell_size) for k in conf.kvec], axis=0)


def _gravity_from_density(dens, ptcl, cosmo, conf: Configuration, a=None, correction=None):
    pot = _gravity_potential_from_density(dens, cosmo.Omega_m, conf, a=a, cosmo=cosmo, correction=correction)
    if conf.compute_mesh is not None and correction is None and conf.mGPU_irfftn_transposed_batched is not None:
        spectral_grads = _spectral_gradient_components(pot, conf)
        grad_meshes = conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)
        return gather_stacked_mesh_halo(ptcl, conf, jnp.moveaxis(grad_meshes, 0, -1))

    grad_meshes = []
    for k in conf.kvec:
        grad = neg_grad(k, pot, conf.cell_size)
        if conf.compute_mesh is None:
            grad = jnp.fft.irfftn(grad).astype(conf.float_dtype)
        else:
            grad = conf.mGPU_irfftn_transposed(grad).astype(conf.float_dtype)
        grad_meshes.append(grad)

    if correction is not None:
        stacked_grad_meshes = jnp.stack(grad_meshes, axis=0)
        return jax.vmap(lambda mesh: gather(ptcl, conf, mesh), in_axes=0, out_axes=-1)(stacked_grad_meshes)

    acc = [gather(ptcl, conf, grad) for grad in grad_meshes]
    return jnp.stack(acc, axis=-1)


def _gravity_mesh_fields_from_density(dens, omega_m, conf: Configuration, a=None, cosmo=None, correction=None):
    pot = _gravity_potential_from_density(dens, omega_m, conf, a=a, cosmo=cosmo, correction=correction)

    if conf.compute_mesh is not None and conf.mGPU_irfftn_transposed_batched is not None:
        spectral_grads = _spectral_gradient_components(pot, conf)
        grad_meshes = conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)
        return tuple(grad_meshes[i] for i in range(grad_meshes.shape[0]))

    grad_meshes = []
    for k in conf.kvec:
        grad = neg_grad(k, pot, conf.cell_size)
        if conf.compute_mesh is None:
            grad = jnp.fft.irfftn(grad)
        else:
            grad = conf.mGPU_irfftn_transposed(grad)
        grad_meshes.append(grad.astype(conf.float_dtype))

    return tuple(grad_meshes)


def _reduce_gather_disp_cot(pmid, disp, unused_index, disp_cot, conf: Configuration):
    if not conf.use_mGPU:
        return disp_cot
    if not conf.multigpu.store_particle_halos:
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
        if unused_index is None
        else jax.lax.stop_gradient(unused_index)
    )
    pmid = jax.lax.stop_gradient(pmid)
    return reduce_local(disp_cot, pmid, unused_index, disp, conf)


def reduce_duplicate_slot_cot(ptcl, cot, conf: Configuration):
    """Sum cotangents across halo-duplicated slots for one particle field."""
    unused_index = None if ptcl.unused_index is None else ptcl.unused_index
    return _reduce_gather_disp_cot(ptcl.pmid, ptcl.disp, unused_index, cot, conf)


def duplicate_slot_counts(ptcl, conf: Configuration):
    if not conf.use_mGPU:
        return jnp.ones_like(ptcl.disp)
    if not conf.multigpu.store_particle_halos:
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


def gravity(a, ptcl, cosmo, conf: Configuration, correction=None):
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""
    dens = scatter(ptcl, conf)
    return _gravity_from_density(dens, ptcl, cosmo, conf, a=a, correction=correction)
