from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from .configuration import Configuration
from .corrections import apply_potential_correction, force_green_kernel, force_uses_interlacing
from .gather import gather, gather_stacked_mesh_halo
from .scatter import scatter, reduce_grad_across_gpus
from .utils import AXIS_NAME


def get_k_squared(kvec, conf):
    """Return ``k^2`` on the standard spectral layout.

    The distributed branch builds only the local slab of the broadcasted
    ``kx^2 + ky^2 + kz^2`` field. That keeps Poisson-kernel construction from
    forcing a dense all-device materialization.
    """
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
    """Return ``k^2`` on the transposed layout emitted by distributed rFFTs."""
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


def get_discrete_k_squared_transposed(kvec, conf):
    """Return the finite-difference Laplacian symbol on the transposed layout.

    The continuum kernel uses ``k^2``. The optional discrete PM Green's
    function replaces each axis with ``2 sin(k dx / 2) / dx``, matching the
    lattice Laplacian more closely near the mesh scale.
    """
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]
    cell_size = jnp.asarray(conf.cell_size, dtype=conf.float_dtype)
    kx_eff = 2 * jnp.sin(kx * cell_size / 2) / cell_size
    ky_eff = 2 * jnp.sin(ky * cell_size / 2) / cell_size
    kz_eff = 2 * jnp.sin(kz * cell_size / 2) / cell_size
    if conf.compute_mesh is None:
        return (
            kx_eff[:, None, None] ** 2
            + ky_eff[None, :, None] ** 2
            + kz_eff[None, None, :] ** 2
        ).astype(conf.float_dtype)

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(conf.compute_mesh, P(None)),
            NamedSharding(conf.compute_mesh, P(AXIS_NAME)),
            NamedSharding(conf.compute_mesh, P(None)),
        ),
        out_shardings=NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)),
    )
    def create_discrete_k_magnitude_transposed(kx_replicated, ky_sharded, kz_replicated):
        local_shard = (
            kx_replicated[:, None, None] ** 2
            + ky_sharded[None, :, None] ** 2
            + kz_replicated[None, None, :] ** 2
        )
        return local_shard.astype(conf.float_dtype)

    return create_discrete_k_magnitude_transposed(kx_eff, ky_eff, kz_eff)


def apply_particle_nyquist_filter(src, masks):
    """Apply broadcastable per-axis particle-Nyquist masks.

    ``masks`` are one-dimensional sharded arrays prepared by
    ``Configuration``. Multiplying them one axis at a time avoids creating a
    dense 3D boolean mask and preserves the existing FFT sharding.
    """
    for mask in masks:
        src = src * mask
    return src


@custom_vjp
def laplace(kvec, src, conf, cosmo=None):
    """Solve Poisson's equation in Fourier space using ``-src / k^2``."""
    k2 = get_k_squared(kvec, conf)

    pot = jnp.where(k2 != 0, - src / k2, 0)

    return pot


def laplace_fwd(kvec, src, conf, cosmo):
    """Forward rule for the Poisson custom VJP."""
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
    """Poisson solve for the transposed spectral layout."""
    k2 = get_k_squared_transposed(kvec, conf)

    pot = jnp.where(k2 != 0, - src / k2, 0)

    return pot


def laplace_transposed_fwd(kvec, src, conf, cosmo):
    """Forward rule for the transposed-layout Poisson custom VJP."""
    pot = laplace_transposed(kvec, src, conf, cosmo)
    return pot, (kvec, conf, cosmo)


def laplace_transposed_bwd(res, pot_cot):
    """Backward rule for the transposed-layout Poisson custom VJP."""
    kvec, conf, cosmo = res
    src_cot = laplace_transposed(kvec, pot_cot, conf, cosmo)
    return None, src_cot, None, None


laplace_transposed.defvjp(laplace_transposed_fwd, laplace_transposed_bwd)


def laplace_transposed_with_kernel(kvec, src, conf, kernel="continuum"):
    """Poisson solve with the selected PM Green's function."""
    if kernel == "continuum":
        return laplace_transposed(kvec, src, conf, None)
    if kernel == "discrete_laplacian":
        k2 = get_discrete_k_squared_transposed(kvec, conf)
    else:
        raise ValueError(f"Unsupported PM Green's function {kernel!r}.")

    denom = jnp.where(k2 != 0, k2, jnp.ones_like(k2))
    pot = -src / denom
    return jnp.where(k2 != 0, pot, jnp.zeros_like(pot))


def neg_grad(k, pot, spacing):
    """Return the Fourier-space negative gradient component ``-i k pot``.

    The Nyquist derivative is set to zero. On a real grid that mode is its own
    conjugate, so an imaginary derivative there would violate the Hermitian
    structure required by the inverse rFFT.
    """
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = jnp.multiply(neg_ik, pot)

    return grad


def _gravity_potential_from_density(dens, omega_m, conf: Configuration, a=None, cosmo=None, correction=None):
    """Convert a real density mesh into a corrected Fourier potential."""
    dens = dens - 1
    dens = dens * (1.5 * omega_m.astype(conf.float_dtype))
    source_real = dens
    if conf.compute_mesh is None:
        dens = jnp.fft.rfftn(dens)
    else:
        dens = conf.mGPU_rfftn_transposed(dens)
    dens = apply_particle_nyquist_filter(dens, conf.particle_nyquist_masks)
    pot = laplace_transposed_with_kernel(conf.kvec, dens, conf, force_green_kernel(correction))
    return apply_potential_correction(pot, a, cosmo, conf, correction, source_real=source_real)


def _density_hat_from_real(dens, conf):
    """FFT a real mesh into the spectral layout used by gravity."""
    return jnp.fft.rfftn(dens) if conf.compute_mesh is None else conf.mGPU_rfftn_transposed(dens)


def _interlacing_phase(conf):
    """Fourier phase that shifts the half-cell interlaced density back."""
    ksum = conf.kvec[0] + conf.kvec[1] + conf.kvec[2]
    return jnp.exp(-1j * ksum * jnp.asarray(conf.cell_size / 2, dtype=conf.float_dtype))


def _gravity_potential_interlaced(ptcl, omega_m, conf: Configuration, a=None, cosmo=None, correction=None):
    """Build a potential from the average of regular and half-cell CIC scatters.

    Interlacing cancels the leading odd aliases from particle assignment. The
    second scatter is shifted by half a mesh cell, transformed, phase-shifted
    back into the original coordinate system, and averaged with the unshifted
    density before the Poisson solve.
    """
    factor = 1.5 * omega_m.astype(conf.float_dtype)
    dens0 = (scatter(ptcl, conf) - 1) * factor
    offset = jnp.asarray(conf.cell_size / 2, dtype=conf.float_dtype)
    dens1 = (scatter(ptcl, conf, offset=offset) - 1) * factor
    dens_hat = 0.5 * (_density_hat_from_real(dens0, conf) + _density_hat_from_real(dens1, conf) * _interlacing_phase(conf))
    dens_hat = apply_particle_nyquist_filter(dens_hat, conf.particle_nyquist_masks)
    pot = laplace_transposed_with_kernel(conf.kvec, dens_hat, conf, force_green_kernel(correction))
    return apply_potential_correction(pot, a, cosmo, conf, correction, source_real=dens0)


def _spectral_gradient_components(pot, conf: Configuration):
    """Stack the three Fourier-space force components for batched iFFTs."""
    return jnp.stack([neg_grad(k, pot, conf.cell_size) for k in conf.kvec], axis=0)


def _can_use_batched_gradient_fft(conf: Configuration):
    """Return whether this configuration has a distributed batched iRFFT path."""
    return conf.compute_mesh is not None and conf.mGPU_irfftn_transposed_batched is not None


def _batched_gradient_meshes_from_potential(pot, conf: Configuration):
    """Transform all three force components with one batched distributed iRFFT."""
    spectral_grads = _spectral_gradient_components(pot, conf)
    return conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)


def _gradient_meshes_from_potential(pot, conf: Configuration, use_batched=True):
    """Return real-space force meshes, optionally using the batched FFT path."""
    if use_batched and _can_use_batched_gradient_fft(conf):
        grad_meshes = _batched_gradient_meshes_from_potential(pot, conf)
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


def _acceleration_from_potential(pot, ptcl, conf: Configuration, use_batched=True, use_vmap_gather=False):
    """Gather force-mesh components at particle positions."""
    if use_batched and _can_use_batched_gradient_fft(conf):
        grad_meshes = _batched_gradient_meshes_from_potential(pot, conf)
        return gather_stacked_mesh_halo(ptcl, conf, jnp.moveaxis(grad_meshes, 0, -1))

    grad_meshes = _gradient_meshes_from_potential(pot, conf, use_batched=False)

    if use_vmap_gather:
        stacked_grad_meshes = jnp.stack(grad_meshes, axis=0)
        return jax.vmap(lambda mesh: gather(ptcl, conf, mesh), in_axes=0, out_axes=-1)(stacked_grad_meshes)

    acc = [gather(ptcl, conf, grad) for grad in grad_meshes]
    return jnp.stack(acc, axis=-1)


def _gravity_from_density(dens, ptcl, cosmo, conf: Configuration, a=None, correction=None):
    """Evaluate particle acceleration from a precomputed density mesh."""
    pot = _gravity_potential_from_density(dens, cosmo.Omega_m, conf, a=a, cosmo=cosmo, correction=correction)
    return _acceleration_from_potential(
        pot,
        ptcl,
        conf,
        use_batched=correction is None,
        use_vmap_gather=correction is not None,
    )


def _gravity_mesh_fields_from_density(dens, omega_m, conf: Configuration, a=None, cosmo=None, correction=None):
    """Return the real-space force meshes generated by a density field."""
    pot = _gravity_potential_from_density(dens, omega_m, conf, a=a, cosmo=cosmo, correction=correction)
    return _gradient_meshes_from_potential(pot, conf)


def _reduce_gather_disp_cot(pmid, disp, unused_index, disp_cot, conf: Configuration):
    """Sum displacement cotangents over duplicate particle-halo slots.

    ``particle_halo`` stores authoritative particles and halo copies, so gather
    backward can produce several cotangent contributions for one physical
    particle. ``mesh_halo`` has no duplicated particle slots and returns the
    cotangent unchanged.
    """
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
    """Count halo-duplicated slots for each physical particle slot."""
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
    if force_uses_interlacing(correction):
        pot = _gravity_potential_interlaced(ptcl, cosmo.Omega_m, conf, a=a, cosmo=cosmo, correction=correction)
        return _acceleration_from_potential(pot, ptcl, conf)

    dens = scatter(ptcl, conf)
    return _gravity_from_density(dens, ptcl, cosmo, conf, a=a, correction=correction)
