from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from .configuration import Configuration
from .corrections import apply_potential_correction, force_green_kernel, force_uses_interlacing
from .gather import _gather, gather, gather_stacked_mesh_halo
from .scatter import scatter, reduce_grad_across_gpus
from .utils import AXIS_NAME


def get_k_squared(kvec, conf):
    """Return ``k^2`` on the standard spectral layout.

    The distributed branch builds only the local slab of the broadcasted
    ``kx^2 + ky^2 + kz^2`` field. That keeps Poisson-kernel construction from
    forcing a dense all-device materialization.
    
    Parameters
    ----------
    kvec : sequence of jax.Array
        Sparse broadcastable Fourier wavevector components on the standard
        spectral layout.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        ``k^2`` field on the standard spectral layout.
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
    """Return ``k^2`` on the transposed layout emitted by distributed rFFTs.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Sparse broadcastable Fourier wavevector components.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        ``k^2`` field on the transposed spectral layout.
    """
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
    
    Parameters
    ----------
    kvec : sequence of jax.Array
        Sparse broadcastable Fourier wavevector components.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Discrete Laplacian symbol on the transposed spectral layout.
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
    
    Parameters
    ----------
    src : jax.Array
        Spectral field to be filtered.
    masks : sequence of jax.Array
        One-dimensional broadcastable masks, typically cached on
        ``Configuration``.

    Returns
    -------
    jax.Array
        Filtered spectral field.
    """
    for mask in masks:
        src = src * mask
    return src


@custom_vjp
def laplace(kvec, src, conf, cosmo=None):
    """Solve Poisson's equation in Fourier space using ``-src / k^2``.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Fourier wavevector components on the standard spectral layout.
    src : jax.Array
        Spectral source field.
    conf : Configuration
        Active simulation configuration.
    cosmo : optional
        Unused compatibility argument preserved for call-shape consistency.

    Returns
    -------
    jax.Array
        Fourier-space potential.
    """
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
    """Poisson solve for the transposed spectral layout.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Fourier wavevector components on the transposed spectral layout.
    src : jax.Array
        Spectral source field.
    conf : Configuration
        Active simulation configuration.
    cosmo : optional
        Unused compatibility argument preserved for call-shape consistency.

    Returns
    -------
    jax.Array
        Fourier-space potential on the transposed layout.
    """
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
    """Poisson solve with the selected PM Green's function.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Fourier wavevector components on the transposed spectral layout.
    src : jax.Array
        Spectral source field.
    conf : Configuration
        Active simulation configuration.
    kernel : {"continuum", "discrete_laplacian"}, optional
        Green's function family to apply.

    Returns
    -------
    jax.Array
        Fourier-space potential on the transposed layout.
    """
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

    Parameters
    ----------
    k : jax.Array
        One Fourier wavevector component.
    pot : jax.Array
        Fourier-space potential.
    spacing : float
        Real-space mesh spacing along the differentiated axis.

    Returns
    -------
    jax.Array
        Fourier-space negative gradient component.

    Notes
    -----
    The Nyquist derivative is set to zero. On a real grid that mode is its own
    conjugate, so an imaginary derivative there would violate the Hermitian
    structure required by the inverse rFFT.
    """
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = jnp.multiply(neg_ik, pot)

    return grad


def _spectral_gradient_components_from_potential(pot, conf: Configuration):
    """Stack Fourier-space force components using cached derivative factors."""
    neg_ik = getattr(conf, "neg_ik", None)
    if neg_ik is None:
        return jnp.stack([neg_grad(k, pot, conf.cell_size) for k in conf.kvec], axis=0)
    return jnp.stack([factor * pot for factor in neg_ik], axis=0)


def _laplace_replicated(kvec, src, conf: Configuration):
    """Poisson solve for a replicated standard spectral layout."""
    kx, ky, kz = [jnp.squeeze(a).astype(conf.float_dtype) for a in kvec]
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    return src * jnp.where(k2 != 0, -1 / k2, 0).astype(conf.float_dtype)


def _spectral_gradient_components_from_density_hat(dens_hat, conf: Configuration):
    """Apply the cached Poisson kernel and derivative factors in one spectral step."""
    if conf.replicated_mesh:
        pot = _laplace_replicated(conf.kvec, dens_hat, conf)
        return _spectral_gradient_components_from_potential(pot, conf)

    pot = laplace_transposed_with_kernel(conf.kvec, dens_hat, conf)
    return _spectral_gradient_components_from_potential(pot, conf)


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
    if conf.replicated_mesh:
        return jnp.fft.rfftn(dens)
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
    return _spectral_gradient_components_from_potential(pot, conf)


def _can_use_batched_gradient_fft(conf: Configuration):
    """Return whether this configuration has a distributed batched iRFFT path."""
    return conf.compute_mesh is not None and conf.mGPU_irfftn_transposed_batched is not None


def _batched_gradient_meshes_from_potential(pot, conf: Configuration):
    """Transform all three force components with one batched distributed iRFFT."""
    spectral_grads = _spectral_gradient_components(pot, conf)
    return conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)


def _gradient_meshes_from_spectral_components(spectral_grads, conf: Configuration, use_batched=True):
    """Return real-space force meshes from prebuilt spectral components."""
    if use_batched and _can_use_batched_gradient_fft(conf):
        grad_meshes = conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)
        return tuple(grad_meshes[i] for i in range(grad_meshes.shape[0]))

    grad_meshes = []
    for grad in spectral_grads:
        if conf.compute_mesh is None:
            grad = jnp.fft.irfftn(grad)
        else:
            grad = conf.mGPU_irfftn_transposed(grad)
        grad_meshes.append(grad.astype(conf.float_dtype))
    return tuple(grad_meshes)


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


def _acceleration_from_density_hat(dens_hat, ptcl, conf: Configuration):
    """Evaluate acceleration directly from the density FFT for the continuum force."""
    spectral_grads = _spectral_gradient_components_from_density_hat(dens_hat, conf)
    if conf.replicated_mesh:
        grad_meshes = jnp.fft.irfftn(spectral_grads, axes=(1, 2, 3)).astype(conf.float_dtype)
        acc = jnp.stack(
            [_gather(ptcl.pmid, ptcl.disp, conf, mesh, 0, 0, None) for mesh in grad_meshes],
            axis=-1,
        )
        if ptcl.unused_index is None:
            return acc
        mask = ptcl.unused_index.reshape(ptcl.unused_index.shape + (1,) * (acc.ndim - 1))
        return jnp.where(mask, jnp.zeros_like(acc), acc)

    if _can_use_batched_gradient_fft(conf):
        grad_meshes = conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)
        return gather_stacked_mesh_halo(ptcl, conf, jnp.moveaxis(grad_meshes, 0, -1))

    grad_meshes = _gradient_meshes_from_spectral_components(spectral_grads, conf, use_batched=False)
    acc = [gather(ptcl, conf, grad) for grad in grad_meshes]
    return jnp.stack(acc, axis=-1)


def _gravity_from_density(dens, ptcl, cosmo, conf: Configuration, a=None, correction=None):
    """Evaluate particle acceleration from a precomputed density mesh."""
    if correction is None:
        dens = dens - 1
        dens = dens * (1.5 * cosmo.Omega_m.astype(conf.float_dtype))
        dens_hat = _density_hat_from_real(dens, conf)
        dens_hat = apply_particle_nyquist_filter(dens_hat, conf.particle_nyquist_masks)
        return _acceleration_from_density_hat(dens_hat, ptcl, conf)

    pot = _gravity_potential_from_density(dens, cosmo.Omega_m, conf, a=a, cosmo=cosmo, correction=correction)
    use_batched = correction is None or conf.corrected_force_batched_fft
    return _acceleration_from_potential(
        pot,
        ptcl,
        conf,
        use_batched=use_batched,
        use_vmap_gather=correction is not None and not use_batched,
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
    """Sum cotangents across halo-duplicated slots for one particle field.

    Parameters
    ----------
    ptcl : Particles
        Particle state whose slots may include halo duplicates.
    cot : jax.Array
        Per-slot cotangent field aligned with ``ptcl``.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Cotangent field reduced to the canonical per-particle convention.
    """
    unused_index = None if ptcl.unused_index is None else ptcl.unused_index
    return _reduce_gather_disp_cot(ptcl.pmid, ptcl.disp, unused_index, cot, conf)


def duplicate_slot_counts(ptcl, conf: Configuration):
    """Count halo-duplicated slots for each physical particle slot.

    Parameters
    ----------
    ptcl : Particles
        Particle state whose slots may include halo duplicates.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Integer count per slot describing how many duplicated copies contribute
        to that physical particle.
    """
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
    """Gravitational accelerations of particles in ``[H_0^2]``.

    Parameters
    ----------
    a : float
        Scale factor of the force evaluation.
    ptcl : Particles
        Particle state whose acceleration is being computed.
    cosmo : Cosmology
        Cosmology providing the matter-density prefactor and correction
        conditioning context.
    conf : Configuration
        Active simulation configuration.
    correction : optional
        Potential-correction object applied on top of the base PM solve.

    Returns
    -------
    jax.Array
        Particle accelerations with the same leading slot structure as
        ``ptcl.disp``.
    """
    if force_uses_interlacing(correction):
        pot = _gravity_potential_interlaced(ptcl, cosmo.Omega_m, conf, a=a, cosmo=cosmo, correction=correction)
        return _acceleration_from_potential(pot, ptcl, conf)

    dens = scatter(ptcl, conf)
    return _gravity_from_density(dens, ptcl, cosmo, conf, a=a, correction=correction)
