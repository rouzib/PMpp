import math
from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from ..core.configuration import Configuration
from ..corrections import (
    apply_potential_correction, force_gradient_kernel, force_green_kernel, force_uses_interlacing,
)
from ..cic.gather import _gather, gather, gather_stacked_mesh_halo
from ..cic.scatter import scatter, reduce_grad_across_gpus
from ..core.utils import AXIS_NAME


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
        return (kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2).astype(conf.float_dtype)

    @partial(
        jax.jit, in_shardings=(
            NamedSharding(conf.compute_mesh,
                          P(AXIS_NAME)), NamedSharding(conf.compute_mesh,
                                                       P(None)), NamedSharding(conf.compute_mesh, P(None)),
        ), out_shardings=NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None))
    )
    def create_k_magnitude_sharded(kx_sharded, ky_replicated, kz_replicated):
        """Creates the magnitude of the k-vector in a JIT-compatible and
        memory-efficient, sharded manner.

        Each device runs this same code, but on its own piece of the data.

        Parameters
        ----------
        kx_sharded
            Local shard of x-axis wavenumbers.
        ky_replicated
            Replicated y-axis wavenumbers.
        kz_replicated
            Replicated z-axis rFFT wavenumbers."""
        kx_b = kx_sharded[:, None, None]
        ky_b = ky_replicated[None, :, None]
        kz_b = kz_replicated[None, None, :]

        local_shard = jnp.array(kx_b**2 + ky_b**2 + kz_b**2)
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
        return (kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2).astype(conf.float_dtype)

    @partial(
        jax.jit, in_shardings=(
            NamedSharding(conf.compute_mesh,
                          P(None)), NamedSharding(conf.compute_mesh,
                                                  P(AXIS_NAME)), NamedSharding(conf.compute_mesh, P(None)),
        ), out_shardings=NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)),
    )
    def create_k_magnitude_transposed(kx_replicated, ky_sharded, kz_replicated):
        """Build transposed-layout squared wavenumber magnitudes on each shard.

        Parameters
        ----------
        kx_replicated
            Replicated x-axis wavenumbers.
        ky_sharded
            Local shard of y-axis wavenumbers.
        kz_replicated
            Replicated z-axis rFFT wavenumbers.
        """
        kx_b = kx_replicated[:, None, None]
        ky_b = ky_sharded[None, :, None]
        kz_b = kz_replicated[None, None, :]

        local_shard = jnp.array(kx_b**2 + ky_b**2 + kz_b**2)
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
        return (kx_eff[:, None, None]**2 + ky_eff[None, :, None]**2 + kz_eff[None, None, :]**2).astype(conf.float_dtype)

    @partial(
        jax.jit, in_shardings=(
            NamedSharding(conf.compute_mesh,
                          P(None)), NamedSharding(conf.compute_mesh,
                                                  P(AXIS_NAME)), NamedSharding(conf.compute_mesh, P(None)),
        ), out_shardings=NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)),
    )
    def create_discrete_k_magnitude_transposed(kx_replicated, ky_sharded, kz_replicated):
        """Build transposed-layout discrete-gradient wavenumber magnitudes on each shard.

        Parameters
        ----------
        kx_replicated
            Replicated x-axis wavenumbers.
        ky_sharded
            Local shard of y-axis wavenumbers.
        kz_replicated
            Replicated z-axis rFFT wavenumbers.
        """
        local_shard = (kx_replicated[:, None, None]**2 + ky_sharded[None, :, None]**2 + kz_replicated[None, None, :]**2)
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

    pot = jnp.where(k2 != 0, -src / k2, 0)

    return pot


def laplace_fwd(kvec, src, conf, cosmo):
    """Forward rule for the Poisson custom VJP.

    Parameters
    ----------
    kvec
        Tuple of spectral wavenumber arrays.
    src
        Real-space source density field.
    conf
        Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
    cosmo
        Cosmology object supplying density, growth, and transfer parameters."""
    pot = laplace(kvec, src, conf, cosmo)
    return pot, (kvec, conf, cosmo)


def laplace_bwd(res, pot_cot):
    """Custom vjp to avoid NaN when using where, as well as to save memory.

    .. _JAX FAQ:
        https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    Parameters
    ----------
    res
        Residual values saved by a custom VJP forward rule.
    pot_cot
        Cotangent of the potential returned by the Laplace solve."""
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

    pot = jnp.where(k2 != 0, -src / k2, 0)

    return pot


def laplace_transposed_fwd(kvec, src, conf, cosmo):
    """Forward rule for the transposed-layout Poisson custom VJP.

    Parameters
    ----------
    kvec
        Tuple of spectral wavenumber arrays.
    src
        Real-space source density field.
    conf
        Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
    cosmo
        Cosmology object supplying density, growth, and transfer parameters."""
    pot = laplace_transposed(kvec, src, conf, cosmo)
    return pot, (kvec, conf, cosmo)


def laplace_transposed_bwd(res, pot_cot):
    """Backward rule for the transposed-layout Poisson custom VJP.

    Parameters
    ----------
    res
        Residual values saved by a custom VJP forward rule.
    pot_cot
        Cotangent of the potential returned by the Laplace solve."""
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


def neg_grad(k, pot, spacing, kernel="spectral"):
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
    if kernel == "spectral":
        nyquist = jnp.pi / spacing
        eps = nyquist * jnp.finfo(k.dtype).eps
        derivative_k = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, k)
    elif kernel == "fastpm_4point":
        phase = k * spacing
        derivative_k = (8.0 * jnp.sin(phase) - jnp.sin(2.0 * phase)) / (6.0 * spacing)
    else:
        raise ValueError(f"Unsupported force-gradient kernel {kernel!r}.")
    neg_ik = -1j * derivative_k

    grad = jnp.multiply(neg_ik, pot)

    return grad


def _spectral_gradient_components_from_potential(pot, conf: Configuration, gradient_kernel="spectral"):
    """Stack Fourier-space force components using cached derivative factors."""
    neg_ik = getattr(conf, "neg_ik", None) if gradient_kernel == "spectral" else None
    if neg_ik is None:
        return jnp.stack([neg_grad(k, pot, conf.cell_size, gradient_kernel) for k in conf.kvec], axis=0)
    return jnp.stack([factor * pot for factor in neg_ik], axis=0)


def _laplace_replicated(kvec, src, conf: Configuration):
    """Poisson solve for a replicated standard spectral layout."""
    kx, ky, kz = [jnp.squeeze(a).astype(conf.float_dtype) for a in kvec]
    k2 = kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2
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
    dens_hat = 0.5 * (
        _density_hat_from_real(dens0, conf) + _density_hat_from_real(dens1, conf) * _interlacing_phase(conf)
    )
    dens_hat = apply_particle_nyquist_filter(dens_hat, conf.particle_nyquist_masks)
    pot = laplace_transposed_with_kernel(conf.kvec, dens_hat, conf, force_green_kernel(correction))
    return apply_potential_correction(pot, a, cosmo, conf, correction, source_real=dens0)


def _spectral_gradient_components(pot, conf: Configuration, gradient_kernel="spectral"):
    """Stack the three Fourier-space force components for batched iFFTs."""
    return _spectral_gradient_components_from_potential(pot, conf, gradient_kernel)


def _can_use_batched_gradient_fft(conf: Configuration):
    """Return whether the distributed batched iRFFT is safe for this mesh.

    The batched inverse produces one local real mesh with a leading component
    dimension.  CUDA/XLA currently misnormalizes that result when its local
    element count exceeds the signed 32-bit indexing range.  For example, the
    eight-way ``2048^3`` force mesh used by a ``1024^3`` particle run with
    ``mesh_shape=2`` contains ``3 * 256 * 2048 * 2048`` local output elements.
    The observed result is too large by exactly ``2048^2``.

    Keep smaller configurations on the faster batched path.  Larger ones use
    the scalar component transforms, whose local arrays remain within the
    supported indexing range.
    """
    if conf.compute_mesh is None or conf.mGPU_irfftn_transposed_batched is None:
        return False

    local_shape = getattr(conf, "local_mesh_with_halo_shape", None)
    if not local_shape:
        local_shape = getattr(conf, "local_mesh_shape", None)
    if not local_shape:
        # Lightweight test doubles and external callers may not expose the
        # derived local shape. Preserve the historical dispatch for them.
        return True

    component_count = int(getattr(conf, "dim", len(local_shape)))
    return component_count * math.prod(int(size) for size in local_shape) <= 2**31 - 1


def _batched_gradient_meshes_from_potential(pot, conf: Configuration, gradient_kernel="spectral"):
    """Transform all three force components with one batched distributed iRFFT."""
    spectral_grads = _spectral_gradient_components(pot, conf, gradient_kernel)
    return conf.mGPU_irfftn_transposed_batched(spectral_grads).astype(conf.float_dtype)


def _streamed_acceleration_from_potential(pot, ptcl, conf: Configuration):
    """Transform and gather one force component at a time.

    The component axis is an inner three-step ``lax.fori_loop``. Consequently
    the real and spectral mesh for one component is consumed before the next
    iteration starts, while the particle acceleration is the only large carry.
    This is deliberately separate from the faster batched path so ordinary
    simulations retain their existing execution plan.
    """
    if conf.dim != 3:
        raise ValueError(f"streamed gravity requires three dimensions, got dim={conf.dim}.")

    cached_factors = getattr(conf, "neg_ik", None)

    def component_branch(axis):
        """Build one statically selected FFT-and-gather loop branch."""
        k = conf.kvec[axis]
        factor = None if cached_factors is None else cached_factors[axis]

        def transform_and_gather(potential):
            spectral = neg_grad(k, potential, conf.cell_size) if factor is None else factor * potential
            if conf.compute_mesh is None:
                mesh = jnp.fft.irfftn(spectral)
            else:
                mesh = conf.mGPU_irfftn_transposed(spectral)
            return gather(ptcl, conf, mesh.astype(conf.float_dtype))

        return transform_and_gather

    branches = tuple(component_branch(axis) for axis in range(conf.dim))

    def body(axis, acc):
        component = jax.lax.switch(axis, branches, pot)
        return jax.lax.dynamic_update_slice_in_dim(acc, component[:, None], axis, axis=1)

    can_reuse_acc = (ptcl.acc is not None and ptcl.acc.shape == ptcl.disp.shape and ptcl.acc.dtype == ptcl.disp.dtype)
    acc = ptcl.acc if can_reuse_acc else jnp.zeros_like(ptcl.disp)
    return jax.lax.fori_loop(0, conf.dim, body, acc)


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


def _gradient_meshes_from_potential(pot, conf: Configuration, use_batched=True, gradient_kernel="spectral"):
    """Return real-space force meshes, optionally using the batched FFT path."""
    if use_batched and _can_use_batched_gradient_fft(conf):
        grad_meshes = _batched_gradient_meshes_from_potential(pot, conf, gradient_kernel)
        return tuple(grad_meshes[i] for i in range(grad_meshes.shape[0]))

    grad_meshes = []
    for k in conf.kvec:
        grad = neg_grad(k, pot, conf.cell_size, gradient_kernel)
        if conf.compute_mesh is None:
            grad = jnp.fft.irfftn(grad)
        else:
            grad = conf.mGPU_irfftn_transposed(grad)
        grad_meshes.append(grad.astype(conf.float_dtype))
    return tuple(grad_meshes)


def _acceleration_from_potential(
    pot, ptcl, conf: Configuration, use_batched=True, use_vmap_gather=False, gradient_kernel="spectral",
):
    """Gather force-mesh components at particle positions."""
    if use_batched and _can_use_batched_gradient_fft(conf):
        grad_meshes = _batched_gradient_meshes_from_potential(pot, conf, gradient_kernel)
        return gather_stacked_mesh_halo(ptcl, conf, jnp.moveaxis(grad_meshes, 0, -1))

    grad_meshes = _gradient_meshes_from_potential(pot, conf, use_batched=False, gradient_kernel=gradient_kernel, )

    if use_vmap_gather:
        stacked_grad_meshes = jnp.stack(grad_meshes, axis=0)
        return jax.vmap(lambda mesh: gather(ptcl, conf, mesh), in_axes=0, out_axes=-1)(stacked_grad_meshes)

    acc = [gather(ptcl, conf, grad) for grad in grad_meshes]
    return jnp.stack(acc, axis=-1)


def _acceleration_from_density_hat(dens_hat, ptcl, conf: Configuration):
    """Evaluate acceleration directly from the density FFT for the continuum force."""
    if not conf.replicated_mesh and not _can_use_batched_gradient_fft(conf):
        # Do not build a three-component spectral stack when the corresponding
        # real stack exceeds the signed 32-bit indexing range.  This matters
        # even before the inverse transform for meshes such as 2560^3, where
        # three local spectral slabs also exceed INT32_MAX elements.  Consume
        # one derivative, inverse transform, and gather at a time instead.
        pot = laplace_transposed_with_kernel(conf.kvec, dens_hat, conf)
        return _streamed_acceleration_from_potential(pot, ptcl, conf)

    spectral_grads = _spectral_gradient_components_from_density_hat(dens_hat, conf)
    if conf.replicated_mesh:
        grad_meshes = jnp.fft.irfftn(spectral_grads, axes=(1, 2, 3)).astype(conf.float_dtype)
        acc = jnp.stack([_gather(ptcl.pmid, ptcl.disp, conf, mesh, 0, 0, None) for mesh in grad_meshes], axis=-1, )
        if ptcl.unused_index is None:
            return acc
        mask = ptcl.unused_index.reshape(ptcl.unused_index.shape + (1, ) * (acc.ndim - 1))
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
        pot, ptcl, conf, use_batched=use_batched, use_vmap_gather=correction is not None and not use_batched,
        gradient_kernel=force_gradient_kernel(correction),
    )


def gravity_streamed(a, ptcl, cosmo, conf: Configuration, correction=None):
    """Evaluate an uncorrected PM force with sequential component FFTs.

    This forward-oriented path avoids constructing stacked three-component
    spectral and real meshes.  It intentionally supports only the ordinary
    uncorrected force; correction and interlacing variants keep using
    :func:`gravity` until they have their own memory-qualified implementation.
    """
    if correction is not None:
        raise ValueError("streamed gravity currently requires correction=None.")
    if conf.compute_mesh is not None and conf.mGPU_irfftn_transposed is None:
        raise ValueError("streamed distributed gravity requires the scalar distributed inverse FFT.")

    dens = scatter(ptcl, conf)
    pot = _gravity_potential_from_density(dens, cosmo.Omega_m, conf, a=a, cosmo=cosmo, correction=None)
    return _streamed_acceleration_from_potential(pot, ptcl, conf)


def _gravity_mesh_fields_from_density(dens, omega_m, conf: Configuration, a=None, cosmo=None, correction=None):
    """Return the real-space force meshes generated by a density field."""
    pot = _gravity_potential_from_density(dens, omega_m, conf, a=a, cosmo=cosmo, correction=correction)
    return _gradient_meshes_from_potential(pot, conf, gradient_kernel=force_gradient_kernel(correction))


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
        shard_map, mesh=conf.compute_mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME), P(AXIS_NAME, None), None,
                  ), out_specs=P(AXIS_NAME, None), check_vma=False,
    )
    def reduce_local(disp_cot_local, pmid_local, unused_local, disp_local, conf_local):
        """Accumulate local displacement cotangents for owned particles.

        Parameters
        ----------
        disp_cot_local
            Local-device shard of the corresponding distributed value.
        pmid_local
            Local-device shard of the corresponding distributed value.
        unused_local
            Local-device shard of the corresponding distributed value.
        disp_local
            Local-device shard of the corresponding distributed value.
        conf_local
            Local-device shard of the corresponding distributed value.
        """
        valid_mask = ~unused_local
        return reduce_grad_across_gpus(disp_cot_local, pmid_local, disp_local, valid_mask, conf_local)

    unused_index = (
        jnp.zeros(disp_cot.shape[0], dtype=jnp.bool_) if unused_index is None else jax.lax.stop_gradient(unused_index)
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
        shard_map, mesh=conf.compute_mesh,
        in_specs=(P(AXIS_NAME, None), P(AXIS_NAME, None), P(AXIS_NAME), P(AXIS_NAME, None), None,
                  ), out_specs=P(AXIS_NAME, None), check_vma=False,
    )
    def count_local(counts_local, pmid_local, unused_local, disp_local, conf_local):
        """Count local gather contributions for owned particles.

        Parameters
        ----------
        counts_local
            Local-device shard of the corresponding distributed value.
        pmid_local
            Local-device shard of the corresponding distributed value.
        unused_local
            Local-device shard of the corresponding distributed value.
        disp_local
            Local-device shard of the corresponding distributed value.
        conf_local
            Local-device shard of the corresponding distributed value.
        """
        valid_mask = ~unused_local
        return reduce_grad_across_gpus(counts_local, pmid_local, disp_local, valid_mask, conf_local)

    unused_index = (
        jnp.zeros(ptcl.disp.shape[0], dtype=jnp.bool_)
        if ptcl.unused_index is None else jax.lax.stop_gradient(ptcl.unused_index)
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
        return _acceleration_from_potential(pot, ptcl, conf, gradient_kernel=force_gradient_kernel(correction), )

    dens = scatter(ptcl, conf)
    return _gravity_from_density(dens, ptcl, cosmo, conf, a=a, correction=correction)
