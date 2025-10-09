from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp

from .configuration import Configuration
from .fft import fftfwd, fftinv, fftfreq
from .scatter import scatter
from .gather import gather


@custom_vjp
def laplace(kvec, src, cosmo=None):
    """Laplace kernel in Fourier space."""
    k2 = sum(k ** 2 for k in kvec)

    pot = jnp.where(k2 != 0, - src / k2, 0)

    return pot


def laplace_fwd(kvec, src, cosmo):
    pot = laplace(kvec, src, cosmo)
    return pot, (kvec, cosmo)


def laplace_bwd(res, pot_cot):
    """Custom vjp to avoid NaN when using where, as well as to save memory.

    .. _JAX FAQ:
        https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where

    """
    kvec, cosmo = res
    src_cot = laplace(kvec, pot_cot, cosmo)
    return None, src_cot, None


laplace.defvjp(laplace_fwd, laplace_bwd)


def neg_grad(k, pot, spacing):
    nyquist = jnp.pi / spacing
    eps = nyquist * jnp.finfo(k.dtype).eps
    neg_ik = jnp.where(jnp.abs(jnp.abs(k) - nyquist) <= eps, 0, -1j * k)

    grad = jnp.multiply(neg_ik, pot)

    return grad


def gravity(a, ptcl, cosmo, conf: Configuration):
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""
    kvec = conf.kvec # fftfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)

    gather_cp = jax.checkpoint(gather, static_argnums=(1,))

    dens = scatter(ptcl, conf)
    dens -= 1
    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = conf.mGPU_rfftn(dens)


    pot = laplace(kvec, dens, cosmo)

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)

        grad = conf.mGPU_irfftn(grad)

        grad = grad.astype(conf.float_dtype)  # no jnp.complex32

        grad = gather_cp(ptcl, conf, grad)

        acc.append(grad)
    acc = jnp.stack(acc, axis=-1)

    return acc
