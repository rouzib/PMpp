from functools import partial

import jax
from jax import custom_vjp
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from .configuration import Configuration
from .scatter import scatter
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


def gravity(a, ptcl, cosmo, conf: Configuration):
    """Gravitational accelerations of particles in [H_0^2], solved on a mesh with FFT."""
    kvec = conf.kvec  # fftfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)

    # gather_cp = jax.checkpoint(gather, static_argnums=(1,))

    dens = scatter(ptcl, conf)
    dens -= 1
    dens *= 1.5 * cosmo.Omega_m.astype(conf.float_dtype)

    dens = conf.mGPU_rfftn(dens)

    pot = laplace(kvec, dens, conf, cosmo)

    acc = []
    for k in kvec:
        grad = neg_grad(k, pot, conf.cell_size)

        grad = conf.mGPU_irfftn(grad)

        grad = grad.astype(conf.float_dtype)  # no jnp.complex32

        # grad = gather_cp(ptcl, conf, grad)
        grad = gather(ptcl, conf, grad)

        acc.append(grad)
    acc = jnp.stack(acc, axis=-1)

    return acc
