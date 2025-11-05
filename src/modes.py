from functools import partial

import jax
from jax import checkpoint, custom_vjp, NamedSharding, jit
from jax import random
import jax.numpy as jnp
from jax._src.numpy.fft import _fft_norm
from jax.sharding import PartitionSpec as P

from .utils import AXIS_NAME
from .boltzmann import linear_power


# TODO follow pmesh to fill the modes in Fourier space
@partial(jax.jit, static_argnames=('real', 'unit_abs'))
def white_noise(seed, conf, real=False, unit_abs=False):
    """White noise Fourier or real modes.

    Parameters
    ----------
    seed : int
        Seed for the pseudo-random number generator.
    conf : Configuration
    real : bool, optional
        Whether to return real or Fourier modes.
    unit_abs : bool, optional
        Whether to set the absolute values to 1.

    Returns
    -------
    modes : jax.Array of conf.float_dtype
        White noise Fourier or real modes, both dimensionless with zero mean and unit
        variance.

    """
    key = random.PRNGKey(seed)

    # sample linear modes on Lagrangian particle grid
    modes = random.normal(key, shape=conf.ptcl_grid_shape, dtype=conf.float_dtype)
    # modes = jax.lax.with_sharding_constraint(modes, NamedSharding(conf.compute_mesh, P(AXIS_NAME)))

    if real and not unit_abs:
        return modes

    modes = conf.mGPU_rfftn(modes)
    modes *= _fft_norm(s=jnp.array(conf.ptcl_grid_shape, dtype=modes.dtype), func_name="rfftn", norm="ortho")

    if unit_abs:
        modes /= jnp.abs(modes)

    if real:
        modes = conf.mGPU_irfftn(modes)

    return modes


@custom_vjp
def _safe_sqrt(x):
    return jnp.sqrt(x)


def _safe_sqrt_fwd(x):
    y = _safe_sqrt(x)
    return y, y


def _safe_sqrt_bwd(y, y_cot):
    x_cot = jnp.where(y != 0, 0.5 / y * y_cot, 0)
    return (x_cot,)


_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_bwd)


# @partial(jit, static_argnums=4)
# @partial(checkpoint, static_argnums=4)
def linear_modes(modes, cosmo, conf, a=None, real=False):
    """Linear matter overdensity Fourier or real modes.

    Parameters
    ----------
    modes : jax.Array
        Fourier or real modes with white noise prior.
    cosmo : Cosmology
    conf : Configuration
    a : float or None, optional
        Scale factors. Default (None) is to not scale the output modes by growth.
    real : bool, optional
        Whether to return real or Fourier modes.

    Returns
    -------
    modes : jax.Array of conf.float_dtype
        Linear matter overdensity Fourier or real modes, in [L^3] or dimensionless,
        respectively.

    Notes
    -----

    .. math::

        \delta(\mathbf{k}) = \sqrt{V P_\mathrm{lin}(k)} \omega(\mathbf{k})

    """
    kvec = conf.kvec_spacing

    # @partial(
    #     jax.jit,
    #     in_shardings=(NamedSharding(conf.compute_mesh, P(AXIS_NAME)),
    #                   NamedSharding(conf.compute_mesh, P()),
    #                   NamedSharding(conf.compute_mesh, P())),
    #     out_shardings=NamedSharding(conf.compute_mesh, P(AXIS_NAME)),
    # )
    # def sharded_sum(*kparts):
    #     bparts = jnp.broadcast_arrays(*kparts)
    #     return jnp.linalg.norm(jnp.stack(bparts, axis=0), axis=0)
    #
    # k = sharded_sum(*kvec)
    # print(k.shape)

    kx, ky, kz = [jnp.squeeze(a) for a in kvec]  # shapes: (Nx,), (Ny,), (Nz,)
    shape = (len(kx), len(ky), len(kz))  # (Nx, Ny, Nz)
    out_sharding = NamedSharding(conf.compute_mesh, P(AXIS_NAME))  # match your layout

    def fill_shard(index):
        # index is a tuple of slices for this shard: (sx, sy, sz)
        sx, sy, sz = index
        kx_s = kx[sx]  # (nx,)
        ky_s = ky[sy]  # (ny,)
        kz_s = kz[sz]  # (nz,)

        # Local (small) broadcasts happen *inside* the shard only:
        kx_b = kx_s[:, None, None]  # (nx,1,1)
        ky_b = ky_s[None, :, None]  # (1,ny,1)
        kz_b = kz_s[None, None, :]  # (1,1,nz)
        return jnp.sqrt(kx_b * kx_b + ky_b * ky_b + kz_b * kz_b).astype(conf.float_dtype)

    # Each device calls fill_shard with its own slice; allocation is on-device.
    k = jax.make_array_from_callback(shape, out_sharding, fill_shard)

    if a is not None:
        a = jnp.asarray(a, dtype=conf.float_dtype)

    Plin = linear_power(k, a, cosmo, conf)

    if jnp.isrealobj(modes):
        modes = conf.mGPU_rfftn(modes)

    modes *= _safe_sqrt(Plin * conf.box_vol)

    if real:
        modes = conf.mGPU_irfftn(modes)

    return modes
