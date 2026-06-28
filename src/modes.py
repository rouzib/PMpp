from functools import partial

import jax
from jax import checkpoint, custom_vjp, NamedSharding, jit
from jax import random
import jax.numpy as jnp
from jax._src.numpy.fft import _fft_norm
from jax.sharding import PartitionSpec as P

from .utils import AXIS_NAME
from .boltzmann import linear_power


@partial(jax.jit, static_argnames=('conf', 'real', 'unit_abs'))
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
    if conf.compute_mesh is not None:
        modes = jax.lax.with_sharding_constraint(modes, NamedSharding(conf.compute_mesh, P(AXIS_NAME)))

    if real and not unit_abs:
        return modes

    if conf.compute_mesh is None:
        modes = jnp.fft.rfftn(modes)
    else:
        modes = conf.mGPU_rfftn_transposed(modes)
    modes *= _fft_norm(s=jnp.array(conf.ptcl_grid_shape, dtype=modes.dtype), func_name="rfftn", norm="ortho")

    if unit_abs:
        modes /= jnp.abs(modes)

    if real:
        if conf.compute_mesh is None:
            modes = jnp.fft.irfftn(modes, s=conf.ptcl_grid_shape)
        else:
            modes = conf.mGPU_irfftn_transposed(modes)

    return modes


def _complex_dtype_for(float_dtype):
    """Return the complex dtype that preserves the requested real precision."""
    if jnp.dtype(float_dtype) == jnp.float64:
        return jnp.complex128
    return jnp.complex64


def _fft_mode_numbers(size):
    """Integer Fourier labels for a full FFT axis.

    JAX/NumPy FFT output stores positive modes first and wraps negative modes
    at the end of the axis. Nested white noise needs the physical integer
    label, not the storage index, so that the same low-k mode hashes to the
    same value at different grid resolutions.
    """
    idx = jnp.arange(size, dtype=jnp.int32)
    return jnp.where(idx < (size + 1) // 2, idx, idx - size)


def _rfft_mode_numbers(size):
    """Integer Fourier labels for the non-negative axis of an rFFT output."""
    return jnp.arange(size // 2 + 1, dtype=jnp.int32)


def _mix_uint32(x):
    """Avalanche-mix uint32 values with the lowbias32 integer finalizer.

    The constants ``0x7FEB352D`` and ``0x846CA68B`` and the shift pattern
    ``16, 15, 16`` come from Chris Wellons' ``lowbias32`` mixer, which was
    found by automated search and tuned for good avalanche behavior. The middle
    shift is ``15`` rather than ``16`` because that specific combination tested
    better; the shifts and multipliers should be viewed as one optimized set,
    not chosen independently. (see https://nullprogram.com/blog/2018/07/31/)

    This is not a cryptographic hash. Here it is just used to deterministically
    scramble integer mode coordinates into pseudo-random-looking uint32 values.
    """
    x = jnp.asarray(x, dtype=jnp.uint32)
    x = jnp.bitwise_xor(x, x >> 16)
    x *= jnp.uint32(0x7FEB352D)
    x = jnp.bitwise_xor(x, x >> 15)
    x *= jnp.uint32(0x846CA68B)
    x = jnp.bitwise_xor(x, x >> 16)
    return x


def _hash_mode_u32(seed, kx, ky, kz, salt):
    """Hash ``(seed, kx, ky, kz, salt)`` into one deterministic uint32 value.

    This builds a mode-local random stream by repeatedly applying
    ``_mix_uint32``. The per-axis constants ``0x9E3779B9``, ``0x85EBCA6B``,
    and ``0xC2B2AE35`` are standard large odd hash constants, used here as
    simple offsets so that ``kx``, ``ky``, and ``kz`` do not enter the mixer
    in overly similar forms. ``salt`` is an extra stream identifier, letting
    us derive independent values for the same Fourier mode, such as separate
    real and imaginary Gaussian variates.
    """
    h = _mix_uint32(jnp.uint32(seed) ^ jnp.uint32(salt))
    h = _mix_uint32(h ^ _mix_uint32(jnp.asarray(kx, dtype=jnp.uint32) + jnp.uint32(0x9E3779B9)))
    h = _mix_uint32(h ^ _mix_uint32(jnp.asarray(ky, dtype=jnp.uint32) + jnp.uint32(0x85EBCA6B)))
    h = _mix_uint32(h ^ _mix_uint32(jnp.asarray(kz, dtype=jnp.uint32) + jnp.uint32(0xC2B2AE35)))
    return h


def _hash_to_uniform(hash_value, dtype):
    """Map a uint32 hash to a uniform variate in ``(0, 1)``.

    The scale factor ``2.3283064365386963e-10`` is exactly ``1 / 2**32`` in
    float form, since a uint32 hash spans ``2**32`` possible values. The
    added ``0.5`` maps each integer to the center of its bin, so the result
    stays strictly inside ``(0, 1)`` rather than ever landing exactly on 0
    or 1.
    """
    dtype = jnp.dtype(dtype)
    return (hash_value.astype(dtype) + jnp.asarray(0.5, dtype=dtype)) * jnp.asarray(2.3283064365386963e-10, dtype=dtype)


def _box_muller(hash_real, hash_imag, dtype):
    """Convert two hashed uint32 streams into standard-normal variates."""
    dtype = jnp.dtype(dtype)
    u1 = _hash_to_uniform(hash_real, dtype)
    u2 = _hash_to_uniform(hash_imag, dtype)
    radius = jnp.sqrt(jnp.asarray(-2.0, dtype=dtype) * jnp.log(u1))
    theta = jnp.asarray(2 * jnp.pi, dtype=dtype) * u2
    return radius * jnp.cos(theta), radius * jnp.sin(theta)


def _is_self_inverse_mode(k_mode, size):
    """Return whether a 1D Fourier mode equals its own negative on the grid."""
    mask = k_mode == 0
    if size % 2 == 0:
        mask = mask | (k_mode == -(size // 2))
    return mask


def _canonical_xy_pair(kx, ky, shape):
    """Choose one representative from each Hermitian pair on self-conjugate planes.

    In an rFFT array, modes with ``kz = 0`` and, for even grids, ``kz = N/2``
    must satisfy ``F(kx, ky, kz) = conj(F(-kx, -ky, kz))``. This helper
    selects the canonical ``(kx, ky)`` member to hash, tells the caller whether
    the stored mode should be conjugated, and identifies points whose imaginary
    part must be exactly zero.
    """
    partner_x = jnp.where(_is_self_inverse_mode(kx, shape[0]), kx, -kx)
    partner_y = jnp.where(_is_self_inverse_mode(ky, shape[1]), ky, -ky)
    keep_self = (kx < partner_x) | ((kx == partner_x) & (ky <= partner_y))
    rep_x = jnp.where(keep_self, kx, partner_x)
    rep_y = jnp.where(keep_self, ky, partner_y)
    self_point = _is_self_inverse_mode(kx, shape[0]) & _is_self_inverse_mode(ky, shape[1])
    return rep_x, rep_y, ~keep_self, self_point


def _nested_fourier_modes_from_numbers(seed, conf, kx_modes, ky_modes, kz_modes, unit_abs):
    """Build resolution-consistent white-noise Fourier coefficients.

    Coefficients are generated from integer Fourier coordinates instead of a
    sequential PRNG draw. The physical mode ``(1, 2, 3)`` therefore receives
    the same random value on a 256^3 and 512^3 grid when the box size and seed
    match. The extra Hermitian handling keeps the inverse rFFT real.
    """
    float_dtype = jnp.dtype(conf.float_dtype)
    complex_dtype = _complex_dtype_for(float_dtype)
    shape = tuple(int(s) for s in conf.ptcl_grid_shape)

    kx = kx_modes[:, None, None]
    ky = ky_modes[None, :, None]
    kz = kz_modes[None, None, :]

    plane_self_conj = kz == 0
    if shape[2] % 2 == 0:
        plane_self_conj = plane_self_conj | (kz == shape[2] // 2)

    rep_x, rep_y, need_conj, self_point = _canonical_xy_pair(kx, ky, shape)
    hash_x = jnp.where(plane_self_conj, rep_x, kx)
    hash_y = jnp.where(plane_self_conj, rep_y, ky)

    gaussian_real, gaussian_imag = _box_muller(
        _hash_mode_u32(seed, hash_x, hash_y, kz, salt=0xA24BAED4),
        _hash_mode_u32(seed, hash_x, hash_y, kz, salt=0x9FB21C65),
        float_dtype,
    )

    sqrt_half = jnp.asarray(0.7071067811865476, dtype=float_dtype)
    coeff = jax.lax.complex(gaussian_real, gaussian_imag) * sqrt_half
    plane_coeff = jnp.where(self_point, jax.lax.complex(gaussian_real, jnp.zeros_like(gaussian_imag)), coeff)
    plane_coeff = jnp.where(need_conj, jnp.conj(plane_coeff), plane_coeff)
    modes = jnp.where(plane_self_conj, plane_coeff, coeff)

    if unit_abs:
        norm = jnp.abs(modes)
        modes = jnp.where(norm != 0, modes / norm, jnp.ones_like(modes))

    return modes


@partial(jax.jit, static_argnames=('conf', 'real', 'unit_abs'))
def white_noise_nested(seed, conf, real=False, unit_abs=False):
    """Nested white noise Fourier or real modes.

    Parameters
    ----------
    seed : int
        Seed for the mode-local deterministic hash stream.
    conf : Configuration
        Active simulation configuration.
    real : bool, optional
        Whether to return a real-space field instead of Fourier coefficients.
    unit_abs : bool, optional
        Whether to normalize Fourier coefficients to unit modulus.

    Returns
    -------
    jax.Array
        White-noise field in Fourier or real space.

    Notes
    -----
    This path is resolution-consistent for fixed box size: the shared
    non-Nyquist low-k Fourier modes are identical across resolutions for the
    same ``seed``.
    """
    kx_modes = _fft_mode_numbers(conf.ptcl_grid_shape[0])
    ky_modes = _fft_mode_numbers(conf.ptcl_grid_shape[1])
    kz_modes = _rfft_mode_numbers(conf.ptcl_grid_shape[2])

    if conf.compute_mesh is not None:
        kx_modes = jax.lax.with_sharding_constraint(kx_modes, NamedSharding(conf.compute_mesh, P(AXIS_NAME)))

    modes = _nested_fourier_modes_from_numbers(seed, conf, kx_modes, ky_modes, kz_modes, unit_abs)

    if conf.compute_mesh is not None:
        modes = _to_transposed_spectral_layout(modes, conf)

    if real:
        if conf.compute_mesh is not None:
            modes = conf.mGPU_irfftn_transposed(modes)
            modes *= _fft_norm(s=jnp.array(conf.ptcl_grid_shape, dtype=modes.dtype), func_name="irfftn", norm="ortho")
        else:
            modes = jnp.fft.irfftn(modes, s=conf.ptcl_grid_shape, norm="ortho")

    return modes


@custom_vjp
def _safe_sqrt(x):
    """Square root whose custom VJP returns zero derivative at exactly zero."""
    return jnp.sqrt(x)


def _safe_sqrt_fwd(x):
    """Forward rule for ``_safe_sqrt``."""
    y = _safe_sqrt(x)
    return y, y


def _safe_sqrt_bwd(y, y_cot):
    """Avoid the infinite ``0.5 / sqrt(x)`` cotangent at ``x == 0``."""
    x_cot = jnp.where(y != 0, 0.5 / y * y_cot, 0)
    return (x_cot,)


_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_bwd)


def _to_transposed_spectral_layout(modes, conf):
    """Apply the sharding layout expected by distributed transposed rFFT data."""
    if conf.compute_mesh is None:
        return modes
    return jax.lax.with_sharding_constraint(
        modes,
        NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)),
    )


def get_k_magnitude(kvec, conf):
    """Return ``|k|`` on the standard spectral layout without dense all-gather.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Sparse broadcastable Fourier wavevector components.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Magnitude of the wavevector on the standard spectral layout.
    """
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]
    if conf.compute_mesh is None:
        return jnp.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(
            conf.float_dtype
        )

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

        local_shard = jnp.sqrt(kx_b ** 2 + ky_b ** 2 + kz_b ** 2)
        return local_shard.astype(conf.float_dtype)

    return create_k_magnitude_sharded(kx, ky, kz)


def get_k_magnitude_transposed(kvec, conf):
    """Return ``|k|`` on the transposed spectral layout used by distributed FFTs.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Sparse broadcastable Fourier wavevector components.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    jax.Array
        Magnitude of the wavevector on the transposed spectral layout.
    """
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]
    if conf.compute_mesh is None:
        return jnp.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(
            conf.float_dtype
        )

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

        local_shard = jnp.sqrt(kx_b ** 2 + ky_b ** 2 + kz_b ** 2)
        return local_shard.astype(conf.float_dtype)

    return create_k_magnitude_transposed(kx, ky, kz)


@partial(jit, static_argnums=4)
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
    k = get_k_magnitude_transposed(kvec, conf)

    if a is not None:
        a = jnp.asarray(a, dtype=conf.float_dtype)

    Plin = linear_power(k, a, cosmo, conf)

    if jnp.isrealobj(modes):
        if conf.compute_mesh is None:
            modes = jnp.fft.rfftn(modes)
        else:
            modes = conf.mGPU_rfftn_transposed(modes)
        modes *= _fft_norm(s=jnp.array(conf.ptcl_grid_shape, dtype=modes.dtype), func_name="rfftn", norm="ortho")
    else:
        modes = _to_transposed_spectral_layout(modes, conf)

    modes *= _safe_sqrt(Plin * conf.box_vol)

    if real:
        if conf.compute_mesh is None:
            modes = jnp.fft.irfftn(modes, s=conf.ptcl_grid_shape)
        else:
            modes = conf.mGPU_irfftn_transposed(modes)

    return modes
