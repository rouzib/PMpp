from __future__ import annotations

import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .scatter import scatter
from .utils import AXIS_NAME

_MAS_POWER = {
    None: 0,
    "NGP": 1,
    "CIC": 2,
    "TSC": 3,
    "PCS": 4,
}


def _normalize_mas(mas: str | None) -> str | None:
    if mas is None:
        return None
    mas = mas.upper()
    if mas not in _MAS_POWER:
        raise ValueError(f"Unsupported MAS={mas!r}. Expected one of {tuple(k for k in _MAS_POWER if k is not None)}.")
    return mas


def _box_size_1d(conf) -> float:
    return float(conf.box_size[0])


def _fundamental_mode(conf):
    return jnp.asarray(2 * jnp.pi / _box_size_1d(conf), dtype=conf.float_dtype)


def _max_shell_index(conf) -> int:
    with jax.ensure_compile_time_eval():
        nyquist = conf.mesh_shape[0] // 2
        return math.floor(math.sqrt(3) * nyquist)


@lru_cache(maxsize=None)
def _shell_statistics(shape: tuple[int, int, int], cell_size: float):
    box_size = float(shape[0]) * float(cell_size)
    k_fundamental = 2 * math.pi / box_size

    mx = np.fft.fftfreq(shape[0]) * shape[0]
    my = np.fft.fftfreq(shape[1]) * shape[1]
    mz = np.fft.rfftfreq(shape[2]) * shape[2]
    mode_x, mode_y, mode_z = np.meshgrid(mx, my, mz, indexing="ij")
    mode_mag = np.sqrt(mode_x ** 2 + mode_y ** 2 + mode_z ** 2)
    shell = np.floor(mode_mag + 1e-6).astype(np.int32)

    multiplicity = np.full(shell.shape, 2, dtype=np.int32)
    multiplicity[..., 0] = 1
    if shape[2] % 2 == 0:
        multiplicity[..., -1] = 1

    num_shells = int(shell.max()) + 1
    nmodes = np.bincount(shell.reshape(-1), weights=multiplicity.reshape(-1), minlength=num_shells).astype(np.int32)
    k_shell = np.bincount(
        shell.reshape(-1),
        weights=(mode_mag * multiplicity).reshape(-1),
        minlength=num_shells,
    )
    valid = np.maximum(nmodes, 1)
    k_shell = (k_shell / valid) * k_fundamental
    return k_shell[1:], nmodes[1:]


def _shell_vectors(conf):
    k_shell, nmodes = _shell_statistics(tuple(int(s) for s in conf.mesh_shape), float(conf.cell_size))
    return (
        lax.stop_gradient(jnp.asarray(k_shell, dtype=conf.float_dtype)),
        lax.stop_gradient(jnp.asarray(nmodes, dtype=jnp.int32)),
    )


def _fft_mode_numbers(size: int, *, real_axis: bool, dtype):
    half = size // 2
    if real_axis:
        return jnp.arange(half + 1, dtype=dtype)
    if size % 2 == 0:
        return jnp.concatenate(
            (
                jnp.arange(0, half, dtype=dtype),
                jnp.arange(-half, 0, dtype=dtype),
            )
        )
    return jnp.concatenate(
        (
            jnp.arange(0, half + 1, dtype=dtype),
            jnp.arange(-half, 0, dtype=dtype),
        )
    )


def _mode_numbers_transposed(conf):
    mode_dtype = jnp.int32
    return (
        lax.stop_gradient(_fft_mode_numbers(conf.mesh_shape[0], real_axis=False, dtype=mode_dtype)),
        lax.stop_gradient(_fft_mode_numbers(conf.mesh_shape[1], real_axis=False, dtype=mode_dtype)),
        lax.stop_gradient(_fft_mode_numbers(conf.mesh_shape[2], real_axis=True, dtype=mode_dtype)),
    )


def _mas_power_deconvolution(mode_x, mode_y, mode_z, conf, mas_power: int):
    if mas_power <= 0:
        return None

    # On the FFT mesh, k * cell_size / (2 pi) == integer_mode / nmesh per axis.
    sx = jnp.sinc(mode_x[:, None, None].astype(conf.float_dtype) / conf.mesh_shape[0])
    sy = jnp.sinc(mode_y[None, :, None].astype(conf.float_dtype) / conf.mesh_shape[1])
    sz = jnp.sinc(mode_z[None, None, :].astype(conf.float_dtype) / conf.mesh_shape[2])

    window = sx * sy * sz
    return window ** (-2 * mas_power)


def _shell_reduce_local(spectral, mode_x, mode_y, mode_z, conf, mas_power: int, num_shells: int):
    mode_sq = (
        mode_x[:, None, None].astype(jnp.int64) ** 2
        + mode_y[None, :, None].astype(jnp.int64) ** 2
        + mode_z[None, None, :].astype(jnp.int64) ** 2
    )
    mode_mag = jnp.sqrt(mode_sq.astype(conf.float_dtype))
    shell = jnp.floor(jnp.sqrt(mode_sq.astype(jnp.float64)) + 1e-9).astype(jnp.int32)
    shell = jnp.clip(shell, 0, num_shells - 1)

    multiplicity = jnp.full(spectral.shape, 2, dtype=jnp.int32)
    multiplicity = multiplicity.at[..., 0].set(1)
    if conf.mesh_shape[-1] % 2 == 0:
        multiplicity = multiplicity.at[..., -1].set(1)

    power = (spectral.real ** 2 + spectral.imag ** 2).astype(conf.float_dtype)
    deconv = _mas_power_deconvolution(mode_x, mode_y, mode_z, conf, mas_power)
    if deconv is not None:
        power = power * deconv.astype(conf.float_dtype)

    mult_f = multiplicity.astype(conf.float_dtype)
    shell_flat = shell.reshape(-1)
    p_sum = jnp.bincount(shell_flat, weights=(power * mult_f).reshape(-1), length=num_shells)
    return p_sum


def _shell_reduce_transposed(spectral, conf, mas_power: int, num_shells: int):
    mode_x, mode_y, mode_z = _mode_numbers_transposed(conf)
    if conf.compute_mesh is None or conf.num_devices == 1:
        return _shell_reduce_local(spectral, mode_x, mode_y, mode_z, conf, mas_power, num_shells)

    @partial(
        shard_map,
        mesh=conf.compute_mesh,
        in_specs=(
            P(None, AXIS_NAME, None),
            P(None),
            P(AXIS_NAME),
            P(None),
        ),
        out_specs=(
            P(None),
        ),
        check_rep=False,
    )
    def local_reduce(spectral_local, mode_x_rep, mode_y_local, mode_z_rep):
        p_sum_local = _shell_reduce_local(
            spectral_local,
            mode_x_rep,
            mode_y_local,
            mode_z_rep,
            conf,
            mas_power,
            num_shells,
        )
        return (lax.psum(p_sum_local, AXIS_NAME),)

    (p_sum,) = local_reduce(spectral, mode_x, mode_y, mode_z)
    return p_sum


def _finalize_shell_averages(p_sum, conf):
    k, nmodes = _shell_vectors(conf)
    pk = p_sum[1:] / nmodes.astype(conf.float_dtype)
    pk = pk * (conf.cell_size ** 3 / conf.mesh_size)
    return k, pk.astype(conf.float_dtype), nmodes


def _spectral_delta(delta, conf):
    if conf.compute_mesh is None:
        return jnp.fft.rfftn(delta)
    return conf.mGPU_rfftn_transposed(delta)


@partial(jax.jit, static_argnames=("conf", "mas"))
def delta_to_pk(delta, conf, mas: str | None = "CIC"):
    """Compute a differentiable isotropic 1D power spectrum from an overdensity field.

    Parameters
    ----------
    delta : jax.Array
        Overdensity field on the PM mesh.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme used to create the field. This controls Fourier-space
        deconvolution and should match the scatter rule. Use `None` to disable
        deconvolution.

    Returns
    -------
    k : jax.Array
        Shell-averaged physical wavenumbers.
    pk : jax.Array
        Shell-averaged isotropic monopole.
    nmodes : jax.Array
        Number of contributing Fourier modes per shell.
    """
    mas = _normalize_mas(mas)
    spectral = _spectral_delta(jnp.asarray(delta, dtype=conf.float_dtype), conf)
    num_shells = _max_shell_index(conf) + 1
    p_sum = _shell_reduce_transposed(spectral, conf, _MAS_POWER[mas], num_shells)
    return _finalize_shell_averages(p_sum, conf)


@partial(jax.jit, static_argnames=("conf", "mas"))
def density_to_pk(density, conf, mas: str | None = "CIC"):
    """Compute a differentiable power spectrum from a density field."""
    density = jnp.asarray(density, dtype=conf.float_dtype)
    mean_density = jnp.mean(density, dtype=conf.float_dtype)
    delta = density / mean_density - 1
    return delta_to_pk(delta, conf, mas=mas)


@partial(jax.jit, static_argnames=("conf", "mas"))
def particles_to_pk(ptcl, conf, mas: str | None = "CIC"):
    """Scatter particles to the PM mesh and return the differentiable P(k)."""
    density = scatter(ptcl, conf)
    return density_to_pk(density, conf, mas=mas)


# Backward-compatible aliases for the earlier public naming.
delta_to_quijote_pk = delta_to_pk
density_to_quijote_pk = density_to_pk
particles_to_quijote_pk = particles_to_pk
