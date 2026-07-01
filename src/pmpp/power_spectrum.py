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
    """Normalize and validate the mass-assignment scheme name."""
    if mas is None:
        return None
    mas = mas.upper()
    if mas not in _MAS_POWER:
        raise ValueError(f"Unsupported MAS={mas!r}. Expected one of {tuple(k for k in _MAS_POWER if k is not None)}.")
    return mas


def _box_size_1d(conf) -> float:
    """Return the scalar box size assumed by isotropic shell binning."""
    return float(conf.box_size[0])


def _fundamental_mode(conf):
    """Return the fundamental wavenumber ``2 pi / L``."""
    return jnp.asarray(2 * jnp.pi / _box_size_1d(conf), dtype=conf.float_dtype)


def _max_shell_index(conf) -> int:
    """Largest integer-radius Fourier shell represented by the mesh."""
    with jax.ensure_compile_time_eval():
        nyquist = conf.mesh_shape[0] // 2
        return math.floor(math.sqrt(3) * nyquist)


@lru_cache(maxsize=None)
def _shell_statistics(shape: tuple[int, int, int], cell_size: float):
    """Host-side shell centers and mode counts for an rFFT mesh.

    The rFFT half-spectrum omits negative ``kz`` modes. ``multiplicity`` counts
    the missing conjugate partners except on self-conjugate ``kz=0`` and
    Nyquist planes, so shell averages match a full complex FFT.
    """
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
    """Return cached shell-center and mode-count arrays on the active backend."""
    k_shell, nmodes = _shell_statistics(tuple(int(s) for s in conf.mesh_shape), float(conf.cell_size))
    return (
        lax.stop_gradient(jnp.asarray(k_shell, dtype=conf.float_dtype)),
        lax.stop_gradient(jnp.asarray(nmodes, dtype=jnp.int32)),
    )


def _fft_mode_numbers(size: int, *, real_axis: bool, dtype):
    """Integer Fourier labels matching FFT storage order."""
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
    """Mode-number vectors for the transposed spectral layout."""
    mode_dtype = jnp.int32
    return (
        lax.stop_gradient(_fft_mode_numbers(conf.mesh_shape[0], real_axis=False, dtype=mode_dtype)),
        lax.stop_gradient(_fft_mode_numbers(conf.mesh_shape[1], real_axis=False, dtype=mode_dtype)),
        lax.stop_gradient(_fft_mode_numbers(conf.mesh_shape[2], real_axis=True, dtype=mode_dtype)),
    )


def _mas_power_deconvolution(mode_x, mode_y, mode_z, conf, mas_power: int):
    """Return the inverse squared mass-assignment window for P(k)."""
    if mas_power <= 0:
        return None

    # On the FFT mesh, k * cell_size / (2 pi) == integer_mode / nmesh per axis.
    sx = jnp.sinc(mode_x[:, None, None].astype(conf.float_dtype) / conf.mesh_shape[0])
    sy = jnp.sinc(mode_y[None, :, None].astype(conf.float_dtype) / conf.mesh_shape[1])
    sz = jnp.sinc(mode_z[None, None, :].astype(conf.float_dtype) / conf.mesh_shape[2])

    window = sx * sy * sz
    return window ** (-2 * mas_power)


def _shell_reduce_local(spectral, mode_x, mode_y, mode_z, conf, mas_power: int, num_shells: int):
    """Accumulate spectral power into integer-radius shells on one local shard."""
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


def _shell_reduce_cross_local(spectral_a, spectral_b, mode_x, mode_y, mode_z, conf, mas_power: int, num_shells: int):
    """Accumulate auto and cross power into integer-radius shells on one shard."""
    mode_sq = (
        mode_x[:, None, None].astype(jnp.int64) ** 2
        + mode_y[None, :, None].astype(jnp.int64) ** 2
        + mode_z[None, None, :].astype(jnp.int64) ** 2
    )
    shell = jnp.floor(jnp.sqrt(mode_sq.astype(jnp.float64)) + 1e-9).astype(jnp.int32)
    shell = jnp.clip(shell, 0, num_shells - 1)

    multiplicity = jnp.full(spectral_a.shape, 2, dtype=jnp.int32)
    multiplicity = multiplicity.at[..., 0].set(1)
    if conf.mesh_shape[-1] % 2 == 0:
        multiplicity = multiplicity.at[..., -1].set(1)

    power_a = (spectral_a.real ** 2 + spectral_a.imag ** 2).astype(conf.float_dtype)
    power_b = (spectral_b.real ** 2 + spectral_b.imag ** 2).astype(conf.float_dtype)
    power_cross = (spectral_a * jnp.conj(spectral_b)).real.astype(conf.float_dtype)
    deconv = _mas_power_deconvolution(mode_x, mode_y, mode_z, conf, mas_power)
    if deconv is not None:
        deconv = deconv.astype(conf.float_dtype)
        power_a = power_a * deconv
        power_b = power_b * deconv
        power_cross = power_cross * deconv

    mult_f = multiplicity.astype(conf.float_dtype)
    shell_flat = shell.reshape(-1)
    weights = mult_f.reshape(-1)
    p_a_sum = jnp.bincount(shell_flat, weights=(power_a.reshape(-1) * weights), length=num_shells)
    p_b_sum = jnp.bincount(shell_flat, weights=(power_b.reshape(-1) * weights), length=num_shells)
    p_cross_sum = jnp.bincount(shell_flat, weights=(power_cross.reshape(-1) * weights), length=num_shells)
    return p_a_sum, p_b_sum, p_cross_sum


def _shell_reduce_transposed(spectral, conf, mas_power: int, num_shells: int):
    """Reduce shell power on single-GPU or distributed transposed spectra."""
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
        """Accumulate local spectral shells before cross-device reduction.

        Parameters
        ----------
        spectral_local
            Local shard of a spectral density field.
        mode_x_rep
            Replicated x-axis shell indices.
        mode_y_local
            Local y-axis shell indices.
        mode_z_rep
            Replicated z-axis shell indices.
        """
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


def _shell_reduce_cross_transposed(spectral_a, spectral_b, conf, mas_power: int, num_shells: int):
    """Reduce shell auto/cross power on single-GPU or distributed spectra."""
    mode_x, mode_y, mode_z = _mode_numbers_transposed(conf)
    if conf.compute_mesh is None or conf.num_devices == 1:
        return _shell_reduce_cross_local(spectral_a, spectral_b, mode_x, mode_y, mode_z, conf, mas_power, num_shells)

    @partial(
        shard_map,
        mesh=conf.compute_mesh,
        in_specs=(
            P(None, AXIS_NAME, None),
            P(None, AXIS_NAME, None),
            P(None),
            P(AXIS_NAME),
            P(None),
        ),
        out_specs=(
            P(None),
            P(None),
            P(None),
        ),
        check_rep=False,
    )
    def local_reduce(spectral_a_local, spectral_b_local, mode_x_rep, mode_y_local, mode_z_rep):
        """Accumulate local spectral shells before cross-device reduction.

        Parameters
        ----------
        spectral_a_local
            Local shard of the first spectral density field.
        spectral_b_local
            Local shard of the second spectral density field.
        mode_x_rep
            Replicated x-axis shell indices.
        mode_y_local
            Local y-axis shell indices.
        mode_z_rep
            Replicated z-axis shell indices.
        """
        p_a_sum_local, p_b_sum_local, p_cross_sum_local = _shell_reduce_cross_local(
            spectral_a_local,
            spectral_b_local,
            mode_x_rep,
            mode_y_local,
            mode_z_rep,
            conf,
            mas_power,
            num_shells,
        )
        return (
            lax.psum(p_a_sum_local, AXIS_NAME),
            lax.psum(p_b_sum_local, AXIS_NAME),
            lax.psum(p_cross_sum_local, AXIS_NAME),
        )

    return local_reduce(spectral_a, spectral_b, mode_x, mode_y, mode_z)


def _finalize_shell_averages(p_sum, conf):
    """Convert shell power sums into physical ``P(k)`` normalization."""
    k, nmodes = _shell_vectors(conf)
    pk = p_sum[1:] / nmodes.astype(conf.float_dtype)
    pk = pk * (conf.cell_size ** 3 / conf.mesh_size)
    return k, pk.astype(conf.float_dtype), nmodes


def _finalize_cross_shell_averages(p_a_sum, p_b_sum, p_cross_sum, conf, eps):
    """Convert shell sums into auto spectra, cross spectrum, and correlation."""
    k, pk_a, nmodes = _finalize_shell_averages(p_a_sum, conf)
    _, pk_b, _ = _finalize_shell_averages(p_b_sum, conf)
    _, pk_cross, _ = _finalize_shell_averages(p_cross_sum, conf)
    eps = jnp.asarray(eps, dtype=conf.float_dtype)
    denom = jnp.sqrt(jnp.maximum(pk_a * pk_b, eps))
    r = pk_cross / denom
    return k, r.astype(conf.float_dtype), pk_cross.astype(conf.float_dtype), pk_a, pk_b, nmodes


def _spectral_delta(delta, conf):
    """FFT an overdensity mesh into the solver's spectral layout."""
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
def delta_to_cross_correlation(delta_a, delta_b, conf, mas: str | None = "CIC", eps: float = 1e-30):
    """Compute the isotropic cross-correlation coefficient of two fields.

    Parameters
    ----------
    delta_a, delta_b : jax.Array
        Overdensity fields on the PM mesh.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme used to create the fields. This controls the
        same Fourier-space deconvolution used by :func:`delta_to_pk`. Use
        ``None`` to disable deconvolution.
    eps : float, optional
        Positive floor for ``P_aa(k) P_bb(k)`` in the correlation denominator.

    Returns
    -------
    k : jax.Array
        Shell-averaged physical wavenumbers.
    r : jax.Array
        Cross-correlation coefficient, ``P_ab / sqrt(P_aa P_bb)``.
    pk_cross : jax.Array
        Shell-averaged real cross spectrum.
    pk_a, pk_b : jax.Array
        Shell-averaged auto spectra for ``delta_a`` and ``delta_b``.
    nmodes : jax.Array
        Number of contributing Fourier modes per shell.
    """
    mas = _normalize_mas(mas)
    spectral_a = _spectral_delta(jnp.asarray(delta_a, dtype=conf.float_dtype), conf)
    spectral_b = _spectral_delta(jnp.asarray(delta_b, dtype=conf.float_dtype), conf)
    num_shells = _max_shell_index(conf) + 1
    p_a_sum, p_b_sum, p_cross_sum = _shell_reduce_cross_transposed(
        spectral_a,
        spectral_b,
        conf,
        _MAS_POWER[mas],
        num_shells,
    )
    return _finalize_cross_shell_averages(p_a_sum, p_b_sum, p_cross_sum, conf, eps)


@partial(jax.jit, static_argnames=("conf", "mas"))
def density_to_pk(density, conf, mas: str | None = "CIC"):
    """Compute a differentiable power spectrum from a density field.

    Parameters
    ----------
    density : jax.Array
        Density field on the PM mesh.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme used to create the field. This controls optional
        Fourier-space deconvolution.

    Returns
    -------
    k : jax.Array
        Shell-averaged physical wavenumbers.
    pk : jax.Array
        Shell-averaged isotropic monopole.
    nmodes : jax.Array
        Number of contributing Fourier modes per shell.
    """
    density = jnp.asarray(density, dtype=conf.float_dtype)
    mean_density = jnp.mean(density, dtype=conf.float_dtype)
    delta = density / mean_density - 1
    return delta_to_pk(delta, conf, mas=mas)


@partial(jax.jit, static_argnames=("conf", "mas"))
def density_to_cross_correlation(density_a, density_b, conf, mas: str | None = "CIC", eps: float = 1e-30):
    """Compute the cross-correlation coefficient from two density fields.

    Parameters
    ----------
    density_a, density_b : jax.Array
        Density fields on the PM mesh.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme used to create the fields.
    eps : float, optional
        Positive floor for the auto-spectrum product in the denominator.

    Returns
    -------
    k : jax.Array
        Shell-averaged physical wavenumbers.
    r : jax.Array
        Cross-correlation coefficient.
    pk_cross : jax.Array
        Shell-averaged real cross spectrum.
    pk_a, pk_b : jax.Array
        Shell-averaged auto spectra for the two fields.
    nmodes : jax.Array
        Number of contributing Fourier modes per shell.
    """
    density_a = jnp.asarray(density_a, dtype=conf.float_dtype)
    density_b = jnp.asarray(density_b, dtype=conf.float_dtype)
    delta_a = density_a / jnp.mean(density_a, dtype=conf.float_dtype) - 1
    delta_b = density_b / jnp.mean(density_b, dtype=conf.float_dtype) - 1
    return delta_to_cross_correlation(delta_a, delta_b, conf, mas=mas, eps=eps)


@partial(jax.jit, static_argnames=("conf", "mas"))
def particles_to_pk(ptcl, conf, mas: str | None = "CIC"):
    """Scatter particles to the PM mesh and return the differentiable ``P(k)``.

    Parameters
    ----------
    ptcl : Particles
        Particle state to scatter.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme matching the scatter rule.

    Returns
    -------
    k : jax.Array
        Shell-averaged physical wavenumbers.
    pk : jax.Array
        Shell-averaged isotropic monopole.
    nmodes : jax.Array
        Number of contributing Fourier modes per shell.
    """
    density = scatter(ptcl, conf)
    return density_to_pk(density, conf, mas=mas)


@partial(jax.jit, static_argnames=("conf", "mas"))
def particles_to_cross_correlation(ptcl_a, ptcl_b, conf, mas: str | None = "CIC", eps: float = 1e-30):
    """Scatter two particle sets and return their cross-correlation coefficient.

    Parameters
    ----------
    ptcl_a, ptcl_b : Particles
        Particle states to compare.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme matching the scatter rule.
    eps : float, optional
        Positive floor for the auto-spectrum product in the denominator.

    Returns
    -------
    k : jax.Array
        Shell-averaged physical wavenumbers.
    r : jax.Array
        Cross-correlation coefficient.
    pk_cross : jax.Array
        Shell-averaged real cross spectrum.
    pk_a, pk_b : jax.Array
        Shell-averaged auto spectra for the two particle sets.
    nmodes : jax.Array
        Number of contributing Fourier modes per shell.
    """
    density_a = scatter(ptcl_a, conf)
    density_b = scatter(ptcl_b, conf)
    return density_to_cross_correlation(density_a, density_b, conf, mas=mas, eps=eps)


def cross_correlation(delta_a, delta_b, conf, mas: str | None = "CIC", eps: float = 1e-30):
    """Alias for :func:`delta_to_cross_correlation` on overdensity fields.

    Parameters
    ----------
    delta_a, delta_b : jax.Array
        Overdensity fields on the PM mesh.
    conf : Configuration
        Active PM++ configuration.
    mas : str or None, optional
        Mass-assignment scheme used to create the fields.
    eps : float, optional
        Positive floor for the auto-spectrum product in the denominator.

    Returns
    -------
    tuple
        Same return values as :func:`delta_to_cross_correlation`.
    """
    return delta_to_cross_correlation(delta_a, delta_b, conf, mas=mas, eps=eps)


# Backward-compatible aliases for the earlier public naming.
delta_to_quijote_pk = delta_to_pk
density_to_quijote_pk = density_to_pk
particles_to_quijote_pk = particles_to_pk
delta_to_quijote_cross_correlation = delta_to_cross_correlation
density_to_quijote_cross_correlation = density_to_cross_correlation
particles_to_quijote_cross_correlation = particles_to_cross_correlation
