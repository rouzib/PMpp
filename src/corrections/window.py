"""Analytic PM assignment-window compensation corrections."""

from dataclasses import field
from functools import partial

import jax
import jax.numpy as jnp

from ..utils import pytree_dataclass


def _logit(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x) - jnp.log1p(-x)


def _bounded_sigmoid(raw, lo, hi, dtype):
    lo = jnp.asarray(lo, dtype=dtype)
    hi = jnp.asarray(hi, dtype=dtype)
    return lo + (hi - lo) * jax.nn.sigmoid(jnp.asarray(raw, dtype=dtype))


@partial(
    pytree_dataclass,
    aux_fields=(
        "assignment_order",
        "windows",
        "alpha",
        "max_gain",
        "taper_start",
        "taper_stop",
        "interlacing",
        "green_kernel",
        "dtype",
    ),
    frozen=True,
    eq=False,
)
class PMWindowCompensationCorrection:
    """Analytic Fourier-space compensation for PM assignment smoothing.

    Particle-mesh gravity is softened by the mass-assignment windows used to
    scatter particles to the mesh and gather forces back to particles.  For CIC
    scatter plus CIC gather, the linear force response contains approximately
    two CIC windows.  This correction partially inverts that response by
    multiplying the Fourier-space potential by a bounded transfer function,

    ``gain(k) = W(k) ** (-(assignment_order * windows * alpha))``,

    where ``W(k)`` is the absolute product of one-dimensional sinc assignment
    windows on the PM force mesh.  The gain is capped and tapered near the
    particle Nyquist scale to avoid amplifying poorly resolved modes.

    Parameters
    ----------
    assignment_order : int, optional
        Order of the assignment window being compensated.  Use ``1`` for NGP,
        ``2`` for CIC, ``3`` for TSC, and ``4`` for PCS.  PM++ currently uses
        CIC scatter/gather in the force path, so the default is ``2``.
    windows : int, optional
        Number of assignment windows to compensate.  A particle-mesh force built
        from CIC scatter and CIC gather has two such windows, so the default is
        ``2``.  Lower values compensate less aggressively.
    alpha : float, optional
        Fractional compensation strength.  ``0`` disables the window boost,
        ``1`` applies the full inverse of the requested assignment windows, and
        values between them provide a stable partial inverse.  Values that are
        too large can add excess small-scale power.
    max_gain : float or None, optional
        Upper bound on the multiplicative potential boost.  A positive value
        clips the transfer function to ``max_gain``.  Set to ``None`` or a
        non-positive value to disable clipping, which is usually less stable
        near the mesh scale.
    taper_start : float, optional
        Start of the high-k taper as a fraction of the particle Nyquist
        wavenumber.  The taper coordinate is
        ``max(abs(kx), abs(ky), abs(kz)) / k_particle_nyquist``.  Below this
        value, the capped gain is applied unchanged.
    taper_stop : float, optional
        End of the high-k taper as a fraction of the particle Nyquist
        wavenumber.  Between ``taper_start`` and ``taper_stop`` the boost is
        smoothly reduced back to unity.  If ``taper_stop <= taper_start``, no
        taper is applied.
    interlacing : bool, optional
        Whether gravity should use interlaced density assignment: one regular
        scatter and one half-cell shifted scatter are averaged in Fourier space
        after phase correction.  This can reduce odd aliasing terms, but costs
        an additional scatter/FFT path and was not the best setting in the
        Quijote 256^3 window-correction sweep.
    green_kernel : {"continuum", "discrete_laplacian"}, optional
        Poisson Green's function used before the window transfer.  ``"continuum"``
        uses ``-1 / k^2``.  ``"discrete_laplacian"`` uses the lattice Laplacian
        symbol with ``2 sin(k_i dx / 2) / dx`` on each axis, which usually gives
        a stronger and more accurate near-mesh response for this correction.
    dtype : dtype, optional
        Floating-point dtype used for evaluating the transfer field.

    Notes
    -----
    This is an analytic force/power-spectrum correction, not a learned
    position correction.  It can substantially improve ``P(k)`` while moving
    individual particles farther from an N-body reference.
    """

    assignment_order: int = 2
    windows: int = 2
    alpha: float = 0.48
    max_gain: float = 4.0
    taper_start: float = 0.75
    taper_stop: float = 1.0
    interlacing: bool = False
    green_kernel: str = "discrete_laplacian"
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))


@partial(
    pytree_dataclass,
    aux_fields=(
        "assignment_order",
        "windows",
        "alpha_min",
        "alpha_max",
        "max_gain_min",
        "max_gain_max",
        "taper_start",
        "taper_stop",
        "interlacing",
        "green_kernel",
        "dtype",
    ),
    frozen=True,
    eq=False,
)
class TrainablePMWindowCompensationCorrection:
    """Trainable bounded PM assignment-window compensation.

    The optimizer sees unconstrained scalar leaves. Evaluation maps them to
    physical values with bounded sigmoids so training cannot produce a negative
    compensation strength or an unbounded high-k gain.
    """

    raw_alpha: jnp.ndarray
    raw_max_gain: jnp.ndarray
    assignment_order: int = 2
    windows: int = 2
    alpha_min: float = 0.0
    alpha_max: float = 0.8
    max_gain_min: float = 1.0
    max_gain_max: float = 8.0
    taper_start: float = 0.75
    taper_stop: float = 1.0
    interlacing: bool = False
    green_kernel: str = "discrete_laplacian"
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        dtype = jnp.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "raw_alpha", jnp.asarray(self.raw_alpha, dtype=dtype))
        object.__setattr__(self, "raw_max_gain", jnp.asarray(self.raw_max_gain, dtype=dtype))


def init_pm_window_compensation_correction(dtype=jnp.float32, **kwargs):
    """Initialize an analytic CIC/PM assignment-window compensation correction."""
    return PMWindowCompensationCorrection(
        assignment_order=kwargs.get("assignment_order", 2),
        windows=kwargs.get("windows", 2),
        alpha=kwargs.get("alpha", kwargs.get("window_alpha", 0.48)),
        max_gain=kwargs.get("max_gain", kwargs.get("window_max_gain", 4.0)),
        taper_start=kwargs.get("taper_start", kwargs.get("window_taper_start", 0.75)),
        taper_stop=kwargs.get("taper_stop", kwargs.get("window_taper_stop", 1.0)),
        interlacing=kwargs.get("interlacing", False),
        green_kernel=kwargs.get("green_kernel", "discrete_laplacian"),
        dtype=dtype,
    )


def init_trainable_pm_window_compensation_correction(dtype=jnp.float32, **kwargs):
    """Initialize a trainable bounded PM-window compensation correction."""
    dtype = jnp.dtype(dtype)
    alpha_min = kwargs.get("window_alpha_min", kwargs.get("alpha_min", 0.0))
    alpha_max = kwargs.get("window_alpha_max", kwargs.get("alpha_max", 0.8))
    max_gain_min = kwargs.get("window_max_gain_min", kwargs.get("max_gain_min", 1.0))
    max_gain_max = kwargs.get("window_max_gain_max", kwargs.get("max_gain_max", 8.0))
    alpha = kwargs.get("alpha", kwargs.get("window_alpha", 0.48))
    max_gain = kwargs.get("max_gain", kwargs.get("window_max_gain", 4.0))
    alpha_unit = (float(alpha) - float(alpha_min)) / (float(alpha_max) - float(alpha_min))
    max_gain_unit = (float(max_gain) - float(max_gain_min)) / (float(max_gain_max) - float(max_gain_min))
    return TrainablePMWindowCompensationCorrection(
        raw_alpha=jnp.asarray(_logit(alpha_unit), dtype=dtype),
        raw_max_gain=jnp.asarray(_logit(max_gain_unit), dtype=dtype),
        assignment_order=kwargs.get("assignment_order", 2),
        windows=kwargs.get("windows", 2),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        max_gain_min=max_gain_min,
        max_gain_max=max_gain_max,
        taper_start=kwargs.get("taper_start", kwargs.get("window_taper_start", 0.75)),
        taper_stop=kwargs.get("taper_stop", kwargs.get("window_taper_stop", 1.0)),
        interlacing=kwargs.get("interlacing", False),
        green_kernel=kwargs.get("green_kernel", "discrete_laplacian"),
        dtype=dtype,
    )


def pm_window_parameters(correction):
    """Return physical ``(alpha, max_gain)`` for fixed or trainable windows."""
    if isinstance(correction, TrainablePMWindowCompensationCorrection):
        alpha = _bounded_sigmoid(correction.raw_alpha, correction.alpha_min, correction.alpha_max, correction.dtype)
        max_gain = _bounded_sigmoid(
            correction.raw_max_gain,
            correction.max_gain_min,
            correction.max_gain_max,
            correction.dtype,
        )
        return alpha, max_gain
    return (
        jnp.asarray(correction.alpha, dtype=correction.dtype),
        None if correction.max_gain is None else jnp.asarray(correction.max_gain, dtype=correction.dtype),
    )


def evaluate_pm_window_compensation(correction, conf):
    """Evaluate the bounded inverse assignment-window transfer on the k grid."""
    kx, ky, kz = [jnp.squeeze(a).astype(correction.dtype) for a in conf.kvec]
    cell_size = jnp.asarray(conf.cell_size, dtype=correction.dtype)
    particle_nyquist = jnp.asarray(jnp.pi / conf.ptcl_spacing, dtype=correction.dtype)
    eps = jnp.asarray(1e-6, dtype=correction.dtype)

    sx = jnp.sinc(kx[:, None, None] * cell_size / (2 * jnp.pi))
    sy = jnp.sinc(ky[None, :, None] * cell_size / (2 * jnp.pi))
    sz = jnp.sinc(kz[None, None, :] * cell_size / (2 * jnp.pi))
    base_window = jnp.maximum(jnp.abs(sx * sy * sz), eps)
    alpha, max_gain = pm_window_parameters(correction)
    exponent = correction.assignment_order * correction.windows * alpha
    gain = base_window ** (-jnp.asarray(exponent, dtype=correction.dtype))
    if max_gain is not None:
        gain = jnp.minimum(gain, max_gain)

    if correction.taper_stop > correction.taper_start:
        qx = jnp.abs(kx[:, None, None]) / particle_nyquist
        qy = jnp.abs(ky[None, :, None]) / particle_nyquist
        qz = jnp.abs(kz[None, None, :]) / particle_nyquist
        q = jnp.maximum(jnp.maximum(qx, qy), qz)
        start = jnp.asarray(correction.taper_start, dtype=correction.dtype)
        stop = jnp.asarray(correction.taper_stop, dtype=correction.dtype)
        t = jnp.clip((q - start) / (stop - start), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        taper = 1.0 - smooth
        gain = 1.0 + (gain - 1.0) * taper

    return jnp.where(base_window == 0, jnp.asarray(1.0, dtype=correction.dtype), gain)
