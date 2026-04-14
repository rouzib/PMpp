"""Analytic PM assignment-window compensation corrections."""

from dataclasses import field
from functools import partial

import jax.numpy as jnp

from ..utils import pytree_dataclass


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
    """Analytic PM assignment-window compensation in Fourier space.

    For CIC scatter plus CIC gather, the linear particle-force response is
    suppressed by two CIC windows. This correction multiplies the potential by a
    bounded inverse window expressed only in dimensionless mesh units.
    """

    assignment_order: int = 2
    windows: int = 2
    alpha: float = 0.5
    max_gain: float = 4.0
    taper_start: float = 0.75
    taper_stop: float = 1.0
    interlacing: bool = False
    green_kernel: str = "continuum"
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))


def init_pm_window_compensation_correction(dtype=jnp.float32, **kwargs):
    """Initialize an analytic CIC/PM assignment-window compensation correction."""
    return PMWindowCompensationCorrection(
        assignment_order=kwargs.get("assignment_order", 2),
        windows=kwargs.get("windows", 2),
        alpha=kwargs.get("alpha", kwargs.get("window_alpha", 0.5)),
        max_gain=kwargs.get("max_gain", kwargs.get("window_max_gain", 4.0)),
        taper_start=kwargs.get("taper_start", kwargs.get("window_taper_start", 0.75)),
        taper_stop=kwargs.get("taper_stop", kwargs.get("window_taper_stop", 1.0)),
        interlacing=kwargs.get("interlacing", False),
        green_kernel=kwargs.get("green_kernel", "continuum"),
        dtype=dtype,
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
    exponent = correction.assignment_order * correction.windows * correction.alpha
    gain = base_window ** (-jnp.asarray(exponent, dtype=correction.dtype))
    if correction.max_gain is not None and correction.max_gain > 0:
        gain = jnp.minimum(gain, jnp.asarray(correction.max_gain, dtype=correction.dtype))

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
