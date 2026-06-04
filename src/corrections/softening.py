"""High-k spectral softening corrections."""

from dataclasses import field
from functools import partial

import jax.numpy as jnp

from ..utils import pytree_dataclass


@partial(
    pytree_dataclass,
    aux_fields=("strength", "start", "stop", "mode", "dtype"),
    frozen=True,
    eq=False,
)
class HighKSofteningCorrection:
    """Smoothly damp force response near the particle Nyquist scale.

    This is a small, isolated version of the high-k filtering commonly used in
    FastPM-style PM force stacks. It is intentionally just a spectral transfer:
    it does not change the base Green kernel, assignment, gradient, or gather.
    """

    strength: float = 0.0
    start: float = 0.7
    stop: float = 1.0
    mode: str = "smoothstep"
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))


def init_high_k_softening_correction(dtype=jnp.float32, **kwargs):
    """Initialize a high-k softening correction."""
    return HighKSofteningCorrection(
        strength=kwargs.get("strength", kwargs.get("softening_strength", 0.0)),
        start=kwargs.get("start", kwargs.get("softening_start", 0.7)),
        stop=kwargs.get("stop", kwargs.get("softening_stop", 1.0)),
        mode=kwargs.get("mode", kwargs.get("softening_mode", "smoothstep")),
        dtype=dtype,
    )


def evaluate_high_k_softening(correction, conf):
    """Evaluate the high-k damping transfer on the PM spectral grid."""
    kx, ky, kz = [jnp.squeeze(a).astype(correction.dtype) for a in conf.kvec]
    particle_nyquist = jnp.asarray(jnp.pi / conf.ptcl_spacing, dtype=correction.dtype)
    qx = jnp.abs(kx[:, None, None]) / particle_nyquist
    qy = jnp.abs(ky[None, :, None]) / particle_nyquist
    qz = jnp.abs(kz[None, None, :]) / particle_nyquist
    q = jnp.maximum(jnp.maximum(qx, qy), qz)

    start = jnp.asarray(correction.start, dtype=correction.dtype)
    stop = jnp.asarray(correction.stop, dtype=correction.dtype)
    t = jnp.clip((q - start) / jnp.maximum(stop - start, jnp.asarray(1e-6, dtype=correction.dtype)), 0.0, 1.0)
    if correction.mode == "smoothstep":
        taper = t * t * (3.0 - 2.0 * t)
    elif correction.mode == "linear":
        taper = t
    else:
        raise ValueError(f"Unsupported high-k softening mode {correction.mode!r}.")
    transfer = 1.0 - jnp.asarray(correction.strength, dtype=correction.dtype) * taper
    return jnp.maximum(transfer, jnp.asarray(0.0, dtype=correction.dtype))
