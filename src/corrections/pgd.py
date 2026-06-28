"""FastPM-style PGD residual potential corrections."""

from dataclasses import field
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from ..utils import AXIS_NAME, pytree_dataclass


@partial(
    pytree_dataclass,
    aux_fields=("alpha0", "A", "B", "kl", "ks", "dtype"),
    frozen=True,
    eq=False,
)
class PGDPotentialCorrection:
    """Band-limited FastPM-style PGD correction to the PM potential.

    FastPM constructs an additive residual potential proportional to

    ``alpha(a) * exp(-kl**2 / k**2 - k**4 / ks**4) * delta_k / k**2``.

    PM++'s base potential is ``-delta_k / k**2``. Reusing the same Green
    function, gradient, FFT, and gather operators makes this additive branch
    equivalent to multiplying the base potential by

    ``1 - alpha(a) * exp(-kl**2 / k**2 - k**4 / ks**4)``.

    The sign is therefore controlled by the sign of ``alpha0``.
    """

    alpha0: float = 0.1
    A: float = 0.0
    B: float = 0.0
    kl: float = 0.1
    ks: float = 1.0
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))


@partial(
    pytree_dataclass,
    aux_fields=("A", "B", "alpha0_scale", "kl_min", "kl_max", "ks_min", "ks_max", "dtype"),
    frozen=True,
    eq=False,
)
class TrainablePGDPotentialCorrection:
    """Trainable bounded FastPM-style PGD correction."""

    raw_alpha0: jnp.ndarray
    raw_kl: jnp.ndarray
    raw_ks: jnp.ndarray
    A: float = 0.0
    B: float = 0.0
    alpha0_scale: float = 0.2
    kl_min: float = 0.02
    kl_max: float = 0.2
    ks_min: float = 0.3
    ks_max: float = 1.2
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        dtype = jnp.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "raw_alpha0", jnp.asarray(self.raw_alpha0, dtype=dtype))
        object.__setattr__(self, "raw_kl", jnp.asarray(self.raw_kl, dtype=dtype))
        object.__setattr__(self, "raw_ks", jnp.asarray(self.raw_ks, dtype=dtype))


def _logit(x):
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x) - jnp.log1p(-x)


def _atanh_scaled(x, scale):
    y = jnp.clip(jnp.asarray(x) / jnp.asarray(scale), -1.0 + 1e-6, 1.0 - 1e-6)
    return jnp.arctanh(y)


def _bounded_sigmoid(raw, lo, hi, dtype):
    lo = jnp.asarray(lo, dtype=dtype)
    hi = jnp.asarray(hi, dtype=dtype)
    return lo + (hi - lo) * jax.nn.sigmoid(jnp.asarray(raw, dtype=dtype))


def init_pgd_potential_correction(dtype=jnp.float32, **kwargs):
    """Initialize a fixed FastPM-style PGD potential correction.

    Keyword arguments accept both bare names such as ``alpha0`` and prefixed
    names such as ``pgd_alpha0`` for compatibility with experiment scripts.
    """
    return PGDPotentialCorrection(
        alpha0=kwargs.get("alpha0", kwargs.get("pgd_alpha0", 0.1)),
        A=kwargs.get("A", kwargs.get("pgd_A", 0.0)),
        B=kwargs.get("B", kwargs.get("pgd_B", 0.0)),
        kl=kwargs.get("kl", kwargs.get("pgd_kl", 0.1)),
        ks=kwargs.get("ks", kwargs.get("pgd_ks", 1.0)),
        dtype=dtype,
    )


def init_trainable_pgd_potential_correction(dtype=jnp.float32, **kwargs):
    """Initialize bounded trainable FastPM-style PGD parameters.

    Keyword arguments accept both bare names and ``pgd_*`` aliases. Physical
    parameters are converted into bounded unconstrained variables suitable for
    optimization.
    """
    dtype = jnp.dtype(dtype)
    alpha0_scale = kwargs.get("pgd_alpha0_scale", kwargs.get("alpha0_scale", 0.2))
    kl_min = kwargs.get("pgd_kl_min", kwargs.get("kl_min", 0.02))
    kl_max = kwargs.get("pgd_kl_max", kwargs.get("kl_max", 0.2))
    ks_min = kwargs.get("pgd_ks_min", kwargs.get("ks_min", 0.3))
    ks_max = kwargs.get("pgd_ks_max", kwargs.get("ks_max", 1.2))
    alpha0 = kwargs.get("alpha0", kwargs.get("pgd_alpha0", 0.0))
    kl = kwargs.get("kl", kwargs.get("pgd_kl", 0.08))
    ks = kwargs.get("ks", kwargs.get("pgd_ks", 0.4))
    kl_unit = (float(kl) - float(kl_min)) / (float(kl_max) - float(kl_min))
    ks_unit = (float(ks) - float(ks_min)) / (float(ks_max) - float(ks_min))
    return TrainablePGDPotentialCorrection(
        raw_alpha0=jnp.asarray(_atanh_scaled(alpha0, alpha0_scale), dtype=dtype),
        raw_kl=jnp.asarray(_logit(kl_unit), dtype=dtype),
        raw_ks=jnp.asarray(_logit(ks_unit), dtype=dtype),
        A=kwargs.get("A", kwargs.get("pgd_A", 0.0)),
        B=kwargs.get("B", kwargs.get("pgd_B", 0.0)),
        alpha0_scale=alpha0_scale,
        kl_min=kl_min,
        kl_max=kl_max,
        ks_min=ks_min,
        ks_max=ks_max,
        dtype=dtype,
    )


def pgd_parameters(correction):
    """Return physical ``(alpha0, A, B, kl, ks)`` for fixed or trainable PGD."""
    if isinstance(correction, TrainablePGDPotentialCorrection):
        alpha0 = jnp.asarray(correction.alpha0_scale, dtype=correction.dtype) * jnp.tanh(correction.raw_alpha0)
        kl = _bounded_sigmoid(correction.raw_kl, correction.kl_min, correction.kl_max, correction.dtype)
        ks = _bounded_sigmoid(correction.raw_ks, correction.ks_min, correction.ks_max, correction.dtype)
        return alpha0, correction.A, correction.B, kl, ks
    return correction.alpha0, correction.A, correction.B, correction.kl, correction.ks


def pgd_alpha(a, alpha0, A, B, dtype=jnp.float32):
    """Return the FastPM PGD time-dependent amplitude."""
    a = jnp.asarray(a, dtype=dtype)
    return jnp.asarray(alpha0, dtype=dtype) * (jnp.asarray(10.0, dtype=dtype) ** (jnp.asarray(A, dtype=dtype) * a**2 - jnp.asarray(B, dtype=dtype) * a))


def _k_squared_transposed(kvec, conf):
    """Return ``k^2`` on the transposed spectral layout without importing gravity."""
    kx, ky, kz = [jnp.squeeze(a) for a in kvec]
    if conf.compute_mesh is None:
        return (kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(conf.float_dtype)

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
        local_shard = (
            kx_replicated[:, None, None] ** 2
            + ky_sharded[None, :, None] ** 2
            + kz_replicated[None, None, :] ** 2
        )
        return local_shard.astype(conf.float_dtype)

    return create_k_magnitude_transposed(kx, ky, kz)


def evaluate_pgd_bandpass(correction, a, conf):
    """Evaluate the PGD intermediate-k band-pass factor."""
    k2 = _k_squared_transposed(conf.kvec, conf).astype(correction.dtype)
    _, _, _, kl, ks = pgd_parameters(correction)
    kl2 = jnp.asarray(kl, dtype=correction.dtype) ** 2
    ks4 = jnp.asarray(ks, dtype=correction.dtype) ** 4
    safe_k2 = jnp.where(k2 > 0, k2, jnp.ones_like(k2))
    band = jnp.exp(-kl2 / safe_k2 - (safe_k2**2) / ks4)
    return jnp.where(k2 > 0, band, jnp.asarray(0.0, dtype=correction.dtype))


def evaluate_pgd_potential_transfer(correction, a, conf):
    """Evaluate the multiplicative potential transfer equivalent to PGD."""
    alpha0, A, B, _, _ = pgd_parameters(correction)
    alpha = pgd_alpha(a, alpha0, A, B, dtype=correction.dtype)
    band = evaluate_pgd_bandpass(correction, a, conf)
    return 1.0 - alpha * band
