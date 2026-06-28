"""Radial Fourier-space potential transfer corrections."""

from dataclasses import field
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from ..utils import is_float0_array, pytree_dataclass
from .common import (
    HaikuModuleBase,
    correction_cosmo_features,
    default_cosmo_features,
    normalized_k_magnitude_transposed,
    require_haiku,
    hk,
)


def _deBoorVectorized(x, t, c, p):
    """Evaluate a B-spline with the vectorized de Boor algorithm."""
    k = jnp.digitize(x, t) - 1

    d = [c[j + k - p] for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter(HaikuModuleBase):
    """Rotationally invariant filter parameterized by a small Haiku network."""

    def __init__(self, n_knots=8, latent_size=16, output_init_scale=None, name=None):
        require_haiku("radial potential corrections")
        super().__init__(name=name)
        self.n_knots = n_knots
        self.latent_size = latent_size
        self.output_init_scale = output_init_scale

    def __call__(self, x, a, cosmo):
        del cosmo

        net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
        net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        w = hk.Linear(self.latent_size)(net)
        k = hk.Linear(self.latent_size)(net)

        output_init_kwargs = {}
        if self.output_init_scale is not None:
            scale = float(self.output_init_scale)
            if scale == 0.0:
                output_init_kwargs = {
                    "w_init": hk.initializers.Constant(0.0),
                    "b_init": hk.initializers.Constant(0.0),
                }
            else:
                output_init_kwargs = {
                    "w_init": hk.initializers.RandomNormal(stddev=scale),
                    "b_init": hk.initializers.Constant(0.0),
                }

        w = hk.Linear(self.n_knots + 1, **output_init_kwargs)(w)
        k = hk.Linear(self.n_knots - 1)(k)

        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])
        w = jnp.concatenate([jnp.zeros((1,)), w])
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3.0), 0.0, 1.0 - 1e-4), ak, w, 3)


@lru_cache(maxsize=None)
def radial_transform(n_knots, latent_size, output_init_scale):
    """Return the cached Haiku transform for the radial spline filter."""
    require_haiku("radial potential corrections")
    return hk.without_apply_rng(
        hk.transform(
            lambda x, a, c: NeuralSplineFourierFilter(
                n_knots=n_knots,
                latent_size=latent_size,
                output_init_scale=output_init_scale,
            )(x, a, c)
        )
    )


@partial(
    pytree_dataclass,
    aux_fields=("n_knots", "latent_size", "output_init_scale", "allow_missing_sigma8", "sigma8_value", "dtype"),
    frozen=True,
    eq=False,
)
class RadialPotentialCorrection:
    """Trainable rotationally invariant multiplicative potential correction."""

    params: dict
    n_knots: int = 8
    latent_size: int = 16
    output_init_scale: float | None = None
    allow_missing_sigma8: bool = False
    sigma8_value: float = 0.8
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        dtype = jnp.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self,
            "params",
            tree_map(lambda x: x if x is None or is_float0_array(x) else jnp.asarray(x, dtype=dtype), self.params),
        )


def init_radial_potential_correction(
    key,
    latent_size=16,
    n_knots=16,
    output_init_scale=None,
    allow_missing_sigma8=False,
    sigma8_value=0.8,
    dtype=jnp.float32,
    conf=None,
    **unused_kwargs,
):
    """Initialize a radial Fourier-space potential correction.

    Parameters
    ----------
    key : jax.Array
        PRNG key used to initialize Haiku parameters.
    latent_size : int, optional
        Hidden width of the conditioning network.
    n_knots : int, optional
        Number of spline knots controlling the radial transfer.
    output_init_scale : float or None, optional
        Optional output-layer initialization scale.
    allow_missing_sigma8 : bool, optional
        Whether later evaluation may fall back to a stored ``sigma8_value``.
    sigma8_value : float, optional
        Stored fallback conditioning value.
    dtype : DTypeLike, optional
        Parameter dtype.
    conf : Configuration, optional
        Representative configuration used to build the normalized ``|k|`` grid.

    Returns
    -------
    RadialPotentialCorrection
        Initialized trainable radial correction.
    """
    del unused_kwargs
    dtype = jnp.dtype(dtype)
    transform = radial_transform(n_knots, latent_size, output_init_scale)
    if conf is None:
        x = jnp.linspace(0.0, jnp.sqrt(3.0), 16, dtype=dtype)
    else:
        x = normalized_k_magnitude_transposed(conf).astype(dtype)
    params = transform.init(
        key,
        x,
        jnp.ones((1,), dtype=dtype),
        default_cosmo_features(dtype),
    )
    return RadialPotentialCorrection(
        params=params,
        n_knots=n_knots,
        latent_size=latent_size,
        output_init_scale=output_init_scale,
        allow_missing_sigma8=allow_missing_sigma8,
        sigma8_value=float(sigma8_value),
        dtype=dtype,
    )


def sample_radial_potential_transfer(correction, radius_fraction, a, cosmo, conf):
    """Evaluate the radial transfer on user-supplied normalized radii."""
    if correction is None:
        return jnp.ones_like(radius_fraction, dtype=conf.float_dtype)
    radius_fraction = jnp.asarray(radius_fraction, dtype=correction.dtype)
    x = radius_fraction * jnp.asarray(jnp.sqrt(3.0), dtype=correction.dtype)
    transform = radial_transform(correction.n_knots, correction.latent_size, correction.output_init_scale)
    residual = transform.apply(
        correction.params,
        x,
        jnp.asarray(a, dtype=correction.dtype),
        correction_cosmo_features(correction, cosmo, correction.dtype),
    )
    transfer = 1.0 + residual
    return jnp.where(radius_fraction == 0, jnp.asarray(1.0, dtype=transfer.dtype), transfer)


def evaluate_radial_potential_transfer(correction, a, cosmo, conf):
    """Evaluate the radial transfer on the full solver k grid."""
    if correction is None:
        return jnp.ones(tuple(conf.mesh_shape[:-1]) + (conf.mesh_shape[-1] // 2 + 1,), dtype=conf.float_dtype)
    k_norm = normalized_k_magnitude_transposed(conf)
    transform = radial_transform(correction.n_knots, correction.latent_size, correction.output_init_scale)
    residual = transform.apply(
        correction.params,
        k_norm.astype(correction.dtype),
        jnp.asarray(a, dtype=correction.dtype),
        correction_cosmo_features(correction, cosmo, correction.dtype),
    )
    return 1.0 + residual
