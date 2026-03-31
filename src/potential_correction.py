from dataclasses import field
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax.typing import ArrayLike

try:
    import haiku as hk
except ImportError:
    hk = None

try:
    import optax
except ImportError:
    optax = None

from .mesh_halo import exchange_owned_mesh_halo_edges, extend_owned_mesh_from_halo_edges
from .utils import is_float0_array, pytree_dataclass


def _require_haiku(feature_name):
    if hk is None:
        raise ImportError(
            f"haiku is required for {feature_name}. Install dm-haiku to enable this correction."
        )


def _require_optax(feature_name):
    if optax is None:
        raise ImportError(
            f"optax is required for {feature_name}. Install optax to enable this optimizer utility."
        )


_HaikuModuleBase = hk.Module if hk is not None else object


def _deBoorVectorized(x, t, c, p):
    """Evaluate a B-spline with the vectorized de Boor algorithm."""
    k = jnp.digitize(x, t) - 1

    d = [c[j + k - p] for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter(_HaikuModuleBase):
    """Rotationally invariant filter parameterized by a small Haiku network.

        See Denise's paper
    """

    def __init__(self, n_knots=8, latent_size=16, name=None):
        _require_haiku("radial potential corrections")
        super().__init__(name=name)
        self.n_knots = n_knots
        self.latent_size = latent_size

    def __call__(self, x, a, cosmo):
        del cosmo

        net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
        net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        w = hk.Linear(self.latent_size)(net)
        k = hk.Linear(self.latent_size)(net)

        w = hk.Linear(self.n_knots + 1)(w)
        k = hk.Linear(self.n_knots - 1)(k)

        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])
        w = jnp.concatenate([jnp.zeros((1,)), w])
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3.0), 0.0, 1.0 - 1e-4), ak, w, 3)


class MeshResidualSourceCorrection(_HaikuModuleBase):
    """Small periodic 3D CNN that predicts a residual source field."""

    def __init__(self, channels=8, depth=4, max_residual=0.1, name=None):
        _require_haiku("mesh CNN potential corrections")
        super().__init__(name=name)
        self.channels = channels
        self.depth = depth
        self.max_residual = max_residual

    def __call__(self, source, a, cosmo, conf):
        dtype = source.dtype
        source_scale = jnp.sqrt(jnp.mean(source ** 2) + jnp.asarray(1e-6, dtype=dtype))
        source_norm = source / source_scale
        features = _mesh_conditioning_channels(source_norm, a, cosmo, dtype)
        x = jnp.concatenate(features, axis=-1)

        x = jax.nn.gelu(_periodic_conv3d(x, self.channels, conf, kernel_shape=3, name="stem"))
        for block_id in range(max(self.depth - 1, 0)):
            residual = x
            x = jax.nn.gelu(
                _periodic_conv3d(x, self.channels, conf, kernel_shape=3, name=f"block_{block_id}_conv1")
            )
            x = _periodic_conv3d(x, self.channels, conf, kernel_shape=3, name=f"block_{block_id}_conv2")
            x = jax.nn.gelu(x + residual)

        out = _periodic_conv3d(
            x,
            1,
            conf,
            kernel_shape=3,
            name="out",
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
        )
        return self.max_residual * jnp.tanh(out[..., 0]) * source_scale


@lru_cache(maxsize=None)
def _radial_transform(n_knots, latent_size):
    _require_haiku("radial potential corrections")
    return hk.without_apply_rng(
        hk.transform(
            lambda x, a, c: NeuralSplineFourierFilter(
                n_knots=n_knots,
                latent_size=latent_size,
            )(x, a, c)
        )
    )


def _mesh_cnn_transform(conf, channels, depth, max_residual):
    _require_haiku("mesh CNN potential corrections")
    return hk.without_apply_rng(
        hk.transform(
            lambda source, a, c: MeshResidualSourceCorrection(
                channels=channels,
                depth=depth,
                max_residual=max_residual,
            )(source, a, c, conf)
        )
    )


def _normalized_k_magnitude_transposed(conf):
    kx, ky, kz = [jnp.squeeze(a).astype(conf.float_dtype) for a in conf.kvec]
    k_nyquist = jnp.asarray(jnp.pi / conf.cell_size, dtype=conf.float_dtype)
    return jnp.sqrt(
        (kx[:, None, None] / k_nyquist) ** 2
        + (ky[None, :, None] / k_nyquist) ** 2
        + (kz[None, None, :] / k_nyquist) ** 2
    )


def _k_squared_transposed(conf, dtype):
    kx, ky, kz = [jnp.squeeze(a).astype(dtype) for a in conf.kvec]
    return kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2


def _default_cosmo_features(dtype):
    return jnp.asarray([0.3, 0.8], dtype=dtype)


def resolve_sigma8(cosmo, dtype, allow_missing_sigma8=False):
    default_sigma8 = float(_default_cosmo_features(dtype)[1])
    if cosmo is None:
        return jnp.asarray(default_sigma8, dtype=dtype)
    try:
        sigma8 = float(jax.device_get(jnp.asarray(cosmo.sigma8, dtype=dtype)))
    except Exception:
        if not allow_missing_sigma8:
            raise ValueError(
                "sigma8 is not initialized on the cosmology. "
                "Run boltzmann()/from_sigma8() first, or set allow_missing_sigma8=True "
                "for lightweight test-only correction initialization."
            ) from None
        sigma8 = default_sigma8
    if not np.isfinite(sigma8):
        if not allow_missing_sigma8:
            raise ValueError(
                "sigma8 is non-finite on the cosmology. "
                "The upstream transfer/varlin initialization failed."
            )
        sigma8 = default_sigma8
    return jnp.asarray(sigma8, dtype=dtype)


def _cosmo_features(cosmo, dtype, allow_missing_sigma8=False):
    if cosmo is None:
        return _default_cosmo_features(dtype)
    if hasattr(cosmo, "shape"):
        return jnp.asarray(cosmo, dtype=dtype)
    sigma8 = resolve_sigma8(cosmo, dtype, allow_missing_sigma8=allow_missing_sigma8)
    return jnp.asarray([cosmo.Omega_m, sigma8], dtype=dtype)


def _correction_cosmo_features(correction, cosmo, dtype):
    if correction is not None and getattr(correction, "sigma8_value", None) is not None:
        if cosmo is None:
            return jnp.asarray([_default_cosmo_features(dtype)[0], correction.sigma8_value], dtype=dtype)
        if hasattr(cosmo, "shape"):
            return jnp.asarray(cosmo, dtype=dtype)
        return jnp.asarray([cosmo.Omega_m, correction.sigma8_value], dtype=dtype)
    allow_missing_sigma8 = False if correction is None else getattr(correction, "allow_missing_sigma8", False)
    return _cosmo_features(cosmo, dtype, allow_missing_sigma8=allow_missing_sigma8)


def _mesh_conditioning_channels(source, a, cosmo, dtype):
    spatial_shape = source.shape + (1,)
    a_channel = jnp.broadcast_to(jnp.asarray(a, dtype=dtype), spatial_shape)
    cosmo_features = _cosmo_features(cosmo, dtype)
    cosmo_channels = [
        jnp.broadcast_to(jnp.asarray(value, dtype=dtype), spatial_shape)
        for value in cosmo_features
    ]
    return [source[..., None], a_channel, *cosmo_channels]


def _periodic_pad_mesh_channels(x, halo_width, conf):
    if halo_width <= 0:
        x_halo = x
    elif conf.compute_mesh is None or conf.num_devices == 1:
        x_halo = jnp.pad(x, ((halo_width, halo_width), (0, 0), (0, 0), (0, 0)), mode="wrap")
    else:
        incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
            x,
            halo_width,
            conf.left_perm,
            conf.right_perm,
        )
        x_halo = extend_owned_mesh_from_halo_edges(x, incoming_left, incoming_right, halo_width)
    return jnp.pad(x_halo, ((0, 0), (halo_width, halo_width), (halo_width, halo_width), (0, 0)), mode="wrap")


def _periodic_conv3d(x, output_channels, conf, kernel_shape=3, name=None, w_init=None, b_init=None):
    _require_haiku("mesh CNN potential corrections")
    halo_width = kernel_shape // 2
    x = _periodic_pad_mesh_channels(x, halo_width, conf)
    conv = hk.Conv3D(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        padding="VALID",
        data_format="NDHWC",
        w_init=w_init,
        b_init=b_init,
        name=name,
    )
    return conv(x[None, ...])[0]


@partial(
    pytree_dataclass,
    aux_fields=("n_knots", "latent_size", "allow_missing_sigma8", "sigma8_value", "dtype"),
    frozen=True,
    eq=False,
)
class RadialPotentialCorrection:
    params: dict
    n_knots: int = 8
    latent_size: int = 16
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


@partial(
    pytree_dataclass,
    aux_fields=("channels", "depth", "max_residual", "allow_missing_sigma8", "sigma8_value", "dtype"),
    frozen=True,
    eq=False,
)
class MeshCNNPotentialCorrection:
    params: dict
    channels: int = 8
    depth: int = 4
    max_residual: float = 0.1
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
    allow_missing_sigma8=False,
    sigma8_value=0.8,
    dtype=jnp.float32,
    conf=None,
    **unused_kwargs,
):
    del unused_kwargs
    dtype = jnp.dtype(dtype)
    transform = _radial_transform(n_knots, latent_size)
    if conf is None:
        x = jnp.linspace(0.0, jnp.sqrt(3.0), 16, dtype=dtype)
    else:
        x = _normalized_k_magnitude_transposed(conf).astype(dtype)
    params = transform.init(
        key,
        x,
        jnp.ones((1,), dtype=dtype),
        _default_cosmo_features(dtype),
    )
    return RadialPotentialCorrection(
        params=params,
        n_knots=n_knots,
        latent_size=latent_size,
        allow_missing_sigma8=allow_missing_sigma8,
        sigma8_value=float(sigma8_value),
        dtype=dtype,
    )


def init_mesh_cnn_potential_correction(
    key,
    channels=8,
    depth=4,
    max_residual=0.1,
    allow_missing_sigma8=False,
    sigma8_value=0.8,
    dtype=jnp.float32,
    conf=None,
    **unused_kwargs,
):
    del unused_kwargs
    dtype = jnp.dtype(dtype)
    transform = _mesh_cnn_transform(conf, channels, depth, max_residual)
    if conf is None:
        source = jnp.zeros((16, 16, 16), dtype=dtype)
    else:
        source = jnp.zeros(tuple(conf.local_mesh_shape), dtype=dtype)
    params = transform.init(
        key,
        source,
        jnp.asarray(1.0, dtype=dtype),
        _default_cosmo_features(dtype),
    )
    return MeshCNNPotentialCorrection(
        params=params,
        channels=channels,
        depth=depth,
        max_residual=max_residual,
        allow_missing_sigma8=allow_missing_sigma8,
        sigma8_value=float(sigma8_value),
        dtype=dtype,
    )


def init_potential_correction(key, model="neural_spline", **kwargs):
    if model in {"neural_spline", "radial_spline", "radial", "radial_mlp"}:
        return init_radial_potential_correction(key, **kwargs)
    if model in {"mesh_cnn", "cnn"}:
        return init_mesh_cnn_potential_correction(key, **kwargs)
    raise ValueError(f"Unsupported correction model {model!r}.")


def sample_potential_transfer(correction, radius_fraction, a, cosmo, conf):
    if isinstance(correction, MeshCNNPotentialCorrection):
        raise TypeError("Mesh CNN corrections do not define a radial transfer curve.")
    if correction is None:
        return jnp.ones_like(radius_fraction, dtype=conf.float_dtype)
    radius_fraction = jnp.asarray(radius_fraction, dtype=correction.dtype)
    x = radius_fraction * jnp.asarray(jnp.sqrt(3.0), dtype=correction.dtype)
    transform = _radial_transform(correction.n_knots, correction.latent_size)
    residual = transform.apply(
        correction.params,
        x,
        jnp.asarray(a, dtype=correction.dtype),
        _correction_cosmo_features(correction, cosmo, correction.dtype),
    )
    transfer = 1.0 + residual
    return jnp.where(radius_fraction == 0, jnp.asarray(1.0, dtype=transfer.dtype), transfer)


def evaluate_radial_potential_transfer(correction, a, cosmo, conf):
    if isinstance(correction, MeshCNNPotentialCorrection):
        raise TypeError("Mesh CNN corrections do not define a radial transfer field.")
    if correction is None:
        return jnp.ones(tuple(conf.mesh_shape[:-1]) + (conf.mesh_shape[-1] // 2 + 1,), dtype=conf.float_dtype)
    k_norm = _normalized_k_magnitude_transposed(conf)
    transform = _radial_transform(correction.n_knots, correction.latent_size)
    residual = transform.apply(
        correction.params,
        k_norm.astype(correction.dtype),
        jnp.asarray(a, dtype=correction.dtype),
        _correction_cosmo_features(correction, cosmo, correction.dtype),
    )
    transfer = 1.0 + residual
    return transfer


def evaluate_potential_transfer(correction, a, cosmo, conf):
    return evaluate_radial_potential_transfer(correction, a, cosmo, conf)


def evaluate_mesh_source_residual(correction, source_real, a, cosmo, conf):
    if correction is None:
        return jnp.zeros_like(source_real, dtype=conf.float_dtype)
    if not isinstance(correction, MeshCNNPotentialCorrection):
        raise TypeError("Mesh source residuals are only defined for mesh CNN corrections.")
    transform = _mesh_cnn_transform(conf, correction.channels, correction.depth, correction.max_residual)
    residual = transform.apply(
        correction.params,
        source_real.astype(correction.dtype),
        jnp.asarray(a, dtype=correction.dtype),
        _correction_cosmo_features(correction, cosmo, correction.dtype),
    )
    return residual.astype(conf.float_dtype)


def apply_potential_correction(pot, a, cosmo, conf, correction, source_real=None):
    if correction is None:
        return pot
    if isinstance(correction, MeshCNNPotentialCorrection):
        if source_real is None:
            raise ValueError("source_real is required for mesh CNN potential corrections.")
        residual_source = evaluate_mesh_source_residual(
            correction,
            source_real,
            1.0 if a is None else a,
            cosmo,
            conf,
        )
        if conf.compute_mesh is None:
            residual_hat = jnp.fft.rfftn(residual_source)
        else:
            residual_hat = conf.mGPU_rfftn_transposed(residual_source)
        k2 = _k_squared_transposed(conf, pot.real.dtype)
        residual_pot = jnp.where(k2 != 0, -residual_hat / k2, 0)
        return pot + residual_pot.astype(pot.dtype)
    transfer = evaluate_potential_transfer(correction, 1.0 if a is None else a, cosmo, conf)
    return pot * transfer.astype(pot.real.dtype)


def zero_potential_correction_cotangent(correction):
    if correction is None:
        return None
    return tree_map(
        lambda x: x if x is None or is_float0_array(x) else jnp.zeros_like(x),
        correction,
    )


def add_potential_correction_cotangents(lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs

    def add_leaf(x, y):
        if x is None:
            return y
        if y is None:
            return x
        if is_float0_array(x):
            return y
        if is_float0_array(y):
            return x
        return x + y

    return tree_map(add_leaf, lhs, rhs)


def build_correction_optimizer(
    learning_rate,
    gradient_clip_norm=100.0,
    optimizer_name="adamax",
    apply_if_finite_steps=100,
):
    _require_optax("potential correction optimizer construction")
    transforms = []
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(gradient_clip_norm))

    if optimizer_name == "adamax":
        transforms.append(optax.adamax(learning_rate))
    elif optimizer_name == "adam":
        transforms.append(optax.adam(learning_rate))
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}.")

    optimizer = optax.chain(*transforms) if len(transforms) > 1 else transforms[0]
    if apply_if_finite_steps and apply_if_finite_steps > 0:
        optimizer = optax.apply_if_finite(optimizer, apply_if_finite_steps)
    return optimizer
