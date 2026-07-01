"""Mesh-CNN residual potential corrections."""

from dataclasses import field
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from ..mesh_halo import (
    exchange_owned_mesh_halo_edges,
    extend_owned_mesh_from_halo_edges,
    maybe_shard_map_mesh_local_op,
    owned_mesh_partition_spec,
)
from ..utils import is_float0_array, pytree_dataclass
from .common import (
    HaikuModuleBase,
    correction_cosmo_features,
    cosmo_features,
    default_cosmo_features,
    require_haiku,
    hk,
)


def _mesh_conditioning_channels(source, potential_real, a, cosmo, dtype):
    """Build per-cell channels used to condition the mesh CNN."""
    spatial_shape = source.shape + (1,)
    a_channel = jnp.broadcast_to(jnp.asarray(a, dtype=dtype), spatial_shape)
    features = cosmo_features(cosmo, dtype)
    cosmo_channels = [
        jnp.broadcast_to(jnp.asarray(value, dtype=dtype), spatial_shape)
        for value in features
    ]
    return [source[..., None], potential_real[..., None], a_channel, *cosmo_channels]


def _mesh_conditioning_vector(a, cosmo, dtype):
    """Build global conditioning features for scalar gain layers."""
    return jnp.concatenate(
        [
            jnp.asarray([a], dtype=dtype),
            cosmo_features(cosmo, dtype),
        ]
    )


def _periodic_pad_mesh_channels(x, halo_width, conf):
    """Pad mesh-channel data periodically, exchanging x halos when sharded."""
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
    """Apply a valid 3D convolution after explicit periodic padding."""
    require_haiku("mesh CNN potential corrections")
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


class MeshResidualPotentialCorrection(HaikuModuleBase):
    """Small periodic 3D CNN that predicts a residual real-space potential field."""

    def __init__(self, channels=8, depth=4, max_residual=0.1, output_init_scale=1e-2, name=None):
        require_haiku("mesh CNN potential corrections")
        super().__init__(name=name)
        self.channels = channels
        self.depth = depth
        self.max_residual = max_residual
        self.output_init_scale = output_init_scale

    def __call__(self, source, potential_real, a, cosmo, conf):
        dtype = source.dtype
        source_scale = jnp.sqrt(jnp.mean(source ** 2) + jnp.asarray(1e-6, dtype=dtype))
        source_norm = source / source_scale
        potential_scale = jnp.sqrt(jnp.mean(potential_real ** 2) + jnp.asarray(1e-6, dtype=dtype))
        potential_norm = potential_real / potential_scale
        features = _mesh_conditioning_channels(source_norm, potential_norm, a, cosmo, dtype)
        conditioning = _mesh_conditioning_vector(a, cosmo, dtype)
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
        conv_residual = self.max_residual * jnp.tanh(out[..., 0]) * potential_scale
        direct_gain = self.max_residual * jnp.tanh(
            hk.Linear(
                1,
                name="direct_gain",
                w_init=hk.initializers.RandomNormal(stddev=self.output_init_scale),
                b_init=hk.initializers.Constant(0.0),
            )(conditioning)[0]
        )
        return direct_gain * potential_real + conv_residual


def mesh_cnn_transform(conf, channels, depth, max_residual, output_init_scale):
    """Return a Haiku transform bound to the mesh CNN architecture and config.

    Parameters
    ----------
    channels
        Number of hidden convolution channels.
    depth
        Number of convolutional residual layers.
    max_residual
        Maximum residual amplitude applied by the mesh CNN correction.
    output_init_scale
        Scale of the final-layer initializer."""
    require_haiku("mesh CNN potential corrections")
    return hk.without_apply_rng(
        hk.transform(
            lambda source, potential_real, a, c: MeshResidualPotentialCorrection(
                channels=channels,
                depth=depth,
                max_residual=max_residual,
                output_init_scale=output_init_scale,
            )(source, potential_real, a, c, conf)
        )
    )


@partial(
    pytree_dataclass,
    aux_fields=("channels", "depth", "max_residual", "output_init_scale", "allow_missing_sigma8", "sigma8_value", "dtype"),
    frozen=True,
    eq=False,
)
class MeshCNNPotentialCorrection:
    """Trainable real-space residual potential correction."""

    params: dict
    channels: int = 8
    depth: int = 4
    max_residual: float = 0.1
    output_init_scale: float = 1e-2
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


def init_mesh_cnn_potential_correction(
    key,
    channels=8,
    depth=4,
    max_residual=0.1,
    output_init_scale=1e-2,
    allow_missing_sigma8=False,
    sigma8_value=0.8,
    dtype=jnp.float32,
    conf=None,
    **unused_kwargs,
):
    """Initialize a mesh-CNN residual potential correction.

    Parameters
    ----------
    key : jax.Array
        PRNG key used to initialize Haiku parameters.
    channels : int, optional
        Hidden channel count in the residual CNN.
    depth : int, optional
        Number of convolutional blocks.
    max_residual : float, optional
        Maximum multiplicative scale applied to the learned residual branch.
    output_init_scale : float, optional
        Output-layer initialization scale.
    allow_missing_sigma8 : bool, optional
        Whether later evaluation may fall back to a stored ``sigma8_value``.
    sigma8_value : float, optional
        Stored fallback conditioning value.
    dtype : DTypeLike, optional
        Parameter dtype.
    conf : Configuration, optional
        Representative configuration used to choose an initialization mesh
        shape. Multi-GPU configs are converted to a single-device init shape.

    Returns
    -------
    MeshCNNPotentialCorrection
        Initialized correction container with Haiku parameters and metadata.

    Parameters
    ----------
    unused_kwargs
        Unused keyword options accepted for API compatibility."""
    del unused_kwargs
    dtype = jnp.dtype(dtype)
    init_conf = conf
    if init_conf is not None and init_conf.use_mGPU:
        init_conf = init_conf.replace(multigpu=None, compute_mesh=None)
    transform = mesh_cnn_transform(init_conf, channels, depth, max_residual, output_init_scale)
    init_shape = (16, 16, 16) if conf is None else tuple(min(int(s), 16) for s in conf.mesh_shape)
    source = jnp.zeros(init_shape, dtype=dtype)
    potential_real = jnp.zeros_like(source)
    params = transform.init(
        key,
        source,
        potential_real,
        jnp.asarray(1.0, dtype=dtype),
        default_cosmo_features(dtype),
    )
    return MeshCNNPotentialCorrection(
        params=params,
        channels=channels,
        depth=depth,
        max_residual=max_residual,
        output_init_scale=output_init_scale,
        allow_missing_sigma8=allow_missing_sigma8,
        sigma8_value=float(sigma8_value),
        dtype=dtype,
    )


def _apply_mesh_cnn_residual(params, source_real, potential_real, a, cosmo_features_in, conf, correction):
    """Apply the mesh CNN and cast the residual back to solver precision."""
    transform = mesh_cnn_transform(
        conf,
        correction.channels,
        correction.depth,
        correction.max_residual,
        correction.output_init_scale,
    )
    residual = transform.apply(
        params,
        source_real,
        potential_real,
        a,
        cosmo_features_in,
    )
    return residual.astype(conf.float_dtype)


def evaluate_mesh_potential_residual(correction, source_real, potential_real, a, cosmo, conf):
    """Evaluate a mesh CNN residual potential on the local or sharded mesh.

    Parameters
    ----------
    correction
        Potential-correction pytree or ``None`` for the uncorrected PM force.
    source_real
        Real-space source density mesh.
    potential_real
        Real-space potential mesh.
    cosmo
        Cosmology object supplying density, growth, and transfer parameters.
    conf
        Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers."""
    if correction is None:
        return jnp.zeros_like(potential_real, dtype=conf.float_dtype)
    if not isinstance(correction, MeshCNNPotentialCorrection):
        raise TypeError("Mesh potential residuals are only defined for mesh CNN corrections.")
    apply_fn = partial(_apply_mesh_cnn_residual, conf=conf, correction=correction)
    mesh_spec = owned_mesh_partition_spec(source_real.ndim)
    apply_fn = maybe_shard_map_mesh_local_op(
        apply_fn,
        conf,
        in_specs=(None, mesh_spec, mesh_spec, None, None),
        out_specs=mesh_spec,
        check_rep=False,
    )
    return apply_fn(
        correction.params,
        source_real.astype(correction.dtype),
        potential_real.astype(correction.dtype),
        jnp.asarray(a, dtype=correction.dtype),
        correction_cosmo_features(correction, cosmo, correction.dtype),
    )


def evaluate_mesh_source_residual(correction, source_real, a, cosmo, conf):
    """Evaluate a mesh CNN source-only residual using a zero input potential.

    Parameters
    ----------
    correction
        Potential-correction pytree or ``None`` for the uncorrected PM force.
    source_real
        Real-space source density mesh.
    cosmo
        Cosmology object supplying density, growth, and transfer parameters.
    conf
        Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers."""
    potential_real = jnp.zeros_like(source_real, dtype=conf.float_dtype)
    return evaluate_mesh_potential_residual(correction, source_real, potential_real, a, cosmo, conf)
