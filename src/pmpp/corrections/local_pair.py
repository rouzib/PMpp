"""Mesh-local, momentum-projected force corrections.

The literal particle graph described by the QUIJOTE correction design would
materialize tens to hundreds of neighbours for every particle at ``256**3``.
That is not a viable representation on the current two-GPU runtime.  This
module provides a scalable mesh-local surrogate: a compact radial convolution
of the deposited particle density.  Kernel weights are shared by lattice
offsets with the same radius, the predicted scalar potential is periodic, and
the force uses a centered finite difference.  The construction is invariant
to particle ordering and equivariant to integer-cell translations and
cubic-lattice rotations.  It does *not* claim continuous SO(3)/translation
equivariance or pairwise antisymmetric messages.  A final global mean
projection enforces zero net authoritative acceleration.  Stacking the radial
layers also makes the effective receptive field wider than the per-layer
``cutoff_cells`` value.

The correction is deliberately independent of the long-range PM potential.
It is evaluated at every force call and returns an acceleration residual that
can be added to the normal PM acceleration.
"""

from __future__ import annotations

from dataclasses import field
from functools import lru_cache, partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map

from ..cic.gather import gather_stacked_mesh_halo
from ..distributed.mesh_halo import maybe_shard_map_mesh_local_op, owned_mesh_partition_spec
from ..cic.scatter import scatter
from ..core.utils import is_float0_array, pytree_dataclass
from .common import HaikuModuleBase, correction_cosmo_features, default_cosmo_features, hk, require_haiku
from .mesh_cnn import _global_mesh_mean, _periodic_pad_mesh_channels


@lru_cache(maxsize=None)
def _radial_shell_layout(cutoff_cells: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a shell-index tensor and per-shell multiplicities."""
    cutoff = float(cutoff_cells)
    if not np.isfinite(cutoff) or cutoff < 1.0:
        raise ValueError("cutoff_cells must be finite and at least one mesh cell")
    radius = int(math.floor(cutoff))
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    xx, yy, zz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    radius2 = xx * xx + yy * yy + zz * zz
    valid = radius2 <= cutoff * cutoff + 1e-7
    shells = np.unique(radius2[valid])
    shell_index = np.full(radius2.shape, -1, dtype=np.int32)
    counts = np.empty((shells.size, ), dtype=np.float32)
    for shell_id, shell_radius2 in enumerate(shells):
        shell_mask = valid & (radius2 == shell_radius2)
        shell_index[shell_mask] = shell_id
        counts[shell_id] = float(np.count_nonzero(shell_mask))
    return shell_index, counts


def _radial_conv3d(x, output_channels, conf, *, cutoff_cells, name, zero_init=False, ):
    """Apply an explicitly periodic convolution with radial shell weights."""
    require_haiku("local pair corrections")
    shell_index_np, counts_np = _radial_shell_layout(float(cutoff_cells))
    radius = shell_index_np.shape[0] // 2
    x = _periodic_pad_mesh_channels(x, radius, conf)

    in_channels = int(x.shape[-1])
    shell_count = int(counts_np.size)
    if zero_init:
        initializer = hk.initializers.Constant(0.0)
    else:
        initializer = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
    shell_weights = hk.get_parameter(
        f"{name}_shell_weights", shape=(shell_count, in_channels, int(output_channels)), dtype=x.dtype,
        init=initializer,
    )
    bias = hk.get_parameter(
        f"{name}_bias", shape=(int(output_channels), ), dtype=x.dtype, init=hk.initializers.Constant(0.0),
    )

    shell_index = jnp.asarray(shell_index_np, dtype=jnp.int32)
    valid = shell_index >= 0
    safe_index = jnp.maximum(shell_index, 0)
    counts = jnp.asarray(counts_np, dtype=x.dtype)
    kernel = shell_weights[safe_index]
    kernel = kernel / counts[safe_index][..., None, None]
    kernel = jnp.where(valid[..., None, None], kernel, jnp.zeros_like(kernel))
    y = jax.lax.conv_general_dilated(
        x[None, ...], kernel, window_strides=(1, 1, 1), padding="VALID", dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )[0]
    return y + bias


class RadialLocalPairPotential(HaikuModuleBase):
    """Two-hidden-layer radial network producing a local scalar potential."""

    def __init__(self, channels=32, cutoff_cells=2.5, max_force_fraction=0.25, output_init_scale=0.0, name=None, ):
        require_haiku("local pair corrections")
        super().__init__(name=name)
        self.channels = int(channels)
        self.cutoff_cells = float(cutoff_cells)
        self.max_force_fraction = float(max_force_fraction)
        self.output_init_scale = float(output_init_scale)

    def __call__(self, source, a, cosmo_features, conf):
        dtype = source.dtype
        source_scale = jnp.sqrt(_global_mesh_mean(source * source, conf) + jnp.asarray(1e-6, dtype=dtype))
        source_norm = source / source_scale
        spatial_shape = source.shape + (1, )
        mesh_ratio = jnp.asarray(conf.mesh_shape[0] / conf.ptcl_grid_shape[0], dtype=dtype)
        scalar_features = [
            jnp.broadcast_to(jnp.asarray(a, dtype=dtype), spatial_shape),
            jnp.broadcast_to(mesh_ratio, spatial_shape),
        ]
        scalar_features.extend(
            jnp.broadcast_to(jnp.asarray(value, dtype=dtype), spatial_shape) for value in cosmo_features
        )
        x = jnp.concatenate([source_norm[..., None], *scalar_features], axis=-1)
        x = jax.nn.gelu(_radial_conv3d(x, self.channels, conf, cutoff_cells=self.cutoff_cells, name="radial_0", ))
        x = jax.nn.gelu(_radial_conv3d(x, self.channels, conf, cutoff_cells=self.cutoff_cells, name="radial_1", ))
        out = _radial_conv3d(x, 1, conf, cutoff_cells=1.0, name="out", zero_init=self.output_init_scale == 0.0,
                             )[..., 0]
        if self.output_init_scale != 0.0:
            gain = hk.get_parameter(
                "output_gain", shape=(), dtype=dtype, init=hk.initializers.Constant(self.output_init_scale),
            )
            out = out * gain

        # A density source has units of one; multiplying by dx**2 gives the
        # scalar potential the units needed for one spatial derivative to be
        # an acceleration-like PM residual.  The 3/2 Omega_m normalization is
        # already present in the long-range Poisson source.
        omega_m = jnp.asarray(cosmo_features[0], dtype=dtype)
        # The local model always lives on the particle grid, including when
        # the long-range force mesh is refined by a factor of two.
        potential_scale = (
            jnp.asarray(1.5, dtype=dtype) * omega_m * source_scale * jnp.asarray(conf.ptcl_spacing**2, dtype=dtype)
        )
        return (jnp.asarray(self.max_force_fraction, dtype=dtype) * jnp.tanh(out) * potential_scale)


def local_pair_transform(conf, channels, cutoff_cells, max_force_fraction, output_init_scale):
    """Build the Haiku transform for a fixed PM configuration."""
    require_haiku("local pair corrections")
    return hk.without_apply_rng(
        hk.transform(
            lambda source, a, features: RadialLocalPairPotential(
                channels=channels, cutoff_cells=cutoff_cells, max_force_fraction=max_force_fraction, output_init_scale=
                output_init_scale,
            )(source, a, features, conf)
        )
    )


@partial(
    pytree_dataclass, aux_fields=(
        "channels", "cutoff_cells", "max_force_fraction", "output_init_scale", "allow_missing_sigma8", "sigma8_value",
        "dtype",
    ), frozen=True, eq=False,
)
class LocalPairCorrection:
    """Trainable finite-support local-pair surrogate.

    ``params`` are dynamic PyTree leaves.  Architecture and normalization
    values are static metadata and must match when loading a checkpoint.
    """

    params: dict
    channels: int = 32
    cutoff_cells: float = 2.5
    max_force_fraction: float = 0.25
    output_init_scale: float = 0.0
    allow_missing_sigma8: bool = False
    sigma8_value: float = 0.8
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        dtype = jnp.dtype(self.dtype)
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        _radial_shell_layout(float(self.cutoff_cells))
        if not 0.0 <= float(self.max_force_fraction) <= 1.0:
            raise ValueError("max_force_fraction must lie in [0, 1]")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self, "params",
            tree_map(lambda x: x if x is None or is_float0_array(x) else jnp.asarray(x, dtype=dtype), self.params,
                     ),
        )

    def acceleration_residual(self, a, ptcl, cosmo, conf):
        """Implement the generic :class:`NBodyCorrection` local-force protocol."""
        return apply_local_pair_correction(self, a, ptcl, cosmo, conf)


class LocalPairPhaseContext(NamedTuple):
    """Owner-aligned local features cached before the phase drift."""

    local_acceleration: jax.Array
    sigma8_value: jax.Array


def init_local_pair_correction(
    key, *, conf, channels=32, cutoff_cells=2.5, max_force_fraction=0.25, output_init_scale=0.0,
    allow_missing_sigma8=False, sigma8_value=0.8, dtype=jnp.float32,
):
    """Initialize a local correction with an exactly zero default output."""
    if conf is None:
        raise ValueError("conf is required to initialize a local pair correction")
    dtype = jnp.dtype(dtype)
    init_conf = conf
    if init_conf.use_mGPU:
        init_conf = init_conf.replace(multigpu=None, compute_mesh=None)
    transform = local_pair_transform(
        init_conf, int(channels), float(cutoff_cells), float(max_force_fraction), float(output_init_scale),
    )
    init_shape = tuple(min(int(size), 12) for size in init_conf.mesh_shape)
    params = transform.init(
        key, jnp.zeros(init_shape, dtype=dtype), jnp.asarray(1.0, dtype=dtype), default_cosmo_features(dtype),
    )
    return LocalPairCorrection(
        params=params, channels=int(channels), cutoff_cells=float(cutoff_cells),
        max_force_fraction=float(max_force_fraction), output_init_scale=float(output_init_scale),
        allow_missing_sigma8=bool(allow_missing_sigma8), sigma8_value=float(sigma8_value), dtype=dtype,
    )


def evaluate_local_pair_potential(correction, source, a, cosmo, conf):
    """Evaluate the periodic scalar potential predicted by ``correction``."""
    if correction is None:
        return jnp.zeros_like(source, dtype=conf.float_dtype)
    if not isinstance(correction, LocalPairCorrection):
        raise TypeError("local pair potential requires LocalPairCorrection")
    transform = local_pair_transform(
        conf, correction.channels, correction.cutoff_cells, correction.max_force_fraction, correction.output_init_scale,
    )
    apply_fn = partial(transform.apply)
    mesh_spec = owned_mesh_partition_spec(source.ndim)
    apply_fn = maybe_shard_map_mesh_local_op(
        apply_fn, conf, in_specs=(None, mesh_spec, None, None), out_specs=mesh_spec, check_rep=False,
    )
    return apply_fn(
        correction.params, source.astype(correction.dtype), jnp.asarray(a, dtype=correction.dtype),
        correction_cosmo_features(correction, cosmo, correction.dtype),
    ).astype(conf.float_dtype)


def _negative_central_gradient(potential, conf):
    """Return a periodic antisymmetric central-difference force mesh."""
    padded = _periodic_pad_mesh_channels(potential[..., None], 1, conf)[..., 0]
    inv_two_dx = jnp.asarray(0.5 / conf.ptcl_spacing, dtype=potential.dtype)
    grad_x = -(padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) * inv_two_dx
    grad_y = -(padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) * inv_two_dx
    grad_z = -(padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) * inv_two_dx
    return jnp.stack([grad_x, grad_y, grad_z], axis=-1)


def _mesh_refinement_factors(conf):
    factors = []
    for mesh_size, particle_size in zip(conf.mesh_shape, conf.ptcl_grid_shape):
        if mesh_size % particle_size != 0:
            raise ValueError(
                "Local pair correction requires integer force-mesh refinement; "
                f"got mesh_shape={conf.mesh_shape} and ptcl_grid_shape={conf.ptcl_grid_shape}."
            )
        factors.append(int(mesh_size // particle_size))
    if len(set(factors)) != 1:
        raise ValueError("Local pair correction requires isotropic mesh refinement.")
    if factors[0] not in (1, 2):
        raise ValueError("Local pair correction currently supports mesh_shape ratios 1 and 2.")
    return tuple(factors)


def _downsample_force_source_to_particle_grid(source, factors):
    if all(factor == 1 for factor in factors):
        return source
    sx, sy, sz = source.shape
    fx, fy, fz = factors
    return source.reshape(sx // fx, fx, sy // fy, fy, sz // fz, fz, ).mean(axis=(1, 3, 5))


def _upsample_particle_force_to_force_mesh(force_mesh, factors):
    if all(factor == 1 for factor in factors):
        return force_mesh
    out = force_mesh
    for axis, factor in enumerate(factors):
        out = jnp.repeat(out, factor, axis=axis)
    return out


def apply_local_pair_correction(correction, a, ptcl, cosmo, conf):
    """Return the particle acceleration residual from a local correction."""
    if correction is None:
        return jnp.zeros_like(ptcl.disp, dtype=conf.float_dtype)
    factors = _mesh_refinement_factors(conf)
    source = scatter(ptcl, conf) - jnp.asarray(1.0, dtype=conf.float_dtype)
    source = _downsample_force_source_to_particle_grid(source, factors)
    potential = evaluate_local_pair_potential(correction, source, a, cosmo, conf)
    mesh_spec = owned_mesh_partition_spec(potential.ndim)
    vector_spec = owned_mesh_partition_spec(potential.ndim + 1)
    gradient_fn = maybe_shard_map_mesh_local_op(
        partial(_negative_central_gradient, conf=conf), conf, in_specs=(mesh_spec, ), out_specs=vector_spec,
        check_rep=False,
    )
    force_mesh = _upsample_particle_force_to_force_mesh(gradient_fn(potential), factors)
    residual = gather_stacked_mesh_halo(ptcl, conf, force_mesh)
    active = jnp.ones(ptcl.disp.shape[:-1], dtype=jnp.bool_)
    if ptcl.unused_index is not None:
        active = active & ~ptcl.unused_index
    authoritative = active
    if ptcl.halo_mask is not None:
        authoritative = authoritative & ~ptcl.halo_mask
    weights = authoritative.astype(residual.dtype)
    count = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=residual.dtype))
    mean = jnp.sum(residual * weights[..., None], axis=tuple(range(residual.ndim - 1))) / count
    residual = jnp.where(active[..., None], residual - mean, jnp.zeros_like(residual))
    return residual.astype(conf.float_dtype)


def _normalized_particle_vector(vector, ptcl):
    """Normalize a vector feature over authoritative particles."""
    vector = jnp.asarray(vector, dtype=ptcl.disp.dtype)
    mask = jnp.ones(ptcl.disp.shape[:-1], dtype=jnp.bool_)
    if ptcl.unused_index is not None:
        mask = mask & ~ptcl.unused_index
    if ptcl.halo_mask is not None:
        mask = mask & ~ptcl.halo_mask
    weights = mask.astype(vector.dtype)
    count = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=vector.dtype))
    rms = jnp.sqrt(jnp.sum(jnp.sum(vector * vector, axis=-1) * weights) / count + jnp.asarray(1e-8, dtype=vector.dtype))
    normalized = vector / rms
    return jnp.where(mask[..., None], normalized, jnp.zeros_like(normalized))


def prepare_local_pair_phase_context(a, ptcl, cosmo, conf, local_pair):
    """Cache local acceleration before a drift can invalidate slab ownership."""
    dtype = ptcl.disp.dtype
    if local_pair is None:
        local_acceleration = jnp.zeros_like(ptcl.disp)
        sigma8_value = jnp.asarray(0.8, dtype=dtype)
    else:
        evaluator = getattr(local_pair, "acceleration_residual", None)
        if evaluator is None:
            raise TypeError(
                "local_pair phase context requires a correction with "
                "acceleration_residual(a, ptcl, cosmo, conf)"
            )
        local_acceleration = evaluator(a, ptcl, cosmo, conf)
        sigma8_value = jnp.asarray(getattr(local_pair, "sigma8_value", 0.8), dtype=dtype)
    return LocalPairPhaseContext(local_acceleration, sigma8_value)


def local_pair_phase_space_head(params, a, ptcl, cosmo, conf, local_pair):
    """Scalar-mixing phase head using local acceleration and velocity bases.

    The MLP produces scalar coefficients only.  Multiplying those scalars by
    physical vector bases preserves the transformation properties those bases
    already have; it does not upgrade the mesh surrogate to continuous
    rotation or translation equivariance.  The enclosing bounded phase
    correction removes the authoritative mean and applies the configured
    0.25-cell limits.
    """
    dtype = ptcl.disp.dtype
    omega_m = jnp.asarray(getattr(cosmo, "Omega_m", 0.3), dtype=dtype)
    if isinstance(local_pair, LocalPairPhaseContext):
        local_acceleration = jnp.asarray(local_pair.local_acceleration, dtype=dtype)
        sigma8 = jnp.asarray(local_pair.sigma8_value, dtype=dtype)
    elif local_pair is None:
        local_acceleration = jnp.zeros_like(ptcl.disp)
        sigma8 = jnp.asarray(0.8, dtype=dtype)
    else:
        evaluator = getattr(local_pair, "acceleration_residual", None)
        if evaluator is None:
            raise TypeError(
                "local_pair_phase_space_head requires a local correction with "
                "acceleration_residual(a, ptcl, cosmo, conf)"
            )
        local_acceleration = evaluator(a, ptcl, cosmo, conf)
        sigma8 = jnp.asarray(getattr(local_pair, "sigma8_value", 0.8), dtype=dtype)
    mesh_ratio = jnp.asarray(conf.mesh_shape[0] / conf.ptcl_grid_shape[0], dtype=dtype)
    scale_factor = jnp.asarray(a, dtype=dtype)
    features = jnp.asarray([scale_factor, scale_factor * scale_factor, mesh_ratio, omega_m, sigma8], dtype=dtype, )
    hidden = jax.nn.gelu(features @ params["input_weight"] + params["input_bias"])
    coefficients = hidden @ params["output_weight"] + params["output_bias"]

    velocity = ptcl.vel if ptcl.vel is not None else jnp.zeros_like(ptcl.disp)
    acceleration = ptcl.acc if ptcl.acc is not None else jnp.zeros_like(ptcl.disp)
    bases = jnp.stack([
        _normalized_particle_vector(local_acceleration, ptcl),
        _normalized_particle_vector(acceleration, ptcl),
        _normalized_particle_vector(velocity, ptcl),
    ], axis=0,
                      )
    raw_displacement = jnp.sum(coefficients[:3, None, None] * bases, axis=0)
    raw_velocity = jnp.sum(coefficients[3:, None, None] * bases, axis=0)
    return raw_displacement, raw_velocity


def init_local_pair_phase_space_correction(
    key, *, channels=32, max_displacement_cells=0.25, max_velocity_cells=0.25, dtype=jnp.float32,
):
    """Initialize a zero-output two-layer phase head for ``NBodyCorrection``."""
    from .nbody import BoundedPhaseSpaceCorrection

    if channels <= 0:
        raise ValueError("channels must be positive")
    dtype = jnp.dtype(dtype)
    input_key, _ = jax.random.split(key)
    input_weight = jax.random.normal(input_key, (5, int(channels)), dtype=dtype)
    input_weight = input_weight / jnp.sqrt(jnp.asarray(5.0, dtype=dtype))
    params = {
        "input_weight": input_weight,
        "input_bias": jnp.zeros((int(channels), ), dtype=dtype),
        "output_weight": jnp.zeros((int(channels), 6), dtype=dtype),
        "output_bias": jnp.zeros((6, ), dtype=dtype),
    }
    return BoundedPhaseSpaceCorrection(
        params=params, apply_fn=local_pair_phase_space_head, context_fn=prepare_local_pair_phase_context,
        max_displacement_cells=float(max_displacement_cells), max_velocity_cells=float(max_velocity_cells),
        mean_free=True, invertible=False, dtype=dtype,
    )


__all__ = [
    "LocalPairCorrection", "LocalPairPhaseContext", "RadialLocalPairPotential", "apply_local_pair_correction",
    "evaluate_local_pair_potential", "init_local_pair_correction", "init_local_pair_phase_space_correction",
    "local_pair_phase_space_head", "prepare_local_pair_phase_context",
]
