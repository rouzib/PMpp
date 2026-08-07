"""Composite force and phase-space corrections for N-body integration.

Potential corrections predate the N-body correction interface and remain valid
inputs to :func:`pmpp.nbody.nbody`.  :class:`NBodyCorrection` adds optional
particle-force and direct phase-space branches without changing that legacy
contract.  The helpers in this module deliberately keep dispatch at Python
trace time so correction parameters remain ordinary dynamic JAX pytree leaves.
"""

from dataclasses import field
from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from ..core.utils import is_float0_array, pytree_dataclass


def _default_phase_space_head(params, a, ptcl, cosmo, conf, local_pair):
    """Broadcast zero-initializable displacement and velocity bias heads."""
    del a, cosmo, conf, local_pair
    displacement = jnp.broadcast_to(params["displacement"], ptcl.disp.shape)
    velocity = jnp.broadcast_to(params["velocity"], ptcl.vel.shape)
    return displacement, velocity


@partial(
    pytree_dataclass, aux_fields=(
        "apply_fn", "context_fn", "max_displacement_cells", "max_velocity_cells", "mean_free", "invertible", "dtype",
    ), frozen=True, eq=False,
)
class BoundedPhaseSpaceCorrection:
    """A bounded direct residual applied once per N-body macro-step.

    ``apply_fn`` has the signature

    ``apply_fn(params, a, ptcl, cosmo, conf, local_pair) -> (raw_disp, raw_vel)``.

    An optional ``context_fn(a, ptcl, cosmo, conf, local_pair)`` runs on the
    owner-aligned pre-drift state.  Its returned pytree is supplied to
    ``apply_fn`` through the existing final argument when the residual is
    evaluated after the raw drift.

    The two raw heads are converted to mean-free residuals and bounded in
    particle-cell units by :func:`apply_phase_space_correction`.  The velocity
    bound is expressed as the displacement that the velocity residual would
    produce over the current macro-step drift factor.

    ``invertible`` is reserved for a future explicit inverse-map protocol.
    Setting it today is rejected so reverse integration can never silently
    apply the forward residual as though it were an inverse.
    """

    params: Any
    apply_fn: Callable = field(default=_default_phase_space_head, repr=False)
    context_fn: Optional[Callable] = field(default=None, repr=False)
    max_displacement_cells: float = 0.25
    max_velocity_cells: float = 0.25
    mean_free: bool = True
    invertible: bool = False
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        if not callable(self.apply_fn):
            raise TypeError("BoundedPhaseSpaceCorrection.apply_fn must be callable.")
        if self.context_fn is not None and not callable(self.context_fn):
            raise TypeError("BoundedPhaseSpaceCorrection.context_fn must be callable or None.")
        if self.max_displacement_cells < 0 or self.max_velocity_cells < 0:
            raise ValueError("Phase-space correction bounds must be non-negative.")
        if self.invertible:
            raise ValueError(
                "invertible=True requires an explicit inverse phase-map protocol, "
                "which is not implemented."
            )
        dtype = jnp.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(
            self, "params",
            tree_map(
                lambda value: value
                if value is None or is_float0_array(value) else jnp.asarray(value, dtype=dtype), self.params,
            ),
        )


@partial(pytree_dataclass, frozen=True, eq=False)
class NBodyCorrection:
    """Dynamic composite correction accepted by the N-body solver.

    Parameters
    ----------
    long_range
        Existing Fourier/mesh potential correction.  Passing such a correction
        directly to ``nbody`` remains supported and is equivalent to placing it
        in this field.
    local_pair
        Optional short-range particle-force block.  It may implement an
        ``acceleration_residual(a, ptcl, cosmo, conf)`` method or be callable
        with those four arguments.
    phase_space
        Optional :class:`BoundedPhaseSpaceCorrection` applied after the raw
        final drift and before particle ownership migration.
    """

    long_range: Any = None
    local_pair: Any = None
    phase_space: Optional[BoundedPhaseSpaceCorrection] = None


def init_bounded_phase_space_correction(
    params=None, *, apply_fn=_default_phase_space_head, context_fn=None, max_displacement_cells=0.25,
    max_velocity_cells=0.25, mean_free=True, invertible=False, dtype=jnp.float32,
):
    """Construct a bounded phase-space correction with zero heads by default."""
    dtype = jnp.dtype(dtype)
    if params is None:
        params = {"displacement": jnp.zeros((3, ), dtype=dtype), "velocity": jnp.zeros((3, ), dtype=dtype), }
    return BoundedPhaseSpaceCorrection(
        params=params, apply_fn=apply_fn, context_fn=context_fn, max_displacement_cells=max_displacement_cells,
        max_velocity_cells=max_velocity_cells, mean_free=mean_free, invertible=invertible, dtype=dtype,
    )


def long_range_correction(correction):
    """Return the potential branch while preserving legacy correction inputs."""
    if isinstance(correction, NBodyCorrection):
        return correction.long_range
    return correction


def local_pair_correction(correction):
    """Return the local particle-force branch of a composite correction."""
    if isinstance(correction, NBodyCorrection):
        return correction.local_pair
    return None


def phase_space_correction(correction):
    """Return the direct phase-space branch of a composite correction."""
    if isinstance(correction, NBodyCorrection):
        return correction.phase_space
    return None


def has_phase_space_correction(correction):
    """Return whether a correction contains a direct phase-space map."""
    return phase_space_correction(correction) is not None


def phase_space_is_invertible(correction):
    """Return whether an active phase-space branch declares exact inversion."""
    phase = phase_space_correction(correction)
    return phase is None or bool(getattr(phase, "invertible", False))


def apply_local_pair_correction(correction, a, ptcl, cosmo, conf):
    """Evaluate a local particle-force residual through a small protocol.

    Concrete local-pair implementations can expose an
    ``acceleration_residual`` method.  A plain callable is also accepted, which
    keeps experimentation possible without adding solver-side type branches.
    """
    if correction is None:
        return jnp.zeros_like(ptcl.disp)
    evaluator = getattr(correction, "acceleration_residual", None)
    if evaluator is not None:
        residual = evaluator(a, ptcl, cosmo, conf)
    elif callable(correction):
        residual = correction(a, ptcl, cosmo, conf)
    else:
        raise TypeError(
            "A local-pair correction must be callable or implement "
            "acceleration_residual(a, ptcl, cosmo, conf)."
        )
    residual = jnp.asarray(residual, dtype=ptcl.disp.dtype)
    if residual.shape != ptcl.disp.shape:
        raise ValueError(
            "Local-pair acceleration residual must match particle displacement "
            f"shape {ptcl.disp.shape}, got {residual.shape}."
        )
    return residual


def _particle_masks(ptcl):
    """Return active and authoritative masks for phase-space normalization."""
    active = jnp.ones(ptcl.disp.shape[:-1], dtype=jnp.bool_)
    if ptcl.unused_index is not None:
        active = active & ~ptcl.unused_index
    authoritative = active
    if ptcl.halo_mask is not None:
        authoritative = authoritative & ~ptcl.halo_mask
    return active, authoritative


def prepare_phase_space_context(correction, a, ptcl, cosmo, conf, local_pair=None):
    """Build optional phase-head context while particle ownership is valid.

    Direct phase residuals are applied after a raw drift but before the one
    ownership migration for that drift.  A context function lets a head cache
    mesh-local features from the pre-drift, owner-aligned state instead of
    performing a mesh scatter after particles may already have crossed a slab.
    Heads without a context function retain the original call protocol.
    """
    if correction is None or correction.context_fn is None:
        return None
    return correction.context_fn(a, ptcl, cosmo, conf, local_pair)


def _mean_free_bounded_vectors(raw, ptcl, bound, mean_free):
    """Map raw vectors to a globally bounded, authoritative-mean-free field."""
    raw = jnp.asarray(raw, dtype=ptcl.disp.dtype)
    if raw.shape != ptcl.disp.shape:
        raise ValueError(
            "Phase-space residual heads must match particle displacement shape "
            f"{ptcl.disp.shape}, got {raw.shape}."
        )

    active, authoritative = _particle_masks(ptcl)
    vectors = jnp.tanh(raw)
    if mean_free:
        weights = authoritative.astype(vectors.dtype)
        count = jnp.maximum(weights.sum(), jnp.asarray(1, dtype=vectors.dtype))
        mean = (vectors * weights[..., None]).sum(axis=tuple(range(vectors.ndim - 1))) / count
        vectors = vectors - mean

    # A single global rescaling preserves an exact zero authoritative mean while
    # enforcing a vector-norm (rather than per-component) bound.
    # Work in squared norm so the zero-initialized heads have a finite VJP.
    # ``linalg.norm`` has an undefined derivative at the origin even though the
    # inactive bound branch should mathematically contribute no gradient.
    norm_sq = jnp.sum(jnp.square(jnp.where(active[..., None], vectors, 0)), axis=-1)
    peak_sq = jnp.max(norm_sq, initial=jnp.asarray(0, dtype=norm_sq.dtype))
    scale = jnp.reciprocal(jnp.sqrt(jnp.maximum(peak_sq, jnp.asarray(1, dtype=peak_sq.dtype))))
    vectors = vectors * scale * jnp.asarray(bound, dtype=vectors.dtype)
    return jnp.where(active[..., None], vectors, 0)


def evaluate_phase_space_residual(
    correction, a, ptcl, cosmo, conf, drift_scale, local_pair=None, context=None, reduction_backend="separate",
):
    """Evaluate bounded displacement and velocity residuals without applying them."""
    if correction is None:
        zeros = jnp.zeros_like(ptcl.disp)
        return zeros, zeros
    if not isinstance(correction, BoundedPhaseSpaceCorrection):
        raise TypeError("phase_space must be a BoundedPhaseSpaceCorrection.")

    # Keep the public six-argument head protocol stable.  Context-aware heads
    # receive their prepared pytree in the existing ``local_pair`` slot; generic
    # heads continue to receive the correction object itself.
    head_context = local_pair if context is None else context
    raw_disp, raw_vel = correction.apply_fn(correction.params, a, ptcl, cosmo, conf, head_context, )
    disp_bound = jnp.asarray(correction.max_displacement_cells * conf.ptcl_spacing, dtype=ptcl.disp.dtype)
    if reduction_backend == "fused":
        return evaluate_phase_space_residual_fused(
            correction, a, ptcl, cosmo, conf, drift_scale, local_pair=local_pair, context=context,
        )
    if reduction_backend != "separate":
        raise ValueError("reduction_backend must be 'separate' or 'fused'")

    disp_delta = _mean_free_bounded_vectors(raw_disp, ptcl, disp_bound, correction.mean_free, )

    drift_scale = jnp.asarray(drift_scale, dtype=ptcl.disp.dtype)
    tiny = jnp.asarray(jnp.finfo(ptcl.disp.dtype).tiny, dtype=ptcl.disp.dtype)
    vel_bound = correction.max_velocity_cells * conf.ptcl_spacing / jnp.maximum(jnp.abs(drift_scale), tiny)
    vel_delta = _mean_free_bounded_vectors(raw_vel, ptcl, vel_bound, correction.mean_free, )
    return disp_delta, vel_delta


def evaluate_phase_space_residual_fused(correction, a, ptcl, cosmo, conf, drift_scale, local_pair=None, context=None, ):
    """Evaluate both phase-space heads with one stacked reduction.

    The separate helper remains available for regression comparisons.  This
    candidate shares the authoritative count, mean, and peak-norm reduction for
    displacement and velocity while preserving the exact mean-free and bound
    semantics.
    """
    if correction is None:
        zeros = jnp.zeros_like(ptcl.disp)
        return zeros, zeros
    if not isinstance(correction, BoundedPhaseSpaceCorrection):
        raise TypeError("phase_space must be a BoundedPhaseSpaceCorrection.")
    head_context = local_pair if context is None else context
    raw_disp, raw_vel = correction.apply_fn(correction.params, a, ptcl, cosmo, conf, head_context)
    raw = jnp.stack([jnp.asarray(raw_disp, dtype=ptcl.disp.dtype), jnp.asarray(raw_vel, dtype=ptcl.disp.dtype)], axis=0)
    if raw.shape[1:] != ptcl.disp.shape:
        raise ValueError("phase-space residual heads must match particle displacement shape")
    active, authoritative = _particle_masks(ptcl)
    vectors = jnp.tanh(raw)
    particle_axes = tuple(range(1, vectors.ndim - 1))
    count = jnp.maximum(jnp.sum(authoritative.astype(vectors.dtype)), jnp.asarray(1, dtype=vectors.dtype))
    if correction.mean_free:
        mean = jnp.sum(vectors * authoritative.astype(vectors.dtype)[None, ..., None], axis=particle_axes) / count
        mean_shape = (2, ) + (1, ) * (vectors.ndim - 2) + (vectors.shape[-1], )
        vectors = vectors - mean.reshape(mean_shape)
    norm_sq = jnp.sum(jnp.square(jnp.where(active[None, ..., None], vectors, 0)), axis=-1)
    peak_sq = jnp.max(norm_sq, axis=tuple(range(1, norm_sq.ndim)), initial=jnp.asarray(0, dtype=norm_sq.dtype))
    drift_scale = jnp.asarray(drift_scale, dtype=ptcl.disp.dtype)
    tiny = jnp.asarray(jnp.finfo(ptcl.disp.dtype).tiny, dtype=ptcl.disp.dtype)
    bounds = jnp.asarray([
        correction.max_displacement_cells * conf.ptcl_spacing,
        correction.max_velocity_cells * conf.ptcl_spacing / jnp.maximum(jnp.abs(drift_scale), tiny),
    ], dtype=vectors.dtype,
                         )
    scale = jnp.reciprocal(jnp.sqrt(jnp.maximum(peak_sq, jnp.asarray(1, dtype=peak_sq.dtype))))
    output = vectors * scale.reshape((2, ) + (1, ) * (vectors.ndim - 2) +
                                     (1, )) * bounds.reshape((2, ) + (1, ) * (vectors.ndim - 2) + (1, ))
    output = jnp.where(active[None, ..., None], output, 0)
    return output[0], output[1]


def apply_phase_space_correction(correction, a, ptcl, cosmo, conf, drift_scale, local_pair=None, context=None, ):
    """Apply one bounded direct residual to particle displacement and velocity."""
    if correction is None:
        return ptcl
    disp_delta, vel_delta = evaluate_phase_space_residual(
        correction, a, ptcl, cosmo, conf, drift_scale, local_pair=local_pair, context=context,
    )
    return ptcl.replace(disp=ptcl.disp + disp_delta, vel=ptcl.vel + vel_delta)


def zero_nbody_correction_cotangent(correction):
    """Return a zero cotangent with the structure of any correction composite."""
    if correction is None:
        return None
    return jax.tree_util.tree_map(
        lambda value: value if value is None or is_float0_array(value) else jnp.zeros_like(value), correction,
    )


def add_nbody_correction_cotangents(lhs, rhs):
    """Add correction cotangents while preserving absent and float0 leaves."""
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

    return jax.tree_util.tree_map(add_leaf, lhs, rhs, is_leaf=lambda value: value is None)
