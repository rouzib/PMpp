"""Public correction dispatch and application helpers."""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from ..utils import is_float0_array
from .combined import CombinedPotentialCorrection
from .mesh_cnn import (
    MeshCNNPotentialCorrection,
    evaluate_mesh_potential_residual as _evaluate_mesh_potential_residual,
    evaluate_mesh_source_residual as _evaluate_mesh_source_residual,
    init_mesh_cnn_potential_correction,
)
from .radial import (
    RadialPotentialCorrection,
    evaluate_radial_potential_transfer as _evaluate_radial_potential_transfer,
    init_radial_potential_correction,
    sample_radial_potential_transfer,
)
from .window import (
    PMWindowCompensationCorrection,
    evaluate_pm_window_compensation,
    init_pm_window_compensation_correction,
)


def init_potential_correction(key, model="neural_spline", **kwargs):
    """Construct a potential-correction pytree from a small model name.

    This factory is the public entry point used by scripts and tests. It keeps
    the rest of the solver from depending on the concrete correction classes.
    """
    if model in {"neural_spline", "radial_spline", "radial", "radial_mlp"}:
        return init_radial_potential_correction(key, **kwargs)
    if model in {"mesh_cnn", "cnn"}:
        return init_mesh_cnn_potential_correction(key, **kwargs)
    if model in {"combined", "hybrid", "spline_cnn", "neural_spline+mesh_cnn"}:
        radial_key, cnn_key = jax.random.split(key)
        radial = init_radial_potential_correction(radial_key, **kwargs)
        mesh_cnn = init_mesh_cnn_potential_correction(cnn_key, **kwargs)
        dtype = kwargs.get("dtype", getattr(radial, "dtype", jnp.float32))
        return CombinedPotentialCorrection(
            radial=radial,
            mesh_cnn=mesh_cnn,
            dtype=dtype,
        )
    if model in {"pm_window", "cic_compensation", "cic_window_compensation"}:
        correction_kwargs = dict(kwargs)
        dtype = correction_kwargs.pop("dtype", jnp.float32)
        return init_pm_window_compensation_correction(dtype=dtype, **correction_kwargs)
    raise ValueError(f"Unsupported correction model {model!r}.")


def sample_potential_transfer(correction, radius_fraction, a, cosmo, conf):
    """Sample the radial Fourier transfer curve for plotting or diagnostics."""
    if isinstance(correction, CombinedPotentialCorrection):
        correction = correction.radial
    if isinstance(correction, MeshCNNPotentialCorrection):
        raise TypeError("Mesh CNN corrections do not define a radial transfer curve.")
    if isinstance(correction, PMWindowCompensationCorrection):
        raise TypeError("PM window compensation is anisotropic; use evaluate_pm_window_compensation instead.")
    if correction is not None and not isinstance(correction, RadialPotentialCorrection):
        raise TypeError(f"Unsupported correction type {type(correction).__name__!s}.")
    return sample_radial_potential_transfer(correction, radius_fraction, a, cosmo, conf)


def evaluate_radial_potential_transfer(correction, a, cosmo, conf):
    """Evaluate the multiplicative Fourier transfer field for radial corrections."""
    if isinstance(correction, CombinedPotentialCorrection):
        correction = correction.radial
    if isinstance(correction, MeshCNNPotentialCorrection):
        raise TypeError("Mesh CNN corrections do not define a radial transfer field.")
    if isinstance(correction, PMWindowCompensationCorrection):
        return evaluate_pm_window_compensation(correction, conf)
    if correction is not None and not isinstance(correction, RadialPotentialCorrection):
        raise TypeError(f"Unsupported correction type {type(correction).__name__!s}.")
    return _evaluate_radial_potential_transfer(correction, a, cosmo, conf)


def evaluate_potential_transfer(correction, a, cosmo, conf):
    """Compatibility alias for callers that predate the correction refactor."""
    return evaluate_radial_potential_transfer(correction, a, cosmo, conf)


def evaluate_mesh_potential_residual(correction, source_real, potential_real, a, cosmo, conf):
    """Evaluate the real-space residual potential predicted by a mesh CNN."""
    if isinstance(correction, CombinedPotentialCorrection):
        correction = correction.mesh_cnn
    return _evaluate_mesh_potential_residual(correction, source_real, potential_real, a, cosmo, conf)


def evaluate_mesh_source_residual(correction, source_real, a, cosmo, conf):
    """Evaluate a mesh CNN residual sourced directly from the density field."""
    if isinstance(correction, CombinedPotentialCorrection):
        correction = correction.mesh_cnn
    return _evaluate_mesh_source_residual(correction, source_real, a, cosmo, conf)


def apply_potential_correction(pot, a, cosmo, conf, correction, source_real=None):
    """Apply any supported correction to a Fourier-space PM potential.

    Radial and PM-window corrections are multiplicative transfer functions in
    Fourier space. Mesh CNN corrections operate in real space, so this helper
    transforms the current potential to real space, predicts a residual
    potential, transforms the residual back, and adds it to ``pot``. Combined
    corrections apply the radial part first and the mesh residual second.
    """
    if correction is None:
        return pot
    if isinstance(correction, CombinedPotentialCorrection):
        pot = apply_potential_correction(pot, a, cosmo, conf, correction.radial, source_real=source_real)
        return apply_potential_correction(pot, a, cosmo, conf, correction.mesh_cnn, source_real=source_real)
    if isinstance(correction, MeshCNNPotentialCorrection):
        if source_real is None:
            raise ValueError("source_real is required for mesh CNN potential corrections.")
        if conf.compute_mesh is None:
            potential_real = jnp.fft.irfftn(pot).astype(conf.float_dtype)
        else:
            potential_real = conf.mGPU_irfftn_transposed(pot).astype(conf.float_dtype)
        residual_potential = evaluate_mesh_potential_residual(
            correction,
            source_real,
            potential_real,
            1.0 if a is None else a,
            cosmo,
            conf,
        )
        if conf.compute_mesh is None:
            residual_hat = jnp.fft.rfftn(residual_potential)
        else:
            residual_hat = conf.mGPU_rfftn_transposed(residual_potential)
        return pot + residual_hat.astype(pot.dtype)

    transfer = evaluate_potential_transfer(correction, 1.0 if a is None else a, cosmo, conf)
    return pot * transfer.astype(pot.real.dtype)


def zero_potential_correction_cotangent(correction):
    """Return a zero cotangent pytree with the same correction structure."""
    if correction is None:
        return None
    return tree_map(
        lambda x: x if x is None or is_float0_array(x) else jnp.zeros_like(x),
        correction,
    )


def add_potential_correction_cotangents(lhs, rhs):
    """Add correction cotangent pytrees while preserving ``None`` and float0 leaves."""
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


def force_uses_interlacing(correction):
    """Return whether gravity should use interlaced CIC assignment."""
    return bool(getattr(correction, "interlacing", False))


def force_green_kernel(correction):
    """Return the Poisson Green's-function selector requested by a correction."""
    return getattr(correction, "green_kernel", "continuum")
