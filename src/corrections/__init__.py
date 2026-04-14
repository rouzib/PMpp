"""Potential correction models and dispatch helpers."""

from .combined import CombinedPotentialCorrection
from .common import build_correction_optimizer, resolve_sigma8
from .core import (
    add_potential_correction_cotangents,
    apply_potential_correction,
    evaluate_mesh_potential_residual,
    evaluate_mesh_source_residual,
    evaluate_potential_transfer,
    evaluate_radial_potential_transfer,
    force_green_kernel,
    force_uses_interlacing,
    init_potential_correction,
    sample_potential_transfer,
    zero_potential_correction_cotangent,
)
from .mesh_cnn import (
    MeshCNNPotentialCorrection,
    MeshResidualPotentialCorrection,
    init_mesh_cnn_potential_correction,
)
from .radial import (
    NeuralSplineFourierFilter,
    RadialPotentialCorrection,
    init_radial_potential_correction,
)
from .window import (
    PMWindowCompensationCorrection,
    evaluate_pm_window_compensation,
    init_pm_window_compensation_correction,
)

__all__ = [
    "CombinedPotentialCorrection",
    "MeshCNNPotentialCorrection",
    "MeshResidualPotentialCorrection",
    "NeuralSplineFourierFilter",
    "PMWindowCompensationCorrection",
    "RadialPotentialCorrection",
    "add_potential_correction_cotangents",
    "apply_potential_correction",
    "build_correction_optimizer",
    "evaluate_mesh_potential_residual",
    "evaluate_mesh_source_residual",
    "evaluate_pm_window_compensation",
    "evaluate_potential_transfer",
    "evaluate_radial_potential_transfer",
    "force_green_kernel",
    "force_uses_interlacing",
    "init_mesh_cnn_potential_correction",
    "init_pm_window_compensation_correction",
    "init_potential_correction",
    "init_radial_potential_correction",
    "resolve_sigma8",
    "sample_potential_transfer",
    "zero_potential_correction_cotangent",
]
