"""Compatibility facade for PM++ potential corrections.

New correction implementations live under :mod:`src.corrections`. This module is
kept so existing scripts/tests that import ``src.potential_correction`` continue
to work.
"""

from .corrections import (
    CombinedPotentialCorrection,
    MeshCNNPotentialCorrection,
    MeshResidualPotentialCorrection,
    NeuralSplineFourierFilter,
    PMWindowCompensationCorrection,
    RadialPotentialCorrection,
    add_potential_correction_cotangents,
    apply_potential_correction,
    build_correction_optimizer,
    evaluate_mesh_potential_residual,
    evaluate_mesh_source_residual,
    evaluate_pm_window_compensation,
    evaluate_potential_transfer,
    evaluate_radial_potential_transfer,
    force_green_kernel,
    force_uses_interlacing,
    init_mesh_cnn_potential_correction,
    init_pm_window_compensation_correction,
    init_potential_correction,
    init_radial_potential_correction,
    resolve_sigma8,
    sample_potential_transfer,
    zero_potential_correction_cotangent,
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
