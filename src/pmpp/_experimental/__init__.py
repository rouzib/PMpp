"""Opt-in optimization candidates used by PM++ development benchmarks.

Nothing in this package changes PM++ automatic production selection.  The
objects here are deliberately small, reference-preserving implementations that
can be selected by benchmark workers and compared against the current solver.
"""

from .cic import (
    CICPlan,
    cic_gather_binned,
    cic_gather_reference,
    cic_scatter_binned,
    cic_scatter_reference,
    make_cic_plan,
)
from .local_pair import (
    radial_shell_average,
    shell_local_pair_convolution,
    dense_local_pair_convolution,
    ratio_two_coarse_deposit,
    ratio_two_coarse_gather,
    shell_layout,
)
from .policy import OptimizationPolicy, OptimizationStatus, resolve_policy
from .reductions import fused_phase_space_reductions, fused_particle_statistics
from .tuning import CIC_TILE_CANDIDATES, STATIC_CIC_TILE_TABLE, select_cic_tile
from .routing import (
    BidirRouteResult,
    RouteMessage,
    compact_stay_descriptors,
    merge_path_route,
    route_pack_bidir,
    transpose_route_cotangent,
)

__all__ = [
    "BidirRouteResult",
    "CIC_TILE_CANDIDATES",
    "CICPlan",
    "OptimizationPolicy",
    "OptimizationStatus",
    "RouteMessage",
    "compact_stay_descriptors",
    "cic_gather_binned",
    "cic_gather_reference",
    "cic_scatter_binned",
    "cic_scatter_reference",
    "dense_local_pair_convolution",
    "fused_particle_statistics",
    "fused_phase_space_reductions",
    "make_cic_plan",
    "merge_path_route",
    "radial_shell_average",
    "ratio_two_coarse_deposit",
    "ratio_two_coarse_gather",
    "resolve_policy",
    "route_pack_bidir",
    "shell_local_pair_convolution",
    "shell_layout",
    "transpose_route_cotangent",
    "STATIC_CIC_TILE_TABLE",
    "select_cic_tile",
]
