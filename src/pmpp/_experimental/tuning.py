"""Offline-only static CIC tile selection."""

from __future__ import annotations

import statistics


CIC_TILE_CANDIDATES = (128, 256, 512)
STATIC_CIC_TILE_TABLE = {
    "default": 128,
    "scalar_scatter": 128,
    "three_channel_gather": 128,
    "scalar_scatter_vjp": 128,
    "three_channel_gather_vjp": 128,
}


def select_cic_tile(operation: str = "default", *, requested: int | None = None, measured: dict[int, list[float]] | None = None) -> int:
    """Select a tile without measuring during normal simulation execution.

    When confidence is tied, the simpler 128-particle tile wins. ``measured``
    is intended for an offline benchmark worker, not a solver call.
    """

    if requested is not None:
        if requested not in CIC_TILE_CANDIDATES:
            raise ValueError(f"requested CIC tile must be one of {CIC_TILE_CANDIDATES}")
        return int(requested)
    if not measured:
        return int(STATIC_CIC_TILE_TABLE.get(operation, STATIC_CIC_TILE_TABLE["default"]))
    medians = {int(tile): statistics.median(float(value) for value in samples) for tile, samples in measured.items() if samples}
    if not medians:
        return int(STATIC_CIC_TILE_TABLE.get(operation, STATIC_CIC_TILE_TABLE["default"]))
    best = min(medians.values())
    baseline = medians.get(128, best)
    # Without paired confidence intervals available here, treat a sub-1%
    # improvement as a tie and retain 128.
    if baseline <= best * 1.01:
        return 128
    return min(medians, key=medians.get)


__all__ = ["CIC_TILE_CANDIDATES", "STATIC_CIC_TILE_TABLE", "select_cic_tile"]
