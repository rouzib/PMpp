"""Optional capacity telemetry and sweep helpers.

Telemetry is host-side and opt-in.  It records counts but never clips arrays or
replaces PM++ synchronized capacity checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CAPACITY_FIELDS = (
    "authoritative_count",
    "left_mover_count",
    "right_mover_count",
    "halo_copy_count",
    "gather_share_count",
)


@dataclass
class CapacityTelemetry:
    """Per-step, per-device maximum utilization record."""

    enabled: bool = False
    records: list[dict[str, Any]] = field(default_factory=list)
    maxima: dict[str, int] = field(default_factory=dict)
    first_maximum_step: dict[str, int] = field(default_factory=dict)

    def record(self, *, step: int, device: int, capacities: dict[str, int], limits: dict[str, int] | None = None):
        if not self.enabled:
            return
        capacities = {name: int(value) for name, value in capacities.items()}
        entry: dict[str, Any] = {"step": int(step), "device": int(device), **capacities}
        if limits:
            entry["utilization"] = {
                name: (value / limits[name] if limits.get(name, 0) else None)
                for name, value in capacities.items()
            }
        self.records.append(entry)
        for name, value in capacities.items():
            if value > self.maxima.get(name, -1):
                self.maxima[name] = value
                self.first_maximum_step[name] = int(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "records": self.records,
            "maxima": self.maxima,
            "first_maximum_step": self.first_maximum_step,
        }


def capacity_sweep_factors() -> tuple[float, ...]:
    return (1.05, 1.10, 1.20, 1.30, 1.50)


def scale_capacity(capacity: int, factor: float) -> int:
    if capacity < 0 or factor <= 0:
        raise ValueError("capacity must be non-negative and factor must be positive")
    return max(1, int(capacity * factor + 0.999999))


__all__ = ["CAPACITY_FIELDS", "CapacityTelemetry", "capacity_sweep_factors", "scale_capacity"]
