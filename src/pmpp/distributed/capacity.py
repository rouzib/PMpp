"""Host-side capacity sizing from successful simulation telemetry."""
import math
import numpy as np


def routing_capacity_report(owned_counts, migration_counts, *, headroom=0.1, overflowed=False):
    """Recommend static capacities from observed per-step/per-shard counts.

    ``migration_counts`` contains uncapped outgoing counts for each direction.
    Both inputs must cover the intended workload, including its densest steps.
    The report is a calibration record, not a bound for different initial
    conditions. No Configuration defaults or overflow checks are changed.
    """
    if overflowed:
        raise ValueError("Cannot calibrate capacities from an overflowed or truncated run")
    if not math.isfinite(headroom) or headroom <= 0:
        raise ValueError("headroom must be positive and finite")
    peaks = []
    for name, counts in (("owned_counts", owned_counts), ("migration_counts", migration_counts)):
        values = np.asarray(counts)
        if values.size == 0 or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"{name} must contain nonnegative integer observations")
        peaks.append(int(values.max()))
    if peaks[0] == 0:
        raise ValueError("At least one owned particle must have been observed")
    capacities = [max(1, math.ceil(peak * (1 + headroom))) for peak in peaks]
    if max(capacities) > 2**31 - 1:
        raise ValueError("Recommended per-device capacity exceeds signed int32 indexing")
    return {
        "observed_max_owned": peaks[0],
        "observed_max_migration": peaks[1],
        "headroom": headroom,
        "owned_observations": int(np.size(owned_counts)),
        "migration_observations": int(np.size(migration_counts)),
        "max_ptcl_per_slice": capacities[0],
        "max_share_ptcl": capacities[1]
    }
