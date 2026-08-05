"""Reference-preserving bidirectional routing and merge-path semantics.

The implementation is written in JAX so it can be used in CPU logical-device
tests and in shard-local GPU diagnostics.  It intentionally keeps fixed-size
buffers and reports the uncapped count; a production CUDA implementation can
use the same record/provenance contract without changing the solver's
canonical particle order.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


INVALID = 0
STAY = 1
SEND_LEFT = 2
SEND_RIGHT = 3
OUT_OF_RANGE = 4
RECORD_FORMAT_VERSION = 2


class RouteMessage(NamedTuple):
    """Fixed-capacity self-describing route message."""

    key: jax.Array
    pmid: jax.Array
    disp: jax.Array
    vel: jax.Array
    valid: jax.Array
    count: jax.Array
    source_index: jax.Array
    source_tag: jax.Array
    version: jax.Array
    auxiliary: jax.Array | None = None


class BidirRouteResult(NamedTuple):
    """Output of one-pass local classification and two stream packing."""

    left: RouteMessage
    right: RouteMessage
    left_count: jax.Array
    right_count: jax.Array
    classification: jax.Array
    stay_key: jax.Array
    stay_source_index: jax.Array
    stay_count: jax.Array


class MergePathResult(NamedTuple):
    """Stable three-stream merge with source provenance."""

    pmid: jax.Array
    disp: jax.Array
    vel: jax.Array
    valid: jax.Array
    key: jax.Array
    source_tag: jax.Array
    source_index: jax.Array
    count: jax.Array


def _interval_mask(value, start: int, end: int, period: int):
    start %= period
    end %= period
    if start < end:
        return (value >= start) & (value < end)
    if start > end:
        return (value >= start) | (value < end)
    return jnp.zeros_like(value, dtype=jnp.bool_)


def _raveled_pmid(pmid, mesh_shape):
    pmid = jnp.asarray(pmid)
    shape = tuple(int(value) for value in mesh_shape)
    wrapped = jnp.mod(pmid.astype(jnp.int32), jnp.asarray(shape, dtype=jnp.int32))
    return (wrapped[:, 0] * shape[1] + wrapped[:, 1]) * shape[2] + wrapped[:, 2]


def _message_from_mask(
    pmid,
    disp,
    vel,
    valid,
    mask,
    *,
    mesh_shape,
    capacity: int,
    source_tag: int,
    auxiliary=None,
):
    indices = jnp.nonzero(mask, size=capacity, fill_value=0)[0].astype(jnp.int32)
    full_count = jnp.sum(mask.astype(jnp.int32))
    kept = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(full_count, capacity)
    keys_all = _raveled_pmid(pmid, mesh_shape)
    key = jnp.where(kept, keys_all[indices], 0)
    out_pmid = jnp.where(kept[:, None], pmid[indices], jnp.zeros_like(pmid[indices]))
    out_disp = jnp.where(kept[:, None], disp[indices], jnp.zeros_like(disp[indices]))
    out_vel = jnp.where(kept[:, None], vel[indices], jnp.zeros_like(vel[indices]))
    out_valid = kept & valid[indices]
    out_source = jnp.where(kept, indices, -jnp.ones_like(indices))
    out_tag = jnp.full((capacity,), source_tag, dtype=jnp.uint8)
    out_tag = jnp.where(out_valid, out_tag, jnp.zeros_like(out_tag))
    out_aux = None if auxiliary is None else jnp.where(
        kept[:, None], auxiliary[indices], jnp.zeros_like(auxiliary[indices])
    )
    return RouteMessage(
        key=key,
        pmid=out_pmid,
        disp=out_disp,
        vel=out_vel,
        valid=out_valid,
        count=full_count,
        source_index=out_source,
        source_tag=out_tag,
        version=jnp.asarray(RECORD_FORMAT_VERSION, dtype=jnp.int32),
        auxiliary=out_aux,
    )


def compact_stay_descriptors(pmid, classification, *, mesh_shape, capacity: int):
    """Return compact ``(raveled key, original slot)`` stay descriptors."""

    mask = jnp.asarray(classification) == STAY
    indices = jnp.nonzero(mask, size=capacity, fill_value=0)[0].astype(jnp.int32)
    count = jnp.sum(mask.astype(jnp.int32))
    kept = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(count, capacity)
    keys = _raveled_pmid(pmid, mesh_shape)[indices]
    return (
        jnp.where(kept, keys, 0),
        jnp.where(kept, indices, -jnp.ones_like(indices)),
        count,
    )


def route_pack_bidir(
    pmid,
    disp,
    vel,
    valid,
    x_mod,
    *,
    global_nmesh: int,
    mesh_shape: tuple[int, int, int],
    owned_start: int,
    owned_end: int,
    slice_width: int,
    num_devices: int,
    capacity: int,
    auxiliary=None,
) -> BidirRouteResult:
    """Classify every authoritative slot once and pack both directions.

    The two-device special case suppresses the right-going export, matching
    PM++'s existing ring convention.  Stable input order is retained by the
    fixed-size ``nonzero`` compaction.
    """

    pmid = jnp.asarray(pmid)
    disp = jnp.asarray(disp)
    vel = jnp.asarray(vel)
    valid = jnp.asarray(valid, dtype=jnp.bool_)
    x_mod = jnp.mod(jnp.asarray(x_mod), global_nmesh)
    n = pmid.shape[0]
    if num_devices == 1:
        stay = valid
        left = jnp.zeros((n,), dtype=jnp.bool_)
        right = left
        out_of_range = jnp.zeros((n,), dtype=jnp.bool_)
    else:
        stay = valid & _interval_mask(x_mod, owned_start, owned_end, global_nmesh)
        left_start = (owned_start - slice_width) % global_nmesh
        right_end = (owned_end + slice_width) % global_nmesh
        left = valid & ~stay & _interval_mask(x_mod, left_start, owned_start, global_nmesh)
        right = (
            valid
            & ~stay
            & (num_devices != 2)
            & _interval_mask(x_mod, owned_end, right_end, global_nmesh)
        )
        out_of_range = valid & ~(stay | left | right)
    classification = jnp.where(
        ~valid,
        INVALID,
        jnp.where(stay, STAY, jnp.where(left, SEND_LEFT, jnp.where(right, SEND_RIGHT, OUT_OF_RANGE))),
    ).astype(jnp.uint8)
    left_message = _message_from_mask(
        pmid,
        disp,
        vel,
        valid,
        left,
        mesh_shape=mesh_shape,
        capacity=capacity,
        source_tag=1,
        auxiliary=auxiliary,
    )
    right_message = _message_from_mask(
        pmid,
        disp,
        vel,
        valid,
        right,
        mesh_shape=mesh_shape,
        capacity=capacity,
        source_tag=2,
        auxiliary=auxiliary,
    )
    stay_key, stay_source_index, stay_count = compact_stay_descriptors(
        pmid, classification, mesh_shape=mesh_shape, capacity=capacity
    )
    del out_of_range  # retained in classification for synchronized error handling.
    return BidirRouteResult(
        left_message,
        right_message,
        left_message.count,
        right_message.count,
        classification,
        stay_key,
        stay_source_index,
        stay_count,
    )


def merge_path_route(
    pmid,
    disp,
    vel,
    valid,
    classification,
    left: RouteMessage,
    right: RouteMessage,
    *,
    mesh_shape: tuple[int, int, int],
    capacity: int,
) -> MergePathResult:
    """Stable-merge stay, left-incoming, and right-incoming streams.

    The composite ordering is ``(raveled_pmid, source_tag)`` with source tags
    ``stay=0, left=1, right=2``.  All candidates are materialized into a
    bounded logical tile in this reference implementation; the CUDA candidate
    can replace this function with CTA co-rank partitions without changing the
    result or provenance contract.
    """

    stay_key, stay_index, stay_count = compact_stay_descriptors(
        pmid, classification, mesh_shape=mesh_shape, capacity=capacity
    )
    stay_valid = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(stay_count, capacity)
    local_pmid = pmid[stay_index.clip(0)]
    local_disp = disp[stay_index.clip(0)]
    local_vel = vel[stay_index.clip(0)]

    all_key = jnp.concatenate([stay_key, left.key, right.key], axis=0)
    all_pmid = jnp.concatenate([local_pmid, left.pmid, right.pmid], axis=0)
    all_disp = jnp.concatenate([local_disp, left.disp, right.disp], axis=0)
    all_vel = jnp.concatenate([local_vel, left.vel, right.vel], axis=0)
    all_valid = jnp.concatenate([stay_valid, left.valid, right.valid], axis=0)
    all_tag = jnp.concatenate(
        [jnp.zeros((capacity,), jnp.uint8), jnp.ones((capacity,), jnp.uint8), jnp.full((capacity,), 2, jnp.uint8)]
    )
    all_source = jnp.concatenate([stay_index, left.source_index, right.source_index], axis=0)
    # A finite sentinel is enough because valid mesh keys are below product(mesh_shape).
    sentinel = math.prod(mesh_shape) * 4 + 3
    sort_key = jnp.where(all_valid, all_key.astype(jnp.int32) * 4 + all_tag.astype(jnp.int32), sentinel)
    order = jnp.argsort(sort_key, stable=True)[:capacity]
    out_valid = all_valid[order]
    out_key = jnp.where(out_valid, all_key[order], 0)
    out_pmid = jnp.where(out_valid[:, None], all_pmid[order], jnp.zeros_like(all_pmid[order]))
    out_disp = jnp.where(out_valid[:, None], all_disp[order], jnp.zeros_like(all_disp[order]))
    out_vel = jnp.where(out_valid[:, None], all_vel[order], jnp.zeros_like(all_vel[order]))
    out_tag = jnp.where(out_valid, all_tag[order], 0).astype(jnp.uint8)
    out_source = jnp.where(out_valid, all_source[order], -1).astype(jnp.int32)
    count = jnp.sum(all_valid.astype(jnp.int32))
    return MergePathResult(out_pmid, out_disp, out_vel, out_valid, out_key, out_tag, out_source, count)


def transpose_route_cotangent(
    output_cotangent,
    source_tag,
    source_index,
    *,
    authoritative_capacity: int,
    left_capacity: int,
    right_capacity: int,
    valid=None,
):
    """Transpose a merged cotangent using the stable provenance arrays."""

    cot = jnp.asarray(output_cotangent)
    if valid is None:
        valid = source_index >= 0
    zero_auth = jnp.zeros((authoritative_capacity,) + cot.shape[1:], dtype=cot.dtype)
    zero_left = jnp.zeros((left_capacity,) + cot.shape[1:], dtype=cot.dtype)
    zero_right = jnp.zeros((right_capacity,) + cot.shape[1:], dtype=cot.dtype)
    auth = zero_auth.at[jnp.where((source_tag == 0) & valid, source_index, 0)].add(
        jnp.where(((source_tag == 0) & valid)[(...,) + (None,) * (cot.ndim - 1)], cot, 0)
    )
    left = zero_left.at[jnp.where((source_tag == 1) & valid, source_index, 0)].add(
        jnp.where(((source_tag == 1) & valid)[(...,) + (None,) * (cot.ndim - 1)], cot, 0)
    )
    right = zero_right.at[jnp.where((source_tag == 2) & valid, source_index, 0)].add(
        jnp.where(((source_tag == 2) & valid)[(...,) + (None,) * (cot.ndim - 1)], cot, 0)
    )
    return auth, left, right


__all__ = [
    "BidirRouteResult",
    "MergePathResult",
    "RouteMessage",
    "compact_stay_descriptors",
    "merge_path_route",
    "route_pack_bidir",
    "transpose_route_cotangent",
]
