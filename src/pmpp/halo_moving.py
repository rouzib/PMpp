"""Particle routing helpers for PM++ multi-GPU slab decompositions.

The N-body drift can move particles across x-slab boundaries. This module
keeps the static-capacity particle buffers canonical after such moves:

* build the authoritative owned-particle block for each device,
* exchange boundary particles when ``particle_halo`` needs duplicated slots,
* keep ``mesh_halo`` authoritative-only storage compact,
* provide explicit transposes used by the hand-written adjoint.

Most private helpers preserve a common invariant: valid particles are packed in
monotonic raveled-``pmid`` order, and invalid/padding entries carry a sentinel
key. That invariant is what lets gather/scatter gradient exchanges match
compact buffers by position instead of doing expensive per-step hash lookups.
"""

from functools import partial
import os

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .utils import AXIS_NAME, pmid_to_idx, raise_error
from .cuda_routing import enabled_for_configuration as cuda_routing_enabled
from .cuda_routing import requested_backend as cuda_routing_backend
from .cuda_routing import route_merge as cuda_route_merge
from .cuda_routing import route_merge_bidir_cuda
from .cuda_routing import route_pack as cuda_route_pack
from .cuda_routing import route_pack_bidir_cuda
from .cuda_routing import route_transpose_scatter as cuda_route_transpose_scatter
from .cuda_routing import route_transpose_split as cuda_route_transpose_split


# Debug-only oracle for the native bidirectional route. This is read at
# import time so it remains a static JIT decision. It must never be enabled
# for timing runs: it executes the canonical JAX route in addition to the FFI
# route and emits one line per shard per move.
_DEBUG_BIDIR_ROUTE = os.environ.get("PMPP_BIDIR_DEBUG_ROUTE", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def _use_bidir_cuda_routing(conf):
    return cuda_routing_enabled(conf) and cuda_routing_backend() == "bidir_mergepath"


@jax.jit
def particles_in_slice_mask(x_mod, slice_start, slice_end):
    """Return the wrapped x-slab membership mask for particle positions.

    Parameters
    ----------
    x_mod : jax.Array
        Particle x positions in wrapped mesh-cell units.
    slice_start, slice_end : int or jax.Array
        Inclusive/exclusive slab bounds in wrapped mesh-cell units.

    Returns
    -------
    jax.Array
        Boolean mask selecting particles inside the slab.
    """
    within_slice = (x_mod >= slice_start) & (x_mod < slice_end)
    across_boundary = (x_mod >= slice_start) | (x_mod < slice_end)
    return jnp.where(slice_start > slice_end, across_boundary, within_slice)


@jax.jit
def compute_halo_mask(x_mod, halo_start, halo_end, unused_indexes):
    """Return the mask of duplicated halo particles for the current slab.

    Parameters
    ----------
    x_mod : jax.Array
        Particle x positions in wrapped mesh-cell units.
    halo_start, halo_end : jax.Array
        Left and right halo-band bounds for the current slab.
    unused_indexes : jax.Array
        Boolean padding mask.

    Returns
    -------
    jax.Array
        Boolean mask selecting active halo-duplicate slots.
    """

    def slice_mask(start, end):
        """Return a periodic interval mask for wrapped x-coordinates.

        Parameters
        ----------
        start
            Inclusive lower bound of the periodic x-interval.
        end
            Exclusive upper bound of the periodic x-interval.
        """
        within_range = (x_mod >= start) & (x_mod < end)
        across_boundary = (x_mod >= start) | (x_mod < end)
        return jnp.where(start > end, across_boundary, within_range)

    mask_start = slice_mask(halo_start[0], halo_start[1])
    mask_end = slice_mask(halo_end[0], halo_end[1])
    return (mask_start | mask_end) & ~unused_indexes


def _key_fill_value(conf):
    """Sentinel raveled key that sorts after every real particle key."""
    return jnp.asarray(conf.mesh_size, dtype=jnp.int32)


def _owned_slice_bounds(global_nMesh, num_gpus, offsets):
    """Return the owned x-slab bounds for the current shard."""
    owned_start = offsets[jax.lax.axis_index(AXIS_NAME)]
    owned_end = (owned_start + global_nMesh // num_gpus) % global_nMesh
    return owned_start, owned_end


def _x_mod_from_disp(pmid, disp, global_nMesh, disp_size):
    """Particle x-position in mesh-cell units, wrapped into ``[0, nMesh)``."""
    return (pmid[:, 0] + disp[:, 0] * disp_size) % global_nMesh


def _capacity_check(count, capacity, message):
    """Raise a JAX-side error when a static-capacity buffer would overflow."""
    _ = jax.lax.cond(
        count > capacity,
        lambda _: raise_error(message, x=count, y=capacity),
        lambda _: None,
        operand=None,
    )


def _compact_sorted_indices(mask, capacity, error_message):
    """Compact valid indices while preserving the source order."""
    count = jnp.sum(mask)
    # Canonical routing checks every migration count with a synchronized pmax
    # before any per-shard compaction. Do not raise a per-shard host callback
    # here: an asynchronous failure on one participant can strand another in
    # the following communication collective. ``error_message`` is retained
    # for the shared helper's stable call signature.
    del error_message
    fill_index = jnp.asarray(mask.shape[0] - 1, dtype=jnp.int32)
    compact_idx = jnp.nonzero(mask, size=capacity, fill_value=fill_index)[0]
    valid = jnp.arange(capacity) < count
    return compact_idx, valid


def _gather_compacted(values, compact_idx, valid, fill_value):
    """Gather compacted payload entries and fill invalid tail slots."""
    gathered = values[compact_idx]
    valid_shape = (valid.shape[0],) + (1,) * (gathered.ndim - 1)
    return jnp.where(
        valid.reshape(valid_shape),
        gathered,
        jnp.asarray(fill_value, dtype=values.dtype),
    )


def _compact_sorted_particles(keys, pmid, disp, vel, acc, mask, capacity, key_fill, error_message):
    """Compact a sorted particle payload into a fixed-capacity buffer."""
    # Canonical callers compact from an already key-sorted authoritative
    # sequence, so a single ordered index extraction can feed every payload.
    compact_idx, valid = _compact_sorted_indices(mask, capacity, error_message)
    keys_compact = _gather_compacted(keys, compact_idx, valid, key_fill)
    pmid_compact = _gather_compacted(pmid, compact_idx, valid, 0)
    disp_compact = _gather_compacted(disp, compact_idx, valid, 0)
    vel_compact = _gather_compacted(vel, compact_idx, valid, 0)
    acc_compact = _gather_compacted(acc, compact_idx, valid, 0)
    return keys_compact, pmid_compact, disp_compact, vel_compact, acc_compact, valid


def _compact_sorted_particles_with_slots(keys, pmid, disp, vel, acc, mask, capacity, key_fill, error_message):
    """Compact particles and remember their original source slots."""
    compact_idx, valid = _compact_sorted_indices(mask, capacity, error_message)
    keys_compact = _gather_compacted(keys, compact_idx, valid, key_fill)
    pmid_compact = _gather_compacted(pmid, compact_idx, valid, 0)
    disp_compact = _gather_compacted(disp, compact_idx, valid, 0)
    vel_compact = _gather_compacted(vel, compact_idx, valid, 0)
    acc_compact = _gather_compacted(acc, compact_idx, valid, 0)
    slots = jnp.where(valid, compact_idx, jnp.asarray(-1, compact_idx.dtype))
    return keys_compact, pmid_compact, disp_compact, vel_compact, acc_compact, valid, slots


def _compact_sorted_particles_no_acc_with_slots(
    keys, pmid, disp, vel, mask, capacity, key_fill, error_message
):
    """Compact a no-acceleration particle stream and retain source slots."""
    compact_idx, valid = _compact_sorted_indices(mask, capacity, error_message)
    keys_compact = _gather_compacted(keys, compact_idx, valid, key_fill)
    pmid_compact = _gather_compacted(pmid, compact_idx, valid, 0)
    disp_compact = _gather_compacted(disp, compact_idx, valid, 0)
    vel_compact = _gather_compacted(vel, compact_idx, valid, 0)
    slots = jnp.where(valid, compact_idx, jnp.asarray(-1, compact_idx.dtype))
    return keys_compact, pmid_compact, disp_compact, vel_compact, valid, slots


def _exchange_compacted_particles_packed(compacted, perm, conf):
    """Exchange a compacted particle stream using two collective payloads.

    ``ppermute`` lowers independently for every pytree leaf.  Passing the
    canonical particle tuple directly therefore creates one communication
    launch for keys, pmid, every floating-point field, and validity.  Mesh-halo
    migration instead sends two dense arrays:

    * integer metadata: ``pmid`` plus one validity column;
    * floating-point payload: all carried vector fields concatenated together.

    Raveled keys are deterministic functions of ``pmid`` and are reconstructed
    on the receiver.  Keeping validity in the pmid dtype avoids mixing integer
    metadata with floating-point state while retaining a single metadata
    collective for arbitrary static capacities.
    """
    _keys, pmid, *payload_and_valid = compacted
    *payload, valid = payload_and_valid

    metadata = jnp.concatenate((pmid, valid[:, None].astype(pmid.dtype)), axis=-1)
    packed_payload = jnp.concatenate(payload, axis=-1)

    incoming_metadata = jax.lax.ppermute(metadata, axis_name=AXIS_NAME, perm=perm)
    incoming_payload = jax.lax.ppermute(packed_payload, axis_name=AXIS_NAME, perm=perm)

    pmid_width = pmid.shape[-1]
    incoming_pmid = incoming_metadata[:, :pmid_width].astype(pmid.dtype)
    incoming_valid = incoming_metadata[:, pmid_width] != 0
    incoming_keys = pmid_to_idx(incoming_pmid, conf)
    incoming_keys = jnp.where(incoming_valid, incoming_keys, _key_fill_value(conf))

    fields = []
    start = 0
    for original in payload:
        stop = start + original.shape[-1]
        fields.append(incoming_payload[:, start:stop].astype(original.dtype))
        start = stop
    return (incoming_keys, incoming_pmid, *fields, incoming_valid)


def _synchronized_capacity_check(count, capacity, message):
    """Fail before routing collectives when any shard exceeds a capacity.

    A host callback on only one shard can abort that participant while its
    neighbours are entering a later ``ppermute``.  Synchronize the count first
    so every shard observes and raises the same failure before any payload
    collective is issued.
    """

    global_count = jax.lax.pmax(count, axis_name=AXIS_NAME)
    _capacity_check(global_count, capacity, message)


def _synchronized_nonzero_check(count, message):
    """Raise a routing-domain error coherently across all mesh shards."""

    global_count = jax.lax.pmax(count, axis_name=AXIS_NAME)
    _ = jax.lax.cond(
        global_count > 0,
        lambda _: raise_error(message, x=global_count),
        lambda _: None,
        operand=None,
    )


def _exchange_compacted_particles(compacted, perm, conf):
    """Exchange migration payloads through the packed mesh-halo collective."""
    return _exchange_compacted_particles_packed(compacted, perm, conf)


def _linear_merge_plan_two(keys_a, valid_a, keys_b, valid_b, capacity, key_fill):
    """Build an exact stable merge plan while searching only the small stream.

    Stream ``a`` is the full authoritative/stay buffer and stream ``b`` is a
    compact migration buffer.  The legacy merge binary-searches ``b`` once for
    every slot in ``a`` and then scatters every particle field.  Here only
    ``b`` searches ``a``.  Its insertion slots mark the sparse holes in the
    output, and a prefix sum maps every other output slot to a contiguous
    element of ``a``.

    Equal keys retain the legacy stable order ``a`` then ``b``.
    """
    count_a = jnp.sum(valid_a)
    count_b = jnp.sum(valid_b)
    total = count_a + count_b
    keys_a_filled = jnp.where(valid_a, keys_a, key_fill)
    keys_b_filled = jnp.where(valid_b, keys_b, key_fill)

    b_source_idx = jnp.arange(keys_b.shape[0], dtype=jnp.int32)
    pos_b = (
        b_source_idx
        + jnp.searchsorted(keys_a_filled, keys_b_filled, side="right").astype(jnp.int32)
    )
    # Force padded migration entries out of bounds so they can never collide
    # with a valid insertion slot in scatter-set lowering.
    pos_b = jnp.where(valid_b, pos_b, jnp.asarray(capacity, pos_b.dtype))

    slots = jnp.arange(capacity, dtype=jnp.int32)
    source_code = jnp.zeros((capacity,), dtype=jnp.int32)
    source_code = source_code.at[pos_b].set(
        jnp.where(valid_b, jnp.int32(1), jnp.int32(0)), mode="drop"
    )
    b_idx = jnp.zeros((capacity,), dtype=jnp.int32)
    b_idx = b_idx.at[pos_b].set(b_source_idx, mode="drop")

    incoming_prefix = jnp.cumsum(source_code != 0, dtype=jnp.int32)
    a_idx = jnp.clip(slots - incoming_prefix, 0, keys_a.shape[0] - 1)
    out_valid = slots < total
    return source_code, a_idx, b_idx, out_valid, total


def _linear_take_two(values_a, values_b, source_code, a_idx, b_idx):
    """Materialize one payload field from a two-stream linear merge plan."""
    use_b = source_code == 1
    mask_shape = use_b.shape + (1,) * (values_a.ndim - 1)
    return jnp.where(use_b.reshape(mask_shape), values_b[b_idx], values_a[a_idx])


def _zero_invalid_merge_values(values, out_valid):
    """Zero padded output slots for a merged particle field."""
    mask_shape = out_valid.shape + (1,) * (values.ndim - 1)
    return jnp.where(out_valid.reshape(mask_shape), values, jnp.zeros_like(values))


def _sparse_stay_source_plan(
    valid_a,
    slots_left,
    valid_left,
    capacity,
    slots_right=None,
    valid_right=None,
):
    """Map compact stay ranks to original slots using only sparse holes.

    The authoritative input is valid in a contiguous prefix.  Migrating
    particles are sparse holes in that prefix.  If the sorted outgoing slots
    are ``h[k]``, then ``h[k] - k`` is the compact-stay rank at which that hole
    must be skipped.  Scattering those sparse skip counts followed by one
    prefix sum gives the inverse stable-compaction map without a full-capacity
    ``jnp.nonzero(stay_mask, size=capacity)``.
    """
    slots = jnp.arange(capacity, dtype=jnp.int32)
    count_left = jnp.sum(valid_left)
    left_index = jnp.arange(slots_left.shape[0], dtype=jnp.int32)
    left_filled = jnp.where(valid_left, slots_left, jnp.int32(capacity))

    hole_counts = jnp.zeros((capacity,), dtype=jnp.int32)
    if slots_right is None:
        left_rank = left_index
        count_right = jnp.int32(0)
    else:
        right_index = jnp.arange(slots_right.shape[0], dtype=jnp.int32)
        right_filled = jnp.where(valid_right, slots_right, jnp.int32(capacity))
        left_rank = left_index + jnp.searchsorted(
            right_filled, left_filled, side="left"
        ).astype(jnp.int32)
        right_rank = right_index + jnp.searchsorted(
            left_filled, right_filled, side="left"
        ).astype(jnp.int32)
        right_threshold = slots_right - right_rank
        right_threshold = jnp.where(valid_right, right_threshold, jnp.int32(capacity))
        hole_counts = hole_counts.at[right_threshold].add(
            valid_right.astype(jnp.int32), mode="drop"
        )
        count_right = jnp.sum(valid_right)

    left_threshold = slots_left - left_rank
    left_threshold = jnp.where(valid_left, left_threshold, jnp.int32(capacity))
    hole_counts = hole_counts.at[left_threshold].add(
        valid_left.astype(jnp.int32), mode="drop"
    )

    skipped = jnp.cumsum(hole_counts, dtype=jnp.int32)
    stay_count = jnp.sum(valid_a) - count_left - count_right
    stay_valid = slots < stay_count
    stay_pos = jnp.where(
        stay_valid,
        jnp.clip(slots + skipped, 0, valid_a.shape[0] - 1),
        jnp.int32(-1),
    )
    return skipped, stay_pos, stay_valid, stay_count


def _sparse_route_plan_two(
    keys_a,
    valid_a,
    outgoing_keys,
    outgoing_valid,
    outgoing_slots,
    keys_b,
    valid_b,
    capacity,
    key_fill,
):
    """Build a stable route plan without materializing the compact stay stream."""
    skipped, stay_pos, stay_valid, count_stay = _sparse_stay_source_plan(
        valid_a, outgoing_slots, outgoing_valid, capacity
    )
    keys_a_filled = jnp.where(valid_a, keys_a, key_fill)
    outgoing_filled = jnp.where(outgoing_valid, outgoing_keys, key_fill)
    keys_b_filled = jnp.where(valid_b, keys_b, key_fill)

    b_source_idx = jnp.arange(keys_b.shape[0], dtype=jnp.int32)
    rank_stay_b = (
        jnp.searchsorted(keys_a_filled, keys_b_filled, side="right")
        - jnp.searchsorted(outgoing_filled, keys_b_filled, side="right")
    ).astype(jnp.int32)
    pos_b = b_source_idx + rank_stay_b
    pos_b = jnp.where(valid_b, pos_b, jnp.int32(capacity))

    slots = jnp.arange(capacity, dtype=jnp.int32)
    source_code = jnp.zeros((capacity,), dtype=jnp.bool_)
    source_code = source_code.at[pos_b].set(
        valid_b, mode="drop"
    )
    incoming_prefix = jnp.cumsum(source_code, dtype=jnp.int32)
    b_idx = jnp.clip(incoming_prefix - 1, 0, keys_b.shape[0] - 1)
    stay_rank = jnp.clip(slots - incoming_prefix, 0, capacity - 1)
    a_idx = jnp.clip(
        stay_rank + skipped[stay_rank], 0, valid_a.shape[0] - 1
    )
    total = count_stay + jnp.sum(valid_b)
    out_valid = slots < total
    return source_code, a_idx, b_idx, stay_rank, out_valid, total, stay_pos, stay_valid


def _sparse_route_plan_one_incoming_two_outgoing(
    keys_a,
    valid_a,
    outgoing_left_keys,
    outgoing_left_valid,
    outgoing_left_slots,
    outgoing_right_keys,
    outgoing_right_valid,
    outgoing_right_slots,
    incoming_keys,
    incoming_valid,
    capacity,
    key_fill,
):
    """Build a full route plan after first merging the two small incoming streams."""
    skipped, stay_pos, stay_valid, count_stay = _sparse_stay_source_plan(
        valid_a,
        outgoing_left_slots,
        outgoing_left_valid,
        capacity,
        outgoing_right_slots,
        outgoing_right_valid,
    )
    keys_a_filled = jnp.where(valid_a, keys_a, key_fill)
    outgoing_left_filled = jnp.where(
        outgoing_left_valid, outgoing_left_keys, key_fill
    )
    outgoing_right_filled = jnp.where(
        outgoing_right_valid, outgoing_right_keys, key_fill
    )
    incoming_filled = jnp.where(incoming_valid, incoming_keys, key_fill)
    incoming_source_idx = jnp.arange(incoming_keys.shape[0], dtype=jnp.int32)
    rank_stay = (
        jnp.searchsorted(keys_a_filled, incoming_filled, side="right")
        - jnp.searchsorted(outgoing_left_filled, incoming_filled, side="right")
        - jnp.searchsorted(outgoing_right_filled, incoming_filled, side="right")
    ).astype(jnp.int32)
    incoming_pos = incoming_source_idx + rank_stay
    incoming_pos = jnp.where(incoming_valid, incoming_pos, jnp.int32(capacity))

    slots = jnp.arange(capacity, dtype=jnp.int32)
    is_incoming = jnp.zeros((capacity,), dtype=jnp.bool_).at[incoming_pos].set(
        incoming_valid, mode="drop"
    )
    incoming_prefix = jnp.cumsum(is_incoming, dtype=jnp.int32)
    incoming_idx = jnp.clip(
        incoming_prefix - 1, 0, incoming_keys.shape[0] - 1
    )
    stay_rank = jnp.clip(slots - incoming_prefix, 0, capacity - 1)
    a_idx = jnp.clip(
        stay_rank + skipped[stay_rank], 0, valid_a.shape[0] - 1
    )
    total = count_stay + jnp.sum(incoming_valid)
    out_valid = slots < total
    return (
        is_incoming,
        a_idx,
        incoming_idx,
        stay_rank,
        out_valid,
        total,
        stay_pos,
        stay_valid,
    )


def _merge_small_incoming_streams(incoming_b, incoming_c, key_fill):
    """Stably merge two share-capacity streams before the full-capacity route."""
    keys_b, *payload_b, valid_b = incoming_b
    keys_c, *payload_c, valid_c = incoming_c
    capacity = keys_b.shape[0] + keys_c.shape[0]
    plan = _linear_merge_plan_two(
        keys_b, valid_b, keys_c, valid_c, capacity, key_fill
    )
    source_code, b_idx, c_idx, out_valid, _ = plan
    out_keys = _linear_take_two(
        keys_b, keys_c, source_code, b_idx, c_idx
    )
    fields = [
        _linear_take_two(b, c, source_code, b_idx, c_idx)
        for b, c in zip(payload_b, payload_c)
    ]
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    fields = [_zero_invalid_merge_values(field, out_valid) for field in fields]
    source_tag = jnp.where(source_code == 1, jnp.int32(2), jnp.int32(1))
    source_tag = jnp.where(out_valid, source_tag, jnp.int32(3))
    source_idx = jnp.where(source_code == 1, c_idx, b_idx)
    source_idx = jnp.where(out_valid, source_idx, jnp.int32(-1))
    return (out_keys, *fields, out_valid), source_tag, source_idx


def _sparse_route_merge_two(
    original,
    outgoing,
    outgoing_slots,
    incoming,
    capacity,
    key_fill,
    error_message,
    provenance=False,
    incoming_tag=jnp.int32(2),
):
    """Merge an original authoritative stream directly with one incoming stream."""
    keys_a, *payload_a, valid_a = original
    outgoing_keys, *_, outgoing_valid = outgoing
    keys_b, *payload_b, valid_b = incoming
    plan = _sparse_route_plan_two(
        keys_a,
        valid_a,
        outgoing_keys,
        outgoing_valid,
        outgoing_slots,
        keys_b,
        valid_b,
        capacity,
        key_fill,
    )
    source_code, a_idx, b_idx, stay_rank, out_valid, total, stay_pos, stay_valid = plan
    _capacity_check(total, capacity, error_message)
    out_keys = _linear_take_two(keys_a, keys_b, source_code, a_idx, b_idx)
    fields = [
        _linear_take_two(a, b, source_code, a_idx, b_idx)
        for a, b in zip(payload_a, payload_b)
    ]
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    fields = [_zero_invalid_merge_values(field, out_valid) for field in fields]
    merged = (out_keys, *fields, out_valid)
    if provenance:
        out_src_tag = jnp.where(source_code == 1, incoming_tag, jnp.int32(0))
        out_src_tag = jnp.where(out_valid, out_src_tag, jnp.int32(3))
        out_src_idx = jnp.where(source_code == 1, b_idx, stay_rank)
        out_src_idx = jnp.where(out_valid, out_src_idx, jnp.int32(-1))
        merged = (*merged, out_src_tag, out_src_idx)
    return merged, stay_pos, stay_valid


def _sparse_route_merge_three(
    original,
    outgoing_left,
    outgoing_left_slots,
    outgoing_right,
    outgoing_right_slots,
    incoming_b,
    incoming_c,
    capacity,
    key_fill,
    error_message,
    provenance=False,
):
    """Merge an original authoritative stream directly with two incoming streams."""
    keys_a, *payload_a, valid_a = original
    outgoing_left_keys, *_, outgoing_left_valid = outgoing_left
    outgoing_right_keys, *_, outgoing_right_valid = outgoing_right
    combined, combined_tag, combined_idx = _merge_small_incoming_streams(
        incoming_b, incoming_c, key_fill
    )
    incoming_keys, *incoming_payload, incoming_valid = combined
    plan = _sparse_route_plan_one_incoming_two_outgoing(
        keys_a,
        valid_a,
        outgoing_left_keys,
        outgoing_left_valid,
        outgoing_left_slots,
        outgoing_right_keys,
        outgoing_right_valid,
        outgoing_right_slots,
        incoming_keys,
        incoming_valid,
        capacity,
        key_fill,
    )
    (
        is_incoming,
        a_idx,
        incoming_idx,
        stay_rank,
        out_valid,
        total,
        stay_pos,
        stay_valid,
    ) = plan
    _capacity_check(total, capacity, error_message)
    out_keys = _linear_take_two(
        keys_a, incoming_keys, is_incoming, a_idx, incoming_idx
    )
    fields = [
        _linear_take_two(a, incoming, is_incoming, a_idx, incoming_idx)
        for a, incoming in zip(payload_a, incoming_payload)
    ]
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    fields = [_zero_invalid_merge_values(field, out_valid) for field in fields]
    merged = (out_keys, *fields, out_valid)
    if provenance:
        out_src_tag = jnp.where(
            is_incoming, combined_tag[incoming_idx], jnp.int32(0)
        )
        out_src_tag = jnp.where(out_valid, out_src_tag, jnp.int32(3))
        out_src_idx = jnp.where(
            is_incoming, combined_idx[incoming_idx], stay_rank
        )
        out_src_idx = jnp.where(out_valid, out_src_idx, jnp.int32(-1))
        merged = (*merged, out_src_tag, out_src_idx)
    return merged, stay_pos, stay_valid


def _pack_left_halo_and_authoritative(
    left_keys,
    left_pmid,
    left_disp,
    left_vel,
    left_acc,
    left_valid,
    auth_keys,
    auth_pmid,
    auth_disp,
    auth_vel,
    auth_acc,
    auth_valid,
    max_ptcl_per_slice,
    halo_start,
    halo_end,
    global_nMesh,
    disp_size,
):
    """Build ``particle_halo`` storage: left-halo copies followed by owned particles."""
    del left_keys, auth_keys
    left_count = jnp.sum(left_valid)
    auth_count = jnp.sum(auth_valid)
    total = left_count + auth_count
    _capacity_check(
        total,
        max_ptcl_per_slice,
        "[ERROR] Exceeded canonical particle storage capacity. "
        "required_slots={x}, max_ptcl_per_slice={y}.",
    )

    pmid = jnp.zeros((max_ptcl_per_slice, left_pmid.shape[1]), dtype=left_pmid.dtype)
    disp = jnp.zeros((max_ptcl_per_slice, left_disp.shape[1]), dtype=left_disp.dtype)
    vel = jnp.zeros((max_ptcl_per_slice, left_vel.shape[1]), dtype=left_vel.dtype)
    acc = jnp.zeros((max_ptcl_per_slice, left_acc.shape[1]), dtype=left_acc.dtype)
    slots = jnp.arange(max_ptcl_per_slice, dtype=jnp.int32)
    left_mask = slots < left_count
    auth_mask = (slots >= left_count) & (slots < total)
    left_idx = jnp.minimum(slots, left_pmid.shape[0] - 1)
    auth_idx = jnp.maximum(slots - left_count.astype(jnp.int32), 0)
    auth_idx = jnp.minimum(auth_idx, auth_pmid.shape[0] - 1)

    pmid = jnp.where(left_mask[:, None], left_pmid[left_idx], pmid)
    disp = jnp.where(left_mask[:, None], left_disp[left_idx], disp)
    vel = jnp.where(left_mask[:, None], left_vel[left_idx], vel)
    acc = jnp.where(left_mask[:, None], left_acc[left_idx], acc)

    pmid = jnp.where(auth_mask[:, None], auth_pmid[auth_idx], pmid)
    disp = jnp.where(auth_mask[:, None], auth_disp[auth_idx], disp)
    vel = jnp.where(auth_mask[:, None], auth_vel[auth_idx], vel)
    acc = jnp.where(auth_mask[:, None], auth_acc[auth_idx], acc)

    unused_index = jnp.arange(max_ptcl_per_slice) >= total
    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
    halo_mask = compute_halo_mask(x_mod, halo_start.squeeze(), halo_end.squeeze(), unused_index)
    return pmid, disp, vel, acc, halo_mask, unused_index


def _pack_authoritative_only(
    auth_pmid,
    auth_disp,
    auth_vel,
    auth_acc,
    auth_valid,
    max_ptcl_per_slice,
):
    """Pack the canonical authoritative block without duplicating halo particles."""
    auth_count = jnp.sum(auth_valid)
    _capacity_check(
        auth_count,
        max_ptcl_per_slice,
        "[ERROR] Exceeded authoritative-only storage capacity. "
        "required_slots={x}, max_ptcl_per_slice={y}.",
    )

    if auth_pmid.shape[0] == max_ptcl_per_slice:
        def _mask_unused(values):
            valid_shape = (auth_valid.shape[0],) + (1,) * (values.ndim - 1)
            return jnp.where(auth_valid.reshape(valid_shape), values, jnp.zeros_like(values))

        pmid = _mask_unused(auth_pmid)
        disp = _mask_unused(auth_disp)
        vel = _mask_unused(auth_vel)
        acc = _mask_unused(auth_acc)
        unused_index = ~auth_valid
        halo_mask = jnp.zeros_like(unused_index)
        return pmid, disp, vel, acc, halo_mask, unused_index

    pmid = jnp.zeros((max_ptcl_per_slice, auth_pmid.shape[1]), dtype=auth_pmid.dtype)
    disp = jnp.zeros((max_ptcl_per_slice, auth_disp.shape[1]), dtype=auth_disp.dtype)
    vel = jnp.zeros((max_ptcl_per_slice, auth_vel.shape[1]), dtype=auth_vel.dtype)
    acc = jnp.zeros((max_ptcl_per_slice, auth_acc.shape[1]), dtype=auth_acc.dtype)
    slots = jnp.arange(max_ptcl_per_slice, dtype=jnp.int32)
    auth_mask = slots < auth_count
    auth_idx = jnp.minimum(slots, auth_pmid.shape[0] - 1)

    pmid = jnp.where(auth_mask[:, None], auth_pmid[auth_idx], pmid)
    disp = jnp.where(auth_mask[:, None], auth_disp[auth_idx], disp)
    vel = jnp.where(auth_mask[:, None], auth_vel[auth_idx], vel)
    acc = jnp.where(auth_mask[:, None], auth_acc[auth_idx], acc)

    unused_index = slots >= auth_count
    halo_mask = jnp.zeros_like(unused_index)
    return pmid, disp, vel, acc, halo_mask, unused_index


def _pack_authoritative_only_no_acc(
    auth_pmid,
    auth_disp,
    auth_vel,
    auth_valid,
    max_ptcl_per_slice,
):
    """Pack an authoritative mesh-halo block while dropping acceleration payload."""
    auth_count = jnp.sum(auth_valid)
    _capacity_check(
        auth_count,
        max_ptcl_per_slice,
        "[ERROR] Exceeded authoritative-only storage capacity. "
        "required_slots={x}, max_ptcl_per_slice={y}.",
    )

    if auth_pmid.shape[0] == max_ptcl_per_slice:
        def _mask_unused(values):
            valid_shape = (auth_valid.shape[0],) + (1,) * (values.ndim - 1)
            return jnp.where(auth_valid.reshape(valid_shape), values, jnp.zeros_like(values))

        pmid = _mask_unused(auth_pmid)
        disp = _mask_unused(auth_disp)
        vel = _mask_unused(auth_vel)
        unused_index = ~auth_valid
        halo_mask = jnp.zeros_like(unused_index)
        return pmid, disp, vel, halo_mask, unused_index

    pmid = jnp.zeros((max_ptcl_per_slice, auth_pmid.shape[1]), dtype=auth_pmid.dtype)
    disp = jnp.zeros((max_ptcl_per_slice, auth_disp.shape[1]), dtype=auth_disp.dtype)
    vel = jnp.zeros((max_ptcl_per_slice, auth_vel.shape[1]), dtype=auth_vel.dtype)
    slots = jnp.arange(max_ptcl_per_slice, dtype=jnp.int32)
    auth_mask = slots < auth_count
    auth_idx = jnp.minimum(slots, auth_pmid.shape[0] - 1)

    pmid = jnp.where(auth_mask[:, None], auth_pmid[auth_idx], pmid)
    disp = jnp.where(auth_mask[:, None], auth_disp[auth_idx], disp)
    vel = jnp.where(auth_mask[:, None], auth_vel[auth_idx], vel)

    unused_index = slots >= auth_count
    halo_mask = jnp.zeros_like(unused_index)
    return pmid, disp, vel, halo_mask, unused_index


def _authoritative_prefix_from_owned_only(
    pmid,
    disp,
    vel,
    acc,
    unused_index,
    conf,
):
    """Treat a mesh-halo packed state as its already-authoritative prefix block."""
    valid = ~unused_index
    keys = pmid_to_idx(pmid, conf)
    keys = jnp.where(valid, keys, _key_fill_value(conf))
    return keys, pmid, disp, vel, acc, valid


def _authoritative_prefix_from_owned_only_no_acc(
    pmid,
    disp,
    vel,
    unused_index,
    conf,
):
    """Treat a mesh-halo packed state as an authoritative prefix without acceleration."""
    valid = ~unused_index
    keys = pmid_to_idx(pmid, conf)
    keys = jnp.where(valid, keys, _key_fill_value(conf))
    return keys, pmid, disp, vel, valid


def _reverse_build_owned_only_cot(full_cot, auth_size, auth_valid):
    """Transpose of `_pack_authoritative_only` for one payload field stack."""
    del auth_size
    valid_mask = auth_valid.reshape((auth_valid.shape[0],) + (1,) * (full_cot.ndim - 1))
    return full_cot * valid_mask.astype(full_cot.dtype)


def _canonical_authoritative_from_full(
    pmid,
    source_disp,
    carried_disp,
    vel,
    acc,
    unused_index,
    global_nMesh,
    disp_size,
    num_gpus,
    offsets,
    conf,
):
    """Extract owned authoritative particles from a full particle-halo slab."""
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    x_mod = _x_mod_from_disp(pmid, source_disp, global_nMesh, disp_size)
    owned_mask = particles_in_slice_mask(x_mod, owned_start, owned_end) & ~unused_index
    keys = pmid_to_idx(pmid, conf)
    return _compact_sorted_particles(
        keys,
        pmid,
        carried_disp,
        vel,
        acc,
        owned_mask,
        pmid.shape[0],
        _key_fill_value(conf),
        "[ERROR] Exceeded authoritative compact capacity. "
        "authoritative_particles={x}, compact_capacity={y}.",
    )


def _canonical_authoritative_from_full_with_slots(
    pmid,
    source_disp,
    carried_disp,
    vel,
    acc,
    unused_index,
    global_nMesh,
    disp_size,
    num_gpus,
    offsets,
    conf,
):
    """Extract owned authoritative particles and keep original slot indices."""
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    x_mod = _x_mod_from_disp(pmid, source_disp, global_nMesh, disp_size)
    owned_mask = particles_in_slice_mask(x_mod, owned_start, owned_end) & ~unused_index
    keys = pmid_to_idx(pmid, conf)
    return _compact_sorted_particles_with_slots(
        keys,
        pmid,
        carried_disp,
        vel,
        acc,
        owned_mask,
        pmid.shape[0],
        _key_fill_value(conf),
        "[ERROR] Exceeded authoritative compact capacity. "
        "authoritative_particles={x}, compact_capacity={y}.",
    )


def _scatter_compact_to_dense(compact_values, compact_slots, compact_valid, out_size):
    """Scatter compact cotangents back to their original dense slots."""
    out = jnp.zeros((out_size,) + compact_values.shape[1:], dtype=compact_values.dtype)
    slots = jnp.where(compact_valid, compact_slots, 0)
    mask = compact_valid.reshape((compact_valid.shape[0],) + (1,) * (compact_values.ndim - 1))
    values = compact_values * mask.astype(compact_values.dtype)
    return out.at[slots].add(values)


def _mask_compact_prefix(compact_values, compact_valid):
    """Zero invalid entries in a compact fixed-capacity buffer."""
    mask = compact_valid.reshape((compact_valid.shape[0],) + (1,) * (compact_values.ndim - 1))
    return compact_values * mask.astype(compact_values.dtype)

def _canonical_route_authoritative(
    keys,
    pmid,
    disp,
    vel,
    acc,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Route authoritative particles to their post-drift owner slabs."""
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    slice_width = global_nMesh // num_gpus
    left_start = (owned_start - slice_width) % global_nMesh
    right_end = (owned_end + slice_width) % global_nMesh

    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
    stay_mask = valid & particles_in_slice_mask(x_mod, owned_start, owned_end)
    send_left_mask = valid & particles_in_slice_mask(x_mod, left_start, owned_start)
    send_right_mask = valid & particles_in_slice_mask(x_mod, owned_end, right_end)
    if num_gpus == 2:
        send_right_mask = jnp.zeros_like(send_right_mask)
    dropped_mask = valid & ~(stay_mask | send_left_mask | send_right_mask)

    _synchronized_nonzero_check(
        jnp.sum(dropped_mask),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        jnp.maximum(jnp.sum(send_left_mask), jnp.sum(send_right_mask)),
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )

    key_fill = _key_fill_value(conf)
    original = (keys, pmid, disp, vel, acc, valid)
    *send_left_items, send_left_pos = _compact_sorted_particles_with_slots(
        keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share,
        key_fill,
        "[ERROR] Exceeded left-migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_left = tuple(send_left_items)
    if num_gpus == 2:
        incoming_from_right = _exchange_compacted_particles(send_left, left_perm, conf)
        merged, _, _ = _sparse_route_merge_two(
            original, send_left, send_left_pos, incoming_from_right, pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        return merged, jnp.sum(send_left[-1])

    *send_right_items, send_right_pos = _compact_sorted_particles_with_slots(
        keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share,
        key_fill,
        "[ERROR] Exceeded right-migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_right = tuple(send_right_items)
    incoming_from_left = _exchange_compacted_particles(send_right, right_perm, conf)
    incoming_from_right = _exchange_compacted_particles(send_left, left_perm, conf)
    merged, _, _ = _sparse_route_merge_three(
        original, send_left, send_left_pos, send_right, send_right_pos,
        incoming_from_left, incoming_from_right, pmid.shape[0], key_fill,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
    )
    max_particles_moved = jnp.maximum(jnp.sum(send_left[-1]), jnp.sum(send_right[-1]))
    return merged, max_particles_moved

def _canonical_route_authoritative_no_acc(
    keys,
    pmid,
    disp,
    vel,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Route authoritative particles to post-drift owner slabs without acceleration."""
    if cuda_routing_enabled(conf):
        return _canonical_route_authoritative_no_acc_cuda(
            keys,
            pmid,
            disp,
            vel,
            valid,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )

    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    slice_width = global_nMesh // num_gpus
    left_start = (owned_start - slice_width) % global_nMesh
    right_end = (owned_end + slice_width) % global_nMesh

    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
    stay_mask = valid & particles_in_slice_mask(x_mod, owned_start, owned_end)
    send_left_mask = valid & particles_in_slice_mask(x_mod, left_start, owned_start)
    send_right_mask = valid & particles_in_slice_mask(x_mod, owned_end, right_end)
    if num_gpus == 2:
        send_right_mask = jnp.zeros_like(send_right_mask)
    dropped_mask = valid & ~(stay_mask | send_left_mask | send_right_mask)

    _synchronized_nonzero_check(
        jnp.sum(dropped_mask),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        jnp.maximum(jnp.sum(send_left_mask), jnp.sum(send_right_mask)),
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )

    key_fill = _key_fill_value(conf)
    original = (keys, pmid, disp, vel, valid)
    *send_left_items, send_left_pos = _compact_sorted_particles_no_acc_with_slots(
        keys, pmid, disp, vel, send_left_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded left-migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_left = tuple(send_left_items)
    if num_gpus == 2:
        incoming_from_right = _exchange_compacted_particles(send_left, left_perm, conf)
        merged, _, _ = _sparse_route_merge_two(
            original, send_left, send_left_pos, incoming_from_right, pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        return merged, jnp.sum(send_left[-1])

    *send_right_items, send_right_pos = _compact_sorted_particles_no_acc_with_slots(
        keys, pmid, disp, vel, send_right_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded right-migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_right = tuple(send_right_items)
    incoming_from_left = _exchange_compacted_particles(send_right, right_perm, conf)
    incoming_from_right = _exchange_compacted_particles(send_left, left_perm, conf)
    merged, _, _ = _sparse_route_merge_three(
        original, send_left, send_left_pos, send_right, send_right_pos,
        incoming_from_left, incoming_from_right, pmid.shape[0], key_fill,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
    )
    max_particles_moved = jnp.maximum(jnp.sum(send_left[-1]), jnp.sum(send_right[-1]))
    return merged, max_particles_moved

def _canonical_route_authoritative_no_acc_bidir_cuda(
    keys,
    pmid,
    disp,
    vel,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Fused bidirectional CUDA route used by the full production pipeline."""
    if not _DEBUG_BIDIR_ROUTE:
        del keys
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    mesh_shape = tuple(int(value) for value in conf.mesh_shape)
    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
    (
        send_left_records,
        send_right_records,
        send_left_count,
        send_right_count,
        classes,
        stay_keys,
        stay_indices,
        stay_count,
    ) = route_pack_bidir_cuda(
        pmid,
        disp,
        vel,
        valid,
        x_mod,
        global_nmesh=global_nMesh,
        mesh_shape=mesh_shape,
        owned_start=owned_start,
        owned_end=owned_end,
        slice_width=global_nMesh // num_gpus,
        num_devices=num_gpus,
        capacity=max_values_to_share,
        stay_capacity=pmid.shape[0],
    )
    _synchronized_nonzero_check(
        jnp.sum(classes == 4),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        jnp.maximum(send_left_count, send_right_count),
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    incoming_from_left = jax.lax.ppermute(
        send_right_records, axis_name=AXIS_NAME, perm=right_perm
    )
    incoming_from_left_count = jax.lax.ppermute(
        send_right_count, axis_name=AXIS_NAME, perm=right_perm
    )
    incoming_from_right = jax.lax.ppermute(
        send_left_records, axis_name=AXIS_NAME, perm=left_perm
    )
    incoming_from_right_count = jax.lax.ppermute(
        send_left_count, axis_name=AXIS_NAME, perm=left_perm
    )
    _synchronized_capacity_check(
        stay_count + incoming_from_left_count + incoming_from_right_count,
        pmid.shape[0],
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
    )
    (
        merged_pmid,
        merged_disp,
        merged_vel,
        merged_valid,
        _merged_tag,
        _merged_idx,
        _merged_key,
        _merged_count,
    ) = route_merge_bidir_cuda(
        pmid,
        disp,
        vel,
        stay_keys,
        stay_indices,
        stay_count,
        incoming_from_left,
        incoming_from_left_count,
        incoming_from_right,
        incoming_from_right_count,
        mesh_shape=mesh_shape,
        capacity=pmid.shape[0],
    )
    merged_valid = merged_valid != 0
    merged_keys = pmid_to_idx(merged_pmid, conf)
    merged_keys = jnp.where(merged_valid, merged_keys, _key_fill_value(conf))
    if _DEBUG_BIDIR_ROUTE:
        left_start = (owned_start - global_nMesh // num_gpus) % global_nMesh
        right_end = (owned_end + global_nMesh // num_gpus) % global_nMesh
        expected_stay = valid & particles_in_slice_mask(x_mod, owned_start, owned_end)
        expected_left = valid & particles_in_slice_mask(x_mod, left_start, owned_start)
        expected_right = valid & particles_in_slice_mask(x_mod, owned_end, right_end)
        if num_gpus == 2:
            expected_right = jnp.zeros_like(expected_right)
        expected_classes = jnp.where(
            expected_stay,
            jnp.uint8(1),
            jnp.where(
                expected_left,
                jnp.uint8(2),
                jnp.where(expected_right, jnp.uint8(3), jnp.where(valid, jnp.uint8(4), jnp.uint8(0))),
            ),
        )
        # `_canonical_route_authoritative` is the established JAX reference;
        # it deliberately does not dispatch to the CUDA implementation.
        reference, _ = _canonical_route_authoritative(
            keys,
            pmid,
            disp,
            vel,
            jnp.zeros_like(disp),
            valid,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
        reference_keys, reference_pmid, reference_disp, reference_vel, _, reference_valid = reference
        active = merged_valid | reference_valid
        row_mismatch = (
            (merged_valid != reference_valid)
            | (active & (merged_keys != reference_keys))
            | (active & jnp.any(merged_pmid != reference_pmid, axis=1))
            | (active & jnp.any(merged_disp != reference_disp, axis=1))
            | (active & jnp.any(merged_vel != reference_vel, axis=1))
        )
        first_mismatch = jnp.min(
            jnp.where(row_mismatch, jnp.arange(pmid.shape[0], dtype=jnp.int32), pmid.shape[0])
        )
        safe_first = jnp.minimum(first_mismatch, pmid.shape[0] - 1)
        active_2d = active.reshape((-1, 1))
        disp_max_error = jnp.max(
            jnp.where(active_2d, jnp.abs(merged_disp - reference_disp), 0.0)
        )
        vel_max_error = jnp.max(
            jnp.where(active_2d, jnp.abs(merged_vel - reference_vel), 0.0)
        )
        # Keep the oracle semantically live even when a JAX debug callback is
        # not emitted by the surrounding custom-VJP lowering.  A native route
        # must reproduce canonical validity, ordering, ids and payload bits
        # exactly; tiny density agreement is not an adequate substitute.
        _synchronized_nonzero_check(
            jnp.sum(row_mismatch),
            "[ERROR] Native bidirectional route disagrees with the canonical route. "
            "mismatched_output_rows={x}.",
        )
        jax.debug.print(
            "PM++ bidir route oracle gpu={gpu}: class_mismatch={class_mismatch}; "
            "pack L/R/S native=({native_left},{native_right},{native_stay}) "
            "expected=({expected_left},{expected_right},{expected_stay}); "
            "merge valid native/reference=({native_valid},{reference_valid}); "
            "row_mismatch={row_mismatch}; first={first}; key native/reference=({native_key},{reference_key}); "
            "pmid native/reference=({native_pmid},{reference_pmid}); "
            "max|disp|={disp_max}; max|vel|={vel_max}",
            gpu=jax.lax.axis_index(AXIS_NAME),
            class_mismatch=jnp.sum(classes != expected_classes),
            native_left=send_left_count,
            native_right=send_right_count,
            native_stay=stay_count,
            expected_left=jnp.sum(expected_left),
            expected_right=jnp.sum(expected_right),
            expected_stay=jnp.sum(expected_stay),
            native_valid=jnp.sum(merged_valid),
            reference_valid=jnp.sum(reference_valid),
            row_mismatch=jnp.sum(row_mismatch),
            first=first_mismatch,
            native_key=merged_keys[safe_first],
            reference_key=reference_keys[safe_first],
            native_pmid=merged_pmid[safe_first],
            reference_pmid=reference_pmid[safe_first],
            disp_max=disp_max_error,
            vel_max=vel_max_error,
        )
    return (
        (merged_keys, merged_pmid.astype(pmid.dtype), merged_disp, merged_vel, merged_valid),
        jnp.maximum(send_left_count, send_right_count),
    )


def _canonical_route_authoritative_no_acc_cuda(
    keys,
    pmid,
    disp,
    vel,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """CUDA pack/JAX exchange/CUDA merge route for authoritative particles.

    The function is deliberately a sibling of the canonical implementation,
    rather than a replacement for it.  All topology decisions and collective
    ordering remain visible in JAX, which keeps the operation compatible with
    XLA's collective scheduler and with the existing hand-written adjoint.
    """
    if _use_bidir_cuda_routing(conf):
        return _canonical_route_authoritative_no_acc_bidir_cuda(
            keys, pmid, disp, vel, valid, global_nMesh, max_values_to_share,
            left_perm, right_perm, num_gpus, disp_size, offsets, conf,
        )
    del keys
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    slice_width = global_nMesh // num_gpus
    mesh_shape = tuple(int(value) for value in conf.mesh_shape)
    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)

    send_left_records, send_left_count, classes = cuda_route_pack(
        pmid,
        disp,
        vel,
        valid,
        x_mod,
        global_nmesh=global_nMesh,
        mesh_shape=mesh_shape,
        owned_start=owned_start,
        owned_end=owned_end,
        slice_width=slice_width,
        direction=-1,
        num_devices=num_gpus,
        capacity=max_values_to_share,
    )
    _synchronized_nonzero_check(
        jnp.sum(classes == 4),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        send_left_count,
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )

    stay_mask = classes == 1
    if num_gpus == 2:
        incoming_from_right = jax.lax.ppermute(
            send_left_records, axis_name=AXIS_NAME, perm=left_perm
        )
        incoming_from_right_count = jax.lax.ppermute(
            send_left_count, axis_name=AXIS_NAME, perm=left_perm
        )
        _synchronized_capacity_check(
            jnp.sum(stay_mask) + incoming_from_right_count,
            pmid.shape[0],
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        merged_pmid, merged_disp, merged_vel, merged_valid = cuda_route_merge(
            pmid,
            disp,
            vel,
            stay_mask,
            incoming_from_right,
            incoming_from_right_count,
            mesh_shape=mesh_shape,
            capacity=pmid.shape[0],
        )
        max_particles_moved = send_left_count
    else:
        # Match the canonical three-stream tie order exactly: the existing
        # stay stream, incoming-from-left (the right-going export), then
        # incoming-from-right (the left-going export).  Two sequential stable
        # merges preserve that order without a special multi-stream FFI ABI.
        send_right_records, send_right_count, _ = cuda_route_pack(
            pmid,
            disp,
            vel,
            valid,
            x_mod,
            global_nmesh=global_nMesh,
            mesh_shape=mesh_shape,
            owned_start=owned_start,
            owned_end=owned_end,
            slice_width=slice_width,
            direction=1,
            num_devices=num_gpus,
            capacity=max_values_to_share,
        )
        _synchronized_capacity_check(
            send_right_count,
            max_values_to_share,
            "[ERROR] Exceeded migration share capacity. "
            "particles_to_share={x}, max_share_ptcl={y}.",
        )
        incoming_from_left = jax.lax.ppermute(
            send_right_records, axis_name=AXIS_NAME, perm=right_perm
        )
        incoming_from_left_count = jax.lax.ppermute(
            send_right_count, axis_name=AXIS_NAME, perm=right_perm
        )
        _synchronized_capacity_check(
            jnp.sum(stay_mask) + incoming_from_left_count,
            pmid.shape[0],
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        merged_pmid, merged_disp, merged_vel, merged_valid = cuda_route_merge(
            pmid,
            disp,
            vel,
            stay_mask,
            incoming_from_left,
            incoming_from_left_count,
            mesh_shape=mesh_shape,
            capacity=pmid.shape[0],
        )
        incoming_from_right = jax.lax.ppermute(
            send_left_records, axis_name=AXIS_NAME, perm=left_perm
        )
        incoming_from_right_count = jax.lax.ppermute(
            send_left_count, axis_name=AXIS_NAME, perm=left_perm
        )
        merged_valid_mask = merged_valid != 0
        _synchronized_capacity_check(
            jnp.sum(merged_valid_mask) + incoming_from_right_count,
            pmid.shape[0],
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        merged_pmid, merged_disp, merged_vel, merged_valid = cuda_route_merge(
            merged_pmid,
            merged_disp,
            merged_vel,
            merged_valid_mask,
            incoming_from_right,
            incoming_from_right_count,
            mesh_shape=mesh_shape,
            capacity=pmid.shape[0],
        )
        max_particles_moved = jnp.maximum(send_left_count, send_right_count)

    merged_valid = merged_valid != 0
    merged_keys = pmid_to_idx(merged_pmid, conf)
    merged_keys = jnp.where(merged_valid, merged_keys, _key_fill_value(conf))
    return (
        (merged_keys, merged_pmid.astype(pmid.dtype), merged_disp, merged_vel, merged_valid),
        max_particles_moved,
    )


def _canonical_route_authoritative_with_aux_bidir_cuda(
    keys,
    pmid,
    disp,
    vel,
    acc,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Fused bidirectional route and provenance plan for full AD."""
    del keys
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    mesh_shape = tuple(int(value) for value in conf.mesh_shape)
    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
    (
        send_left_records,
        send_right_records,
        send_left_count,
        send_right_count,
        classes,
        stay_keys,
        stay_indices,
        stay_count,
    ) = route_pack_bidir_cuda(
        pmid, disp, vel, valid, x_mod,
        global_nmesh=global_nMesh, mesh_shape=mesh_shape,
        owned_start=owned_start, owned_end=owned_end,
        slice_width=global_nMesh // num_gpus, num_devices=num_gpus,
        capacity=max_values_to_share,
        stay_capacity=pmid.shape[0],
    )
    stay_mask = classes == 1
    send_left_mask = classes == 2
    send_right_mask = classes == 3
    _synchronized_nonzero_check(
        jnp.sum(classes == 4),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        jnp.maximum(send_left_count, send_right_count), max_values_to_share,
        "[ERROR] Exceeded migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )

    auth_size = pmid.shape[0]
    auth_slots = jnp.arange(auth_size, dtype=jnp.int32)
    stay_pos = jnp.compress(
        stay_mask, auth_slots, axis=0, size=auth_size,
        fill_value=jnp.asarray(-1, dtype=jnp.int32),
    )
    stay_valid = jnp.arange(auth_size, dtype=jnp.int32) < stay_count
    send_left_pos = jnp.compress(
        send_left_mask, auth_slots, axis=0, size=max_values_to_share,
        fill_value=jnp.asarray(-1, dtype=jnp.int32),
    )
    send_right_pos = jnp.compress(
        send_right_mask, auth_slots, axis=0, size=max_values_to_share,
        fill_value=jnp.asarray(-1, dtype=jnp.int32),
    )
    send_left_valid = jnp.arange(max_values_to_share, dtype=jnp.int32) < send_left_count
    send_right_valid = jnp.arange(max_values_to_share, dtype=jnp.int32) < send_right_count

    stay_acc = jnp.compress(
        stay_mask, acc, axis=0, size=auth_size,
        fill_value=jnp.asarray(0, dtype=acc.dtype),
    )
    send_left_acc = jnp.compress(
        send_left_mask, acc, axis=0, size=max_values_to_share,
        fill_value=jnp.asarray(0, dtype=acc.dtype),
    )
    send_right_acc = jnp.compress(
        send_right_mask, acc, axis=0, size=max_values_to_share,
        fill_value=jnp.asarray(0, dtype=acc.dtype),
    )
    incoming_from_left = jax.lax.ppermute(send_right_records, axis_name=AXIS_NAME, perm=right_perm)
    incoming_from_left_count = jax.lax.ppermute(send_right_count, axis_name=AXIS_NAME, perm=right_perm)
    incoming_from_left_acc = jax.lax.ppermute(send_right_acc, axis_name=AXIS_NAME, perm=right_perm)
    incoming_from_right = jax.lax.ppermute(send_left_records, axis_name=AXIS_NAME, perm=left_perm)
    incoming_from_right_count = jax.lax.ppermute(send_left_count, axis_name=AXIS_NAME, perm=left_perm)
    incoming_from_right_acc = jax.lax.ppermute(send_left_acc, axis_name=AXIS_NAME, perm=left_perm)
    _synchronized_capacity_check(
        stay_count + incoming_from_left_count + incoming_from_right_count,
        auth_size,
        "[ERROR] Exceeded canonical authoritative capacity after migration. required_particles={x}, max_ptcl_per_slice={y}.",
    )
    (
        merged_pmid, merged_disp, merged_vel, merged_valid,
        merged_tag, merged_idx, _merged_key, _merged_count,
    ) = route_merge_bidir_cuda(
        pmid, disp, vel, stay_keys, stay_indices, stay_count,
        incoming_from_left, incoming_from_left_count,
        incoming_from_right, incoming_from_right_count,
        mesh_shape=mesh_shape, capacity=auth_size,
    )
    merged_valid = merged_valid != 0
    merged_idx = jnp.where(merged_valid, merged_idx, jnp.int32(-1))
    safe_auth_idx = jnp.clip(merged_idx, 0, max(auth_size, 1) - 1)
    safe_share_idx = jnp.clip(merged_idx, 0, max(max_values_to_share, 1) - 1)
    zero_acc = jnp.zeros_like(acc)
    merged_acc = jnp.where(
        (merged_tag == 0).reshape((-1, 1)),
        stay_acc[safe_auth_idx],
        jnp.where(
            (merged_tag == 1).reshape((-1, 1)),
            incoming_from_left_acc[safe_share_idx],
            jnp.where(
                (merged_tag == 2).reshape((-1, 1)),
                incoming_from_right_acc[safe_share_idx],
                zero_acc[:auth_size],
            ),
        ),
    )
    merged_keys = pmid_to_idx(merged_pmid, conf)
    merged_keys = jnp.where(merged_valid, merged_keys, _key_fill_value(conf))
    route_aux = (
        stay_pos, stay_valid, send_left_pos, send_left_valid,
        send_right_pos, send_right_valid, merged_tag, merged_idx,
    )
    return (
        merged_keys, merged_pmid.astype(pmid.dtype), merged_disp,
        merged_vel, merged_acc, merged_valid,
    ), route_aux


def _canonical_route_authoritative_with_aux_cuda(
    keys,
    pmid,
    disp,
    vel,
    acc,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Recompute the CUDA route plan used by the hand-written adjoint.

    The forward record intentionally carries no acceleration.  The adjoint
    only needs the route provenance, so acceleration is compacted/exchanged
    separately here solely to preserve this helper's existing internal return
    contract; the FFI record remains the fixed 32-byte displacement/velocity
    record used by forward routing.

    The experimental bidirectional merge-path kernel is deliberately not used
    here. Its primal route is safe to use in the forward simulation, but its
    three-stream provenance reconstruction has not qualified against the
    production full gradient. The existing two-pass CUDA route below has the
    established transpose contract and reconstructs the same canonical route.
    """
    del keys
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    slice_width = global_nMesh // num_gpus
    mesh_shape = tuple(int(value) for value in conf.mesh_shape)
    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)

    send_left_records, send_left_count, classes = cuda_route_pack(
        pmid,
        disp,
        vel,
        valid,
        x_mod,
        global_nmesh=global_nMesh,
        mesh_shape=mesh_shape,
        owned_start=owned_start,
        owned_end=owned_end,
        slice_width=slice_width,
        direction=-1,
        num_devices=num_gpus,
        capacity=max_values_to_share,
    )
    stay_mask = classes == 1
    send_left_mask = classes == 2
    send_right_mask = classes == 3
    _synchronized_nonzero_check(
        jnp.sum(classes == 4),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        send_left_count,
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )

    auth_size = pmid.shape[0]
    auth_slots = jnp.arange(auth_size, dtype=jnp.int32)
    stay_pos = jnp.compress(
        stay_mask,
        auth_slots,
        axis=0,
        size=auth_size,
        fill_value=jnp.asarray(-1, dtype=jnp.int32),
    )
    stay_valid = jnp.arange(auth_size, dtype=jnp.int32) < jnp.sum(stay_mask)
    send_left_pos = jnp.compress(
        send_left_mask,
        auth_slots,
        axis=0,
        size=max_values_to_share,
        fill_value=jnp.asarray(-1, dtype=jnp.int32),
    )
    send_left_valid = (
        jnp.arange(max_values_to_share, dtype=jnp.int32) < send_left_count
    )

    zero_acc = jnp.zeros_like(acc)
    stay_acc = jnp.compress(
        stay_mask,
        acc,
        axis=0,
        size=auth_size,
        fill_value=jnp.asarray(0, dtype=acc.dtype),
    )
    send_left_acc = jnp.compress(
        send_left_mask,
        acc,
        axis=0,
        size=max_values_to_share,
        fill_value=jnp.asarray(0, dtype=acc.dtype),
    )

    incoming_from_right = jax.lax.ppermute(
        send_left_records, axis_name=AXIS_NAME, perm=left_perm
    )
    incoming_from_right_count = jax.lax.ppermute(
        send_left_count, axis_name=AXIS_NAME, perm=left_perm
    )
    incoming_from_right_acc = jax.lax.ppermute(
        send_left_acc, axis_name=AXIS_NAME, perm=left_perm
    )

    _synchronized_capacity_check(
        jnp.sum(stay_mask) + incoming_from_right_count,
        auth_size,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
    )
    if num_gpus == 2:
        merged_pmid, merged_disp, merged_vel, merged_valid, raw_tag, raw_idx = (
            cuda_route_merge(
                pmid,
                disp,
                vel,
                stay_mask,
                incoming_from_right,
                incoming_from_right_count,
                mesh_shape=mesh_shape,
                capacity=auth_size,
                auxiliary=True,
            )
        )
        merged_valid = merged_valid != 0
        merged_tag = jnp.where(
            merged_valid, jnp.where(raw_tag == 1, jnp.uint8(2), raw_tag), jnp.uint8(3)
        )
        merged_idx = jnp.where(merged_valid, raw_idx, jnp.int32(-1))
        safe_auth_idx = jnp.clip(merged_idx, 0, max(auth_size, 1) - 1)
        safe_share_idx = jnp.clip(merged_idx, 0, max(max_values_to_share, 1) - 1)
        merged_acc = jnp.where(
            (merged_tag == 0).reshape((-1, 1)),
            stay_acc[safe_auth_idx],
            jnp.where(
                (merged_tag == 2).reshape((-1, 1)),
                incoming_from_right_acc[safe_share_idx],
                zero_acc[:auth_size],
            ),
        )
        send_right_pos = jnp.full(
            (max_values_to_share,), -1, dtype=jnp.int32
        )
        send_right_valid = jnp.zeros(
            (max_values_to_share,), dtype=jnp.bool_
        )
        route_aux = (
            stay_pos,
            stay_valid,
            send_left_pos,
            send_left_valid,
            send_right_pos,
            send_right_valid,
            merged_tag,
            merged_idx,
        )
        merged_keys = pmid_to_idx(merged_pmid, conf)
        merged_keys = jnp.where(merged_valid, merged_keys, _key_fill_value(conf))
        return (
            merged_keys,
            merged_pmid.astype(pmid.dtype),
            merged_disp,
            merged_vel,
            merged_acc,
            merged_valid,
        ), route_aux

    send_right_records, send_right_count, _ = cuda_route_pack(
        pmid,
        disp,
        vel,
        valid,
        x_mod,
        global_nmesh=global_nMesh,
        mesh_shape=mesh_shape,
        owned_start=owned_start,
        owned_end=owned_end,
        slice_width=slice_width,
        direction=1,
        num_devices=num_gpus,
        capacity=max_values_to_share,
    )
    _synchronized_capacity_check(
        send_right_count,
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_right_pos = jnp.compress(
        send_right_mask,
        auth_slots,
        axis=0,
        size=max_values_to_share,
        fill_value=jnp.asarray(-1, dtype=jnp.int32),
    )
    send_right_valid = (
        jnp.arange(max_values_to_share, dtype=jnp.int32) < send_right_count
    )
    send_right_acc = jnp.compress(
        send_right_mask,
        acc,
        axis=0,
        size=max_values_to_share,
        fill_value=jnp.asarray(0, dtype=acc.dtype),
    )
    incoming_from_left = jax.lax.ppermute(
        send_right_records, axis_name=AXIS_NAME, perm=right_perm
    )
    incoming_from_left_count = jax.lax.ppermute(
        send_right_count, axis_name=AXIS_NAME, perm=right_perm
    )
    incoming_from_left_acc = jax.lax.ppermute(
        send_right_acc, axis_name=AXIS_NAME, perm=right_perm
    )

    _synchronized_capacity_check(
        jnp.sum(stay_mask) + incoming_from_left_count,
        auth_size,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
    )
    first_pmid, first_disp, first_vel, first_valid, first_raw_tag, first_raw_idx = (
        cuda_route_merge(
            pmid,
            disp,
            vel,
            stay_mask,
            incoming_from_left,
            incoming_from_left_count,
            mesh_shape=mesh_shape,
            capacity=auth_size,
            auxiliary=True,
        )
    )
    first_valid = first_valid != 0
    first_tag = jnp.where(first_valid, first_raw_tag, jnp.uint8(3))
    first_idx = jnp.where(first_valid, first_raw_idx, jnp.int32(-1))
    _synchronized_capacity_check(
        jnp.sum(first_valid) + incoming_from_right_count,
        auth_size,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
    )
    merged_pmid, merged_disp, merged_vel, merged_valid, raw_tag, raw_idx = (
        cuda_route_merge(
            first_pmid,
            first_disp,
            first_vel,
            first_valid,
            incoming_from_right,
            incoming_from_right_count,
            mesh_shape=mesh_shape,
            capacity=auth_size,
            auxiliary=True,
        )
    )
    merged_valid = merged_valid != 0
    safe_first_idx = jnp.clip(raw_idx, 0, max(auth_size, 1) - 1)
    prior_tag = first_tag[safe_first_idx]
    prior_idx = first_idx[safe_first_idx]
    merged_tag = jnp.where(
        merged_valid & (raw_tag == 0),
        prior_tag,
        jnp.where(merged_valid, jnp.uint8(2), jnp.uint8(3)),
    )
    merged_idx = jnp.where(
        merged_valid & (raw_tag == 0), prior_idx,
        jnp.where(merged_valid, raw_idx, jnp.int32(-1)),
    )
    safe_auth_idx = jnp.clip(merged_idx, 0, max(auth_size, 1) - 1)
    safe_share_idx = jnp.clip(merged_idx, 0, max(max_values_to_share, 1) - 1)
    merged_acc = jnp.where(
        (merged_tag == 0).reshape((-1, 1)),
        stay_acc[safe_auth_idx],
        jnp.where(
            (merged_tag == 1).reshape((-1, 1)),
            incoming_from_left_acc[safe_share_idx],
            jnp.where(
                (merged_tag == 2).reshape((-1, 1)),
                incoming_from_right_acc[safe_share_idx],
                zero_acc[:auth_size],
            ),
        ),
    )
    merged_keys = pmid_to_idx(merged_pmid, conf)
    merged_keys = jnp.where(merged_valid, merged_keys, _key_fill_value(conf))
    route_aux = (
        stay_pos,
        stay_valid,
        send_left_pos,
        send_left_valid,
        send_right_pos,
        send_right_valid,
        merged_tag,
        merged_idx,
    )
    return (
        merged_keys,
        merged_pmid.astype(pmid.dtype),
        merged_disp,
        merged_vel,
        merged_acc,
        merged_valid,
    ), route_aux


def _canonical_route_authoritative_with_aux(
    keys,
    pmid,
    disp,
    vel,
    acc,
    valid,
    global_nMesh,
    max_values_to_share,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Route authoritative particles and save the data needed by the transpose."""
    if cuda_routing_enabled(conf):
        return _canonical_route_authoritative_with_aux_cuda(
            keys,
            pmid,
            disp,
            vel,
            acc,
            valid,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
    owned_start, owned_end = _owned_slice_bounds(global_nMesh, num_gpus, offsets)
    slice_width = global_nMesh // num_gpus
    left_start = (owned_start - slice_width) % global_nMesh
    right_end = (owned_end + slice_width) % global_nMesh

    x_mod = _x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
    stay_mask = valid & particles_in_slice_mask(x_mod, owned_start, owned_end)
    send_left_mask = valid & particles_in_slice_mask(x_mod, left_start, owned_start)
    send_right_mask = valid & particles_in_slice_mask(x_mod, owned_end, right_end)
    if num_gpus == 2:
        send_right_mask = jnp.zeros_like(send_right_mask)

    dropped_mask = valid & ~(stay_mask | send_left_mask | send_right_mask)
    _synchronized_nonzero_check(
        jnp.sum(dropped_mask),
        "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
        "particles_outside_neighbor_range={x}.",
    )
    _synchronized_capacity_check(
        jnp.maximum(jnp.sum(send_left_mask), jnp.sum(send_right_mask)),
        max_values_to_share,
        "[ERROR] Exceeded migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )

    key_fill = _key_fill_value(conf)
    original = (keys, pmid, disp, vel, acc, valid)
    *send_left_items, send_left_pos = _compact_sorted_particles_with_slots(
        keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share,
        key_fill,
        "[ERROR] Exceeded left-migration share capacity. "
        "particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_left = tuple(send_left_items)
    if num_gpus == 2:
        incoming_from_right = _exchange_compacted_particles(send_left, left_perm, conf)
        merged, stay_pos, stay_valid = _sparse_route_merge_two(
            original, send_left, send_left_pos, incoming_from_right, pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
            provenance=True, incoming_tag=jnp.int32(2),
        )
        send_right_pos = jnp.full((max_values_to_share,), -1, dtype=jnp.int32)
        send_right_valid = jnp.zeros((max_values_to_share,), dtype=jnp.bool_)
    else:
        *send_right_items, send_right_pos = _compact_sorted_particles_with_slots(
            keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share,
            key_fill,
            "[ERROR] Exceeded right-migration share capacity. "
            "particles_to_share={x}, max_share_ptcl={y}.",
        )
        send_right = tuple(send_right_items)
        incoming_from_left = _exchange_compacted_particles(send_right, right_perm, conf)
        incoming_from_right = _exchange_compacted_particles(send_left, left_perm, conf)
        merged, stay_pos, stay_valid = _sparse_route_merge_three(
            original, send_left, send_left_pos, send_right, send_right_pos,
            incoming_from_left, incoming_from_right, pmid.shape[0], key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
            provenance=True,
        )
        send_right_valid = send_right[-1]

    route_aux = (
        stay_pos, stay_valid, send_left_pos, send_left[-1], send_right_pos,
        send_right_valid, merged[-2], merged[-1],
    )
    return merged[:6], route_aux

def _reverse_build_full_cot(
    full_cot,
    auth_pmid,
    auth_disp,
    auth_valid,
    halo_end,
    max_ptcl_per_slice,
    max_halo_values_to_share,
    global_nMesh,
    left_perm,
    right_perm,
    disp_size,
):
    """Transpose ``_canonical_build_full_from_authoritative`` for one payload."""
    auth_pos = jnp.arange(auth_pmid.shape[0], dtype=jnp.int32)
    x_mod = _x_mod_from_disp(auth_pmid, auth_disp, global_nMesh, disp_size)
    right_halo_mask = auth_valid & particles_in_slice_mask(
        x_mod, halo_end.squeeze()[0], halo_end.squeeze()[1]
    )
    right_halo_pos = jnp.compress(
        right_halo_mask,
        auth_pos,
        axis=0,
        size=max_halo_values_to_share,
        fill_value=jnp.asarray(-1, auth_pos.dtype),
    )
    right_halo_valid = jnp.arange(max_halo_values_to_share) < jnp.sum(right_halo_mask)
    left_halo_valid = jax.lax.ppermute(right_halo_valid, axis_name=AXIS_NAME, perm=right_perm)

    left_count = jnp.sum(left_halo_valid)
    auth_count = jnp.sum(auth_valid)
    slots = jnp.arange(max_ptcl_per_slice, dtype=jnp.int32)
    left_mask = slots < left_count
    auth_mask = (slots >= left_count) & (slots < (left_count + auth_count))

    left_cot = jnp.compress(
        left_mask,
        full_cot,
        axis=0,
        size=max_halo_values_to_share,
        fill_value=jnp.asarray(0, full_cot.dtype),
    )
    auth_cot = jnp.compress(
        auth_mask,
        full_cot,
        axis=0,
        size=auth_pmid.shape[0],
        fill_value=jnp.asarray(0, full_cot.dtype),
    )

    outbound_right_cot = jax.lax.ppermute(left_cot, axis_name=AXIS_NAME, perm=left_perm)
    valid_mask = right_halo_valid.reshape((right_halo_valid.shape[0],) + (1,) * (full_cot.ndim - 1))
    auth_cot = auth_cot.at[jnp.where(right_halo_valid, right_halo_pos, 0)].add(
        outbound_right_cot * valid_mask.astype(full_cot.dtype)
    )
    return auth_cot


def _reverse_route_cot(
    merged_cot,
    stay_pos,
    stay_valid,
    send_left_pos,
    send_left_valid,
    send_right_pos,
    send_right_valid,
    merge_src_tag,
    merge_src_idx,
    auth_size,
    max_values_to_share,
    left_perm,
    right_perm,
    conf=None,
):
    """Transpose the authoritative particle migration route."""
    if conf is not None and cuda_routing_enabled(conf):
        stay_cot, incoming_from_left_cot, incoming_from_right_cot = (
            cuda_route_transpose_split(
                merged_cot,
                merge_src_tag,
                merge_src_idx,
                auth_size=auth_size,
                share_capacity=max_values_to_share,
            )
        )
        send_right_cot = jax.lax.ppermute(
            incoming_from_left_cot, axis_name=AXIS_NAME, perm=left_perm
        )
        send_left_cot = jax.lax.ppermute(
            incoming_from_right_cot, axis_name=AXIS_NAME, perm=right_perm
        )
        return cuda_route_transpose_scatter(
            stay_cot,
            send_left_cot,
            send_right_cot,
            stay_pos,
            stay_valid,
            send_left_pos,
            send_left_valid,
            send_right_pos,
            send_right_valid,
            auth_size=auth_size,
            share_capacity=max_values_to_share,
        )

    dtype = merged_cot.dtype
    cot_shape = merged_cot.shape[1:]
    stay_cot = jnp.zeros((stay_pos.shape[0],) + cot_shape, dtype=dtype)
    incoming_from_left_cot = jnp.zeros((max_values_to_share,) + cot_shape, dtype=dtype)
    incoming_from_right_cot = jnp.zeros((max_values_to_share,) + cot_shape, dtype=dtype)

    stay_mask = merge_src_tag == 0
    incoming_left_mask = merge_src_tag == 1
    incoming_right_mask = merge_src_tag == 2
    broadcast_shape = (merged_cot.shape[0],) + (1,) * (merged_cot.ndim - 1)
    stay_scale = stay_mask.reshape(broadcast_shape).astype(dtype)
    incoming_left_scale = incoming_left_mask.reshape(broadcast_shape).astype(dtype)
    incoming_right_scale = incoming_right_mask.reshape(broadcast_shape).astype(dtype)

    stay_cot = stay_cot.at[jnp.where(stay_mask, merge_src_idx, 0)].add(
        merged_cot * stay_scale
    )
    incoming_from_left_cot = incoming_from_left_cot.at[jnp.where(incoming_left_mask, merge_src_idx, 0)].add(
        merged_cot * incoming_left_scale
    )
    incoming_from_right_cot = incoming_from_right_cot.at[jnp.where(incoming_right_mask, merge_src_idx, 0)].add(
        merged_cot * incoming_right_scale
    )

    send_right_cot = jax.lax.ppermute(incoming_from_left_cot, axis_name=AXIS_NAME, perm=left_perm)
    send_left_cot = jax.lax.ppermute(incoming_from_right_cot, axis_name=AXIS_NAME, perm=right_perm)

    auth_cot = jnp.zeros((auth_size,) + cot_shape, dtype=dtype)
    stay_valid_scale = stay_valid.reshape((stay_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)).astype(dtype)
    send_left_valid_scale = send_left_valid.reshape((send_left_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)).astype(dtype)
    send_right_valid_scale = send_right_valid.reshape((send_right_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)).astype(dtype)
    auth_cot = auth_cot.at[jnp.where(stay_valid, stay_pos, 0)].add(
        stay_cot * stay_valid_scale
    )
    auth_cot = auth_cot.at[jnp.where(send_left_valid, send_left_pos, 0)].add(
        send_left_cot * send_left_valid_scale
    )
    auth_cot = auth_cot.at[jnp.where(send_right_valid, send_right_pos, 0)].add(
        send_right_cot * send_right_valid_scale
    )
    return auth_cot


def _reverse_route_cot_two_gpu(
    merged_cot,
    stay_pos,
    stay_valid,
    send_left_pos,
    send_left_valid,
    _send_right_pos,
    _send_right_valid,
    merge_src_tag,
    merge_src_idx,
    auth_size,
    max_values_to_share,
    left_perm,
    right_perm,
    conf=None,
):
    """2-GPU fast path for _reverse_route_cot.

    In the 2-GPU topology send_right is always zero, so merge_src_tag never
    equals 1 (incoming_from_left). We skip the zero-ppermute on that side
    and the associated scatter.
    """
    if conf is not None and cuda_routing_enabled(conf):
        stay_cot, _incoming_from_left_cot, incoming_from_right_cot = (
            cuda_route_transpose_split(
                merged_cot,
                merge_src_tag,
                merge_src_idx,
                auth_size=auth_size,
                share_capacity=max_values_to_share,
            )
        )
        send_left_cot = jax.lax.ppermute(
            incoming_from_right_cot, axis_name=AXIS_NAME, perm=right_perm
        )
        zero_send_right = jnp.zeros_like(send_left_cot)
        return cuda_route_transpose_scatter(
            stay_cot,
            send_left_cot,
            zero_send_right,
            stay_pos,
            stay_valid,
            send_left_pos,
            send_left_valid,
            _send_right_pos,
            _send_right_valid,
            auth_size=auth_size,
            share_capacity=max_values_to_share,
        )

    dtype = merged_cot.dtype
    cot_shape = merged_cot.shape[1:]
    stay_cot = jnp.zeros((stay_pos.shape[0],) + cot_shape, dtype=dtype)
    incoming_from_right_cot = jnp.zeros((max_values_to_share,) + cot_shape, dtype=dtype)

    stay_mask = merge_src_tag == 0
    incoming_right_mask = merge_src_tag == 2
    broadcast_shape = (merged_cot.shape[0],) + (1,) * (merged_cot.ndim - 1)
    stay_scale = stay_mask.reshape(broadcast_shape).astype(dtype)
    incoming_right_scale = incoming_right_mask.reshape(broadcast_shape).astype(dtype)

    stay_cot = stay_cot.at[jnp.where(stay_mask, merge_src_idx, 0)].add(
        merged_cot * stay_scale
    )
    incoming_from_right_cot = incoming_from_right_cot.at[
        jnp.where(incoming_right_mask, merge_src_idx, 0)
    ].add(merged_cot * incoming_right_scale)

    # In the 2-GPU case only incoming_from_right exists; its cotangents must
    # travel back to the source via right_perm (reverse of the left_perm that
    # was used to send send_left -> incoming_from_right in the forward).
    send_left_cot = jax.lax.ppermute(incoming_from_right_cot, axis_name=AXIS_NAME, perm=right_perm)

    auth_cot = jnp.zeros((auth_size,) + cot_shape, dtype=dtype)
    stay_valid_scale = stay_valid.reshape(
        (stay_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)
    ).astype(dtype)
    send_left_valid_scale = send_left_valid.reshape(
        (send_left_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)
    ).astype(dtype)
    auth_cot = auth_cot.at[jnp.where(stay_valid, stay_pos, 0)].add(
        stay_cot * stay_valid_scale
    )
    auth_cot = auth_cot.at[jnp.where(send_left_valid, send_left_pos, 0)].add(
        send_left_cot * send_left_valid_scale
    )
    return auth_cot


def halo_move_pullback_from_prestate_shard_map(
    pmid,
    source_disp,
    carried_disp,
    vel,
    acc,
    halo_end,
    unused_indexes,
    disp_cot,
    vel_cot,
    acc_cot,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Pull cotangents through a canonical ``particle_halo`` move.

    Parameters
    ----------
    pmid, source_disp, carried_disp, vel, acc : jax.Array
        Pre-move particle state. ``source_disp`` is the authoritative
        pre-migration displacement while ``carried_disp`` is the transported
        post-drift displacement used to rebuild the moved state.
    halo_end : jax.Array
        Upper halo bound for the current shard.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    disp_cot, vel_cot, acc_cot : jax.Array
        Cotangents with respect to the moved displacement, velocity, and
        acceleration arrays.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration, halo rebuild, and per-slab storage.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the move.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Cotangents for the input displacement, velocity, and acceleration
        buffers before the canonical halo move.
    """
    # Reverse the canonical move in two logical stages:
    # 1. recover the authoritative sequence before the move,
    # 2. transpose the deterministic route/build back to the original slots.
    (
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        auth_slots,
    ) = _canonical_authoritative_from_full_with_slots(
        pmid,
        source_disp,
        carried_disp,
        vel,
        acc,
        unused_indexes,
        global_nMesh,
        disp_size,
        num_gpus,
        offsets,
        conf,
    )
    (
        _merged_keys,
        merged_pmid,
        merged_disp,
        _merged_vel,
        _merged_acc,
        merged_valid,
    ), route_aux = _canonical_route_authoritative_with_aux(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        global_nMesh,
        max_values_to_share,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    )
    (
        stay_pos,
        stay_valid,
        send_left_pos,
        send_left_valid,
        send_right_pos,
        send_right_valid,
        merge_src_tag,
        merge_src_idx,
    ) = route_aux

    payload_cot = jnp.stack((disp_cot, vel_cot, acc_cot), axis=-1)
    merged_payload_cot = _reverse_build_full_cot(
        payload_cot,
        merged_pmid,
        merged_disp,
        merged_valid,
        halo_end,
        max_ptcl_per_slice,
        max_halo_values_to_share,
        global_nMesh,
        left_perm,
        right_perm,
        disp_size,
    )

    _reverse_route_fn = _reverse_route_cot_two_gpu if num_gpus == 2 else _reverse_route_cot
    auth_payload_cot = _reverse_route_fn(
        merged_payload_cot,
        stay_pos,
        stay_valid,
        send_left_pos,
        send_left_valid,
        send_right_pos,
        send_right_valid,
        merge_src_tag,
        merge_src_idx,
        auth_pmid.shape[0],
        max_values_to_share,
        left_perm,
        right_perm,
        conf,
    )

    input_payload_cot = _scatter_compact_to_dense(
        auth_payload_cot,
        auth_slots,
        auth_valid,
        pmid.shape[0],
    )
    return input_payload_cot[..., 0], input_payload_cot[..., 1], input_payload_cot[..., 2]


def halo_move_pullback_mesh_halo_from_prestate_shard_map(
    pmid,
    source_disp,
    carried_disp,
    vel,
    acc,
    halo_end,
    unused_indexes,
    disp_cot,
    vel_cot,
    acc_cot,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Pull cotangents through a ``mesh_halo`` authoritative-only move.

    Parameters
    ----------
    pmid, source_disp, carried_disp, vel, acc : jax.Array
        Pre-move authoritative particle state. ``source_disp`` is accepted for
        API parity with the canonical variant but is not used by this
        authoritative-only reconstruction.
    halo_end : jax.Array
        Unused placeholder kept for a shared call signature with the canonical
        path.
    unused_indexes : jax.Array
        Boolean mask marking inactive authoritative slots.
    disp_cot, vel_cot, acc_cot : jax.Array
        Cotangents with respect to the moved displacement, velocity, and
        acceleration buffers.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities passed through the shared initialization path. Only
        ``max_values_to_share`` is used directly here.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the move.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Cotangents for the input displacement, velocity, and acceleration
        buffers before the mesh-halo move.
    """
    del halo_end, max_halo_values_to_share, max_ptcl_per_slice
    (
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
    ) = _authoritative_prefix_from_owned_only(
        pmid,
        carried_disp,
        vel,
        acc,
        unused_indexes,
        conf,
    )
    (
        _merged_keys,
        _merged_pmid,
        _merged_disp,
        _merged_vel,
        _merged_acc,
        merged_valid,
    ), route_aux = _canonical_route_authoritative_with_aux(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        global_nMesh,
        max_values_to_share,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    )
    (
        stay_pos,
        stay_valid,
        send_left_pos,
        send_left_valid,
        send_right_pos,
        send_right_valid,
        merge_src_tag,
        merge_src_idx,
    ) = route_aux

    payload_cot = jnp.stack((disp_cot, vel_cot, acc_cot), axis=-1)
    merged_payload_cot = _reverse_build_owned_only_cot(
        payload_cot,
        auth_pmid.shape[0],
        merged_valid,
    )
    _reverse_route_fn = _reverse_route_cot_two_gpu if num_gpus == 2 else _reverse_route_cot
    auth_payload_cot = _reverse_route_fn(
        merged_payload_cot,
        stay_pos,
        stay_valid,
        send_left_pos,
        send_left_valid,
        send_right_pos,
        send_right_valid,
        merge_src_tag,
        merge_src_idx,
        auth_pmid.shape[0],
        max_values_to_share,
        left_perm,
        right_perm,
        conf,
    )
    input_payload_cot = _mask_compact_prefix(auth_payload_cot, auth_valid)
    return input_payload_cot[..., 0], input_payload_cot[..., 1], input_payload_cot[..., 2]


def _canonical_build_full_from_authoritative(
    auth_keys,
    auth_pmid,
    auth_disp,
    auth_vel,
    auth_acc,
    auth_valid,
    halo_start,
    halo_end,
    max_ptcl_per_slice,
    max_halo_values_to_share,
    global_nMesh,
    right_perm,
    disp_size,
    conf,
):
    """Rebuild deterministic ``particle_halo`` storage from authoritative particles."""
    # The stored particle slab is deterministic:
    # 1. authoritative owned particles,
    # 2. exported right-edge particles mirrored from the neighbor as left halo.
    x_mod = _x_mod_from_disp(auth_pmid, auth_disp, global_nMesh, disp_size)
    right_halo_mask = auth_valid & particles_in_slice_mask(
        x_mod, halo_end.squeeze()[0], halo_end.squeeze()[1]
    )
    key_fill = _key_fill_value(conf)
    outbound_right_halo = _compact_sorted_particles(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        right_halo_mask,
        max_halo_values_to_share,
        key_fill,
        "[ERROR] Exceeded halo-share capacity while rebuilding canonical storage. "
        "particles_to_share={x}, max_halo_share_ptcl={y}.",
    )
    incoming_left_halo = jax.lax.ppermute(outbound_right_halo, axis_name=AXIS_NAME, perm=right_perm)
    return _pack_left_halo_and_authoritative(
        *incoming_left_halo,
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        max_ptcl_per_slice,
        halo_start,
        halo_end,
        global_nMesh,
        disp_size,
    )


def move_particles_canonical_shard_map(
    pmid,
    disp_before,
    disp_after,
    vel,
    acc,
    halo_start,
    halo_end,
    unused_indexes,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Move particles across slabs and rebuild duplicated particle-halo storage.

    Parameters
    ----------
    pmid, disp_before, disp_after, vel, acc : jax.Array
        Canonical particle-halo storage before migration and the authoritative
        displacement after the drift that may cross slab boundaries.
    halo_start, halo_end : jax.Array
        Halo interval for the local shard.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers, duplicated halo rebuilds, and
        final per-shard particle storage.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the move.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple
        Updated ``(pmid, disp, vel, acc, halo_mask, unused_indexes,
        overflow_flag, max_particles_moved)`` for the next particle-halo step.
    """
    # Forward halo move:
    # 1. drop duplicated storage and keep the authoritative slab only,
    # 2. reroute that slab based on post-drift positions,
    # 3. rebuild the deterministic duplicated storage for the next step.
    auth = _canonical_authoritative_from_full(
        pmid,
        disp_before,
        disp_after,
        vel,
        acc,
        unused_indexes,
        global_nMesh,
        disp_size,
        num_gpus,
        offsets,
        conf,
    )
    (auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid), max_particles_moved = (
        _canonical_route_authoritative(
            *auth,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
    )
    pmid, disp, vel, acc, halo_mask, unused_indexes = _canonical_build_full_from_authoritative(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        halo_start,
        halo_end,
        max_ptcl_per_slice,
        max_halo_values_to_share,
        global_nMesh,
        right_perm,
        disp_size,
        conf,
    )
    return pmid, disp, vel, acc, halo_mask, unused_indexes, jnp.bool_(False), max_particles_moved


def move_particles_mesh_halo_shard_map(
    pmid,
    disp_before,
    disp_after,
    vel,
    acc,
    halo_start,
    halo_end,
    unused_indexes,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Move particles across slabs while storing only authoritative particles.

    Parameters
    ----------
    pmid, disp_before, disp_after, vel, acc : jax.Array
        Authoritative-only mesh-halo particle state before migration and the
        post-drift displacement used to determine new ownership.
    halo_start, halo_end : jax.Array
        Unused placeholders retained for signature compatibility with the
        canonical mover.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers and authoritative per-shard
        storage. ``max_halo_values_to_share`` is unused in this mode.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the move.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple
        Updated ``(pmid, disp, vel, acc, halo_mask, unused_indexes,
        overflow_flag, max_particles_moved)`` for the next mesh-halo step.
    """
    del disp_before, halo_start, halo_end, max_halo_values_to_share
    auth = _authoritative_prefix_from_owned_only(
        pmid,
        disp_after,
        vel,
        acc,
        unused_indexes,
        conf,
    )
    (_auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid), max_particles_moved = (
        _canonical_route_authoritative(
            *auth,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
    )
    pmid, disp, vel, acc, halo_mask, unused_indexes = _pack_authoritative_only(
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        max_ptcl_per_slice,
    )
    return pmid, disp, vel, acc, halo_mask, unused_indexes, jnp.bool_(False), max_particles_moved


def move_particles_mesh_halo_no_acc_shard_map(
    pmid,
    disp_before,
    disp_after,
    vel,
    halo_start,
    halo_end,
    unused_indexes,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Move mesh-halo particles for a drift that is immediately followed by force.

    Parameters
    ----------
    pmid, disp_before, disp_after, vel : jax.Array
        Authoritative-only particle state before migration and the post-drift
        displacement used to determine new ownership.
    halo_start, halo_end : jax.Array
        Unused placeholders retained for signature compatibility with the full
        mover.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers and authoritative per-shard
        storage. ``max_halo_values_to_share`` is unused in this mode.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the move.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    The following force overwrites acceleration, so this path routes only the
    position and velocity payload while preserving the same sorted authoritative
    particle order as the full canonical mover.

    Returns
    -------
    tuple
        Updated ``(pmid, disp, vel, halo_mask, unused_indexes, overflow_flag,
        max_particles_moved)`` for the next force-producing step.
    """
    del disp_before, halo_start, halo_end, max_halo_values_to_share
    auth = _authoritative_prefix_from_owned_only_no_acc(
        pmid,
        disp_after,
        vel,
        unused_indexes,
        conf,
    )
    (_auth_keys, auth_pmid, auth_disp, auth_vel, auth_valid), max_particles_moved = (
        _canonical_route_authoritative_no_acc(
            *auth,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
    )
    pmid, disp, vel, halo_mask, unused_indexes = _pack_authoritative_only_no_acc(
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_valid,
        max_ptcl_per_slice,
    )
    return pmid, disp, vel, halo_mask, unused_indexes, jnp.bool_(False), max_particles_moved


def reconstruct_pre_drift_canonical_shard_map(
    pmid,
    disp,
    vel,
    acc,
    halo_start,
    halo_end,
    unused_indexes,
    drift_factor,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Reconstruct the canonical pre-drift particle-halo state from post-drift data.

    Parameters
    ----------
    pmid, disp, vel, acc : jax.Array
        Post-drift canonical particle-halo storage.
    halo_start, halo_end : jax.Array
        Halo interval for the local shard.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    drift_factor : jax.Array
        Scalar displacement multiplier used by the drift step being inverted.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers, halo rebuilds, and final
        particle storage.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the reconstruction.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple
        Reconstructed ``(pmid, disp, vel, acc, unused_indexes, halo_mask)``
        immediately before the drift.
    """
    auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid = _canonical_authoritative_from_full(
        pmid,
        disp,
        disp,
        vel,
        acc,
        unused_indexes,
        global_nMesh,
        disp_size,
        num_gpus,
        offsets,
        conf,
    )
    auth_disp = auth_disp - auth_vel * drift_factor.astype(auth_disp.dtype)
    (auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid), _ = _canonical_route_authoritative(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        global_nMesh,
        max_values_to_share,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    )
    pmid, disp, vel, acc, halo_mask, unused_index = _canonical_build_full_from_authoritative(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        halo_start,
        halo_end,
        max_ptcl_per_slice,
        max_halo_values_to_share,
        global_nMesh,
        right_perm,
        disp_size,
        conf,
    )
    return pmid, disp, vel, acc, unused_index, halo_mask


def reconstruct_pre_drift_mesh_halo_shard_map(
    pmid,
    disp,
    vel,
    acc,
    halo_start,
    halo_end,
    unused_indexes,
    drift_factor,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Reconstruct the pre-drift authoritative-only state from post-drift data.

    Parameters
    ----------
    pmid, disp, vel, acc : jax.Array
        Post-drift authoritative mesh-halo storage.
    halo_start, halo_end : jax.Array
        Unused placeholders retained for signature compatibility with the
        canonical reconstruction.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    drift_factor : jax.Array
        Scalar displacement multiplier used by the drift step being inverted.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers and authoritative particle
        storage. ``max_halo_values_to_share`` is unused in this mode.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the reconstruction.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple
        Reconstructed ``(pmid, disp, vel, acc, unused_indexes, halo_mask)``
        immediately before the drift.
    """
    del halo_start, halo_end, max_halo_values_to_share
    auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid = _authoritative_prefix_from_owned_only(
        pmid,
        disp,
        vel,
        acc,
        unused_indexes,
        conf,
    )
    auth_disp = auth_disp - auth_vel * drift_factor.astype(auth_disp.dtype)
    (_auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid), _ = _canonical_route_authoritative(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        global_nMesh,
        max_values_to_share,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    )
    pmid, disp, vel, acc, halo_mask, unused_index = _pack_authoritative_only(
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        max_ptcl_per_slice,
    )
    return pmid, disp, vel, acc, unused_index, halo_mask


def reconstruct_pre_drift_and_pullback_mesh_halo_shard_map(
    pmid,
    disp,
    vel,
    acc,
    unused_indexes,
    drift_factor,
    disp_cot,
    vel_cot,
    acc_cot,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Fused mesh-halo reconstruction plus halo-move pullback for drift adjoints.

    Parameters
    ----------
    pmid, disp, vel, acc : jax.Array
        Post-drift authoritative particle state.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    drift_factor : jax.Array
        Scalar displacement multiplier used by the drift step being inverted.
    disp_cot, vel_cot, acc_cot : jax.Array
        Cotangents arriving at the post-drift state.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers and authoritative particle
        storage.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the reconstruction.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple
        Reconstructed pre-drift state followed by the pulled-back cotangents:
        ``(pmid, disp, vel, acc, unused_indexes, halo_mask, disp_cot,
        vel_cot, acc_cot)``.
    """
    pre_pmid, pre_disp, pre_vel, pre_acc, pre_unused_index, pre_halo_mask = (
        reconstruct_pre_drift_mesh_halo_shard_map(
            pmid,
            disp,
            vel,
            acc,
            conf.halo_start,
            conf.halo_end,
            unused_indexes,
            drift_factor,
            global_nMesh,
            max_values_to_share,
            max_halo_values_to_share,
            max_ptcl_per_slice,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
    )
    disp_before_halo = pre_disp + pre_vel * drift_factor.astype(pre_disp.dtype)
    disp_pullback, vel_pullback, acc_pullback = halo_move_pullback_mesh_halo_from_prestate_shard_map(
        pre_pmid,
        pre_disp,
        disp_before_halo,
        pre_vel,
        pre_acc,
        conf.halo_end,
        pre_unused_index,
        disp_cot,
        vel_cot,
        acc_cot,
        global_nMesh,
        max_values_to_share,
        max_halo_values_to_share,
        max_ptcl_per_slice,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    )
    return (
        pre_pmid,
        pre_disp,
        pre_vel,
        pre_acc,
        pre_unused_index,
        pre_halo_mask,
        disp_pullback,
        vel_pullback,
        acc_pullback,
    )


def reconstruct_pre_drift_and_pullback_canonical_shard_map(
    pmid,
    disp,
    vel,
    acc,
    unused_indexes,
    drift_factor,
    disp_cot,
    vel_cot,
    acc_cot,
    global_nMesh,
    max_values_to_share,
    max_halo_values_to_share,
    max_ptcl_per_slice,
    left_perm,
    right_perm,
    num_gpus,
    disp_size,
    offsets,
    conf,
):
    """Fused particle-halo reconstruction plus halo-move pullback for drift adjoints.

    Parameters
    ----------
    pmid, disp, vel, acc : jax.Array
        Post-drift canonical particle-halo storage.
    unused_indexes : jax.Array
        Boolean mask marking inactive particle slots.
    drift_factor : jax.Array
        Scalar displacement multiplier used by the drift step being inverted.
    disp_cot, vel_cot, acc_cot : jax.Array
        Cotangents arriving at the post-drift state.
    global_nMesh : int
        Global mesh resolution along the decomposed axis.
    max_values_to_share, max_halo_values_to_share, max_ptcl_per_slice : int
        Static capacities for migration buffers, halo rebuilds, and final
        particle storage.
    left_perm, right_perm : tuple
        ``ppermute`` routes describing left and right neighbor exchanges.
    num_gpus : int
        Number of shards participating in the reconstruction.
    disp_size : float
        Physical size of one mesh cell.
    offsets : tuple[float, float]
        Offsets used to sort wrapped coordinates deterministically.
    conf : Configuration
        Active runtime configuration.

    Returns
    -------
    tuple
        Reconstructed pre-drift state followed by the pulled-back cotangents:
        ``(pmid, disp, vel, acc, unused_indexes, halo_mask, disp_cot,
        vel_cot, acc_cot)``.
    """
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    halo_start = conf.halo_start[gpu_id]
    halo_end = conf.halo_end[gpu_id]
    pre_pmid, pre_disp, pre_vel, pre_acc, pre_unused_index, pre_halo_mask = (
        reconstruct_pre_drift_canonical_shard_map(
            pmid,
            disp,
            vel,
            acc,
            halo_start,
            halo_end,
            unused_indexes,
            drift_factor,
            global_nMesh,
            max_values_to_share,
            max_halo_values_to_share,
            max_ptcl_per_slice,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
    )
    disp_before_halo = pre_disp + pre_vel * drift_factor.astype(pre_disp.dtype)
    disp_pullback, vel_pullback, acc_pullback = halo_move_pullback_from_prestate_shard_map(
        pre_pmid,
        pre_disp,
        disp_before_halo,
        pre_vel,
        pre_acc,
        halo_end,
        pre_unused_index,
        disp_cot,
        vel_cot,
        acc_cot,
        global_nMesh,
        max_values_to_share,
        max_halo_values_to_share,
        max_ptcl_per_slice,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    )
    return (
        pre_pmid,
        pre_disp,
        pre_vel,
        pre_acc,
        pre_unused_index,
        pre_halo_mask,
        disp_pullback,
        vel_pullback,
        acc_pullback,
    )


@partial(jax.jit, static_argnames=["global_nMesh", "disp_size"])
def compute_halo_mask_shard_map(pmid, disp, unused_indexes, halo_start, halo_end, global_nMesh, disp_size):
    """Compute halo masks from sharded particle positions.

    Parameters
    ----------
    pmid : jax.Array
        Particle mesh ids on one shard.
    disp : jax.Array
        Particle displacements on one shard.
    unused_indexes : jax.Array
        Boolean padding mask.
    halo_start, halo_end : jax.Array
        Halo-band bounds for the current shard.
    global_nMesh : int
        Global mesh size along the decomposed axis.
    disp_size : float
        Inverse mesh-cell size used to convert displacements into cell units.

    Returns
    -------
    jax.Array
        Boolean halo mask for the shard.
    """
    x_mod = (pmid[:, 0] + disp[:, 0] * disp_size) % global_nMesh
    return compute_halo_mask(x_mod, halo_start.squeeze(), halo_end.squeeze(), unused_indexes)


def _halo_capacity(conf):
    """Return the static capacity for halo-copy exchange buffers."""
    if conf.max_halo_share_ptcl is not None:
        return conf.max_halo_share_ptcl
    return min(
        conf.max_ptcl_per_slice,
        (conf.max_ptcl_per_slice * conf.ptcl_halo_width + conf.local_mesh_shape[0] - 1)
        // conf.local_mesh_shape[0],
    )


def initialize_mGPU_halo_movement_canonical(conf):
    """Create the sharded forward particle-movement callable.

    Parameters
    ----------
    conf : Configuration
        Configuration defining the active multi-GPU mode and capacities.

    Returns
    -------
    callable
        Sharded particle-routing callable specialized to the active mode.
    """
    if conf.num_devices == 1:
        def _halo_noop(pmid, disp_before, disp_after, vel, acc, halo_start, halo_end, unused_indexes):
            del disp_before, halo_start, halo_end
            return (
                pmid,
                disp_after,
                vel,
                acc,
                jnp.zeros_like(unused_indexes),
                unused_indexes,
                jnp.bool_(False),
                jnp.int32(0),
            )
        return _halo_noop

    move_fn = move_particles_canonical_shard_map
    if conf.multigpu_mode == "mesh_halo":
        move_fn = move_particles_mesh_halo_shard_map

    func = partial(
        move_fn,
        global_nMesh=conf.nMesh,
        max_values_to_share=conf.max_share_ptcl,
        max_halo_values_to_share=_halo_capacity(conf),
        max_ptcl_per_slice=conf.max_ptcl_per_slice,
        left_perm=conf.left_perm,
        right_perm=conf.right_perm,
        num_gpus=conf.num_devices,
        disp_size=conf.disp_size,
        offsets=conf.offsets,
        conf=conf,
    )
    return shard_map(
        func,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(AXIS_NAME),
        ),
        out_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(),
            P(),
        ),
        check_rep=False,
    )


def initialize_mGPU_halo_movement_no_acc(conf):
    """Create the sharded mesh-halo mover for drifts followed immediately by force.

    Parameters
    ----------
    conf : Configuration
        Configuration defining the active multi-GPU mode and capacities.

    Returns
    -------
    callable or None
        Specialized mover that skips carrying acceleration, or ``None`` when
        the optimization does not apply.
    """
    if conf.num_devices == 1 or conf.multigpu_mode != "mesh_halo":
        return None

    func = partial(
        move_particles_mesh_halo_no_acc_shard_map,
        global_nMesh=conf.nMesh,
        max_values_to_share=conf.max_share_ptcl,
        max_halo_values_to_share=_halo_capacity(conf),
        max_ptcl_per_slice=conf.max_ptcl_per_slice,
        left_perm=conf.left_perm,
        right_perm=conf.right_perm,
        num_gpus=conf.num_devices,
        disp_size=conf.disp_size,
        offsets=conf.offsets,
        conf=conf,
    )
    return shard_map(
        func,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(AXIS_NAME),
        ),
        out_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(),
            P(),
        ),
        check_rep=False,
    )


def initialize_mGPU_reconstruct_pre_drift(conf):
    """Create the sharded pre-drift reconstruction callable.

    Parameters
    ----------
    conf : Configuration
        Configuration defining the active multi-GPU mode and capacities.

    Returns
    -------
    callable
        Reconstruction callable used by the drift adjoint.
    """
    if conf.num_devices == 1:
        def _reconstruct_noop(pmid, disp, vel, acc, halo_start, halo_end, unused_indexes, drift_factor):
            del halo_start, halo_end, drift_factor
            halo_mask = jnp.zeros_like(unused_indexes)
            return pmid, disp, vel, acc, unused_indexes, halo_mask
        return _reconstruct_noop

    reconstruct_fn = reconstruct_pre_drift_canonical_shard_map
    if conf.multigpu_mode == "mesh_halo":
        reconstruct_fn = reconstruct_pre_drift_mesh_halo_shard_map

    func = partial(
        reconstruct_fn,
        global_nMesh=conf.nMesh,
        max_values_to_share=conf.max_share_ptcl,
        max_halo_values_to_share=_halo_capacity(conf),
        max_ptcl_per_slice=conf.max_ptcl_per_slice,
        left_perm=conf.left_perm,
        right_perm=conf.right_perm,
        num_gpus=conf.num_devices,
        disp_size=conf.disp_size,
        offsets=conf.offsets,
        conf=conf,
    )
    return shard_map(
        func,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(),
        ),
        out_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
        ),
        check_rep=False,
    )


def initialize_mGPU_reconstruct_pre_drift_pullback(conf):
    """Create the fused reconstruction/pullback callable when available.

    Parameters
    ----------
    conf : Configuration
        Configuration defining the active multi-GPU mode and capacities.

    Returns
    -------
    callable or None
        Fused reconstruction/pullback callable, or ``None`` when unavailable.
    """
    if conf.num_devices == 1:
        return None

    pullback_fn = reconstruct_pre_drift_and_pullback_canonical_shard_map
    if conf.multigpu_mode == "mesh_halo":
        pullback_fn = reconstruct_pre_drift_and_pullback_mesh_halo_shard_map

    func = partial(
        pullback_fn,
        global_nMesh=conf.nMesh,
        max_values_to_share=conf.max_share_ptcl,
        max_halo_values_to_share=_halo_capacity(conf),
        max_ptcl_per_slice=conf.max_ptcl_per_slice,
        left_perm=conf.left_perm,
        right_perm=conf.right_perm,
        num_gpus=conf.num_devices,
        disp_size=conf.disp_size,
        offsets=conf.offsets,
        conf=conf,
    )
    return shard_map(
        func,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
        ),
        out_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
        ),
        check_rep=False,
    )


def initialize_mGPU_halo_move_pullback(conf):
    """Create the sharded halo-move transpose callable.

    Parameters
    ----------
    conf : Configuration
        Configuration defining the active multi-GPU mode and capacities.

    Returns
    -------
    callable
        Pullback callable for routing cotangents through halo movement.
    """
    if conf.num_devices == 1:
        def _pullback_noop(
            pmid,
            source_disp,
            carried_disp,
            vel,
            acc,
            halo_end,
            unused_indexes,
            disp_cot,
            vel_cot,
            acc_cot,
        ):
            del pmid, source_disp, carried_disp, vel, acc, halo_end, unused_indexes
            return disp_cot, vel_cot, acc_cot
        return _pullback_noop

    pullback_fn = halo_move_pullback_from_prestate_shard_map
    if conf.multigpu_mode == "mesh_halo":
        pullback_fn = halo_move_pullback_mesh_halo_from_prestate_shard_map

    func = partial(
        pullback_fn,
        global_nMesh=conf.nMesh,
        max_values_to_share=conf.max_share_ptcl,
        max_halo_values_to_share=_halo_capacity(conf),
        max_ptcl_per_slice=conf.max_ptcl_per_slice,
        left_perm=conf.left_perm,
        right_perm=conf.right_perm,
        num_gpus=conf.num_devices,
        disp_size=conf.disp_size,
        offsets=conf.offsets,
        conf=conf,
    )
    return shard_map(
        func,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
        ),
        out_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
        ),
        check_rep=False,
    )


def initialize_mGPU_compute_halo_mask(conf):
    """Create the sharded halo-mask helper for the active multi-GPU mode.

    Parameters
    ----------
    conf : Configuration
        Configuration defining the active multi-GPU mode.

    Returns
    -------
    callable
        Halo-mask helper specialized to the active runtime path.
    """
    if conf.multigpu_mode == "mesh_halo":
        if conf.num_devices == 1:
            def _compute_halo_mask_mesh_halo_noop(pmid, disp, unused_indexes, halo_start, halo_end):
                del pmid, disp, halo_start, halo_end
                return jnp.zeros_like(unused_indexes)
            return _compute_halo_mask_mesh_halo_noop

        def _zero_halo_mask_shard(pmid, disp, unused_indexes, halo_start, halo_end):
            del pmid, disp, halo_start, halo_end
            return jnp.zeros_like(unused_indexes)

        return shard_map(
            _zero_halo_mask_shard,
            mesh=conf.compute_mesh,
            in_specs=(
                P(AXIS_NAME, None),
                P(AXIS_NAME, None),
                P(AXIS_NAME),
                P(AXIS_NAME),
                P(AXIS_NAME),
            ),
            out_specs=P(AXIS_NAME),
            check_rep=False,
        )

    if conf.num_devices == 1:
        def _compute_halo_mask_noop(pmid, disp, unused_indexes, halo_start, halo_end):
            del halo_start, halo_end
            x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % conf.nMesh
            return compute_halo_mask(x_mod, conf.halo_start.squeeze(), conf.halo_end.squeeze(), unused_indexes)
        return _compute_halo_mask_noop

    func = partial(
        compute_halo_mask_shard_map,
        global_nMesh=conf.nMesh,
        disp_size=conf.disp_size,
    )
    return shard_map(
        func,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            P(AXIS_NAME),
            P(AXIS_NAME),
        ),
        out_specs=P(AXIS_NAME),
        check_rep=False,
    )
