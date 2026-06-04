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

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .utils import AXIS_NAME, pmid_to_idx, raise_error


@jax.jit
def particles_in_slice_mask(x_mod, slice_start, slice_end):
    """Return the wrapped x-slab membership mask for particle positions."""
    within_slice = (x_mod >= slice_start) & (x_mod < slice_end)
    across_boundary = (x_mod >= slice_start) | (x_mod < slice_end)
    return jnp.where(slice_start > slice_end, across_boundary, within_slice)


@jax.jit
def compute_halo_mask(x_mod, halo_start, halo_end, unused_indexes):
    """Return the mask of duplicated halo particles for the current slab."""

    def slice_mask(start, end):
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
    _capacity_check(count, capacity, error_message)
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


def _sorted_merge_two(
    keys_a,
    pmid_a,
    disp_a,
    vel_a,
    acc_a,
    valid_a,
    keys_b,
    pmid_b,
    disp_b,
    vel_b,
    acc_b,
    valid_b,
    capacity,
    key_fill,
    error_message,
):
    """Merge two sorted fixed-capacity particle streams."""
    count_a = jnp.sum(valid_a)
    count_b = jnp.sum(valid_b)
    total = count_a + count_b
    _capacity_check(total, capacity, error_message)
    Na = keys_a.shape[0]
    Nb = keys_b.shape[0]
    keys_a_filled = jnp.where(valid_a, keys_a, key_fill)
    keys_b_filled = jnp.where(valid_b, keys_b, key_fill)

    pos_a = (
        jnp.arange(Na, dtype=jnp.int32)
        + jnp.searchsorted(keys_b_filled, keys_a_filled, side='left').astype(jnp.int32)
    )
    pos_b = (
        jnp.arange(Nb, dtype=jnp.int32)
        + jnp.searchsorted(keys_a_filled, keys_b_filled, side='right').astype(jnp.int32)
    )

    out_keys = jnp.full(capacity, key_fill, dtype=keys_a.dtype)
    out_pmid = jnp.zeros((capacity,) + pmid_a.shape[1:], dtype=pmid_a.dtype)
    out_disp = jnp.zeros((capacity,) + disp_a.shape[1:], dtype=disp_a.dtype)
    out_vel = jnp.zeros((capacity,) + vel_a.shape[1:], dtype=vel_a.dtype)
    out_acc = jnp.zeros((capacity,) + acc_a.shape[1:], dtype=acc_a.dtype)

    out_keys = out_keys.at[pos_a].set(keys_a_filled, mode='drop').at[pos_b].set(keys_b_filled, mode='drop')
    out_pmid = out_pmid.at[pos_a].set(pmid_a, mode='drop').at[pos_b].set(pmid_b, mode='drop')
    out_disp = out_disp.at[pos_a].set(disp_a, mode='drop').at[pos_b].set(disp_b, mode='drop')
    out_vel = out_vel.at[pos_a].set(vel_a, mode='drop').at[pos_b].set(vel_b, mode='drop')
    out_acc = out_acc.at[pos_a].set(acc_a, mode='drop').at[pos_b].set(acc_b, mode='drop')

    out_valid = jnp.arange(capacity) < total
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    valid_shape = (out_valid.shape[0],) + (1,) * (out_pmid.ndim - 1)
    out_pmid = jnp.where(out_valid.reshape(valid_shape), out_pmid, jnp.zeros_like(out_pmid))
    out_disp = jnp.where(out_valid.reshape(valid_shape), out_disp, jnp.zeros_like(out_disp))
    out_vel = jnp.where(out_valid.reshape(valid_shape), out_vel, jnp.zeros_like(out_vel))
    out_acc = jnp.where(out_valid.reshape(valid_shape), out_acc, jnp.zeros_like(out_acc))
    return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid


def _sorted_merge_two_with_provenance(
    keys_a,
    pmid_a,
    disp_a,
    vel_a,
    acc_a,
    valid_a,
    keys_b,
    pmid_b,
    disp_b,
    vel_b,
    acc_b,
    valid_b,
    capacity,
    key_fill,
    error_message,
    src_tag_b=jnp.int32(2),
):
    """Merge two sorted streams and record which input produced each output."""
    # Searchsorted-based merge with provenance tracking.
    # Same position formula as _sorted_merge_two; additionally scatter src_tag/src_idx.
    count_a = jnp.sum(valid_a)
    count_b = jnp.sum(valid_b)
    total = count_a + count_b
    _capacity_check(total, capacity, error_message)

    Na = keys_a.shape[0]
    Nb = keys_b.shape[0]

    keys_a_filled = jnp.where(valid_a, keys_a, key_fill)
    keys_b_filled = jnp.where(valid_b, keys_b, key_fill)

    pos_a = (jnp.arange(Na, dtype=jnp.int32)
             + jnp.searchsorted(keys_b_filled, keys_a_filled, side='left').astype(jnp.int32))
    pos_b = (jnp.arange(Nb, dtype=jnp.int32)
             + jnp.searchsorted(keys_a_filled, keys_b_filled, side='right').astype(jnp.int32))

    out_keys = jnp.full(capacity, key_fill, dtype=keys_a.dtype)
    out_pmid = jnp.zeros((capacity,) + pmid_a.shape[1:], dtype=pmid_a.dtype)
    out_disp = jnp.zeros((capacity,) + disp_a.shape[1:], dtype=disp_a.dtype)
    out_vel = jnp.zeros((capacity,) + vel_a.shape[1:], dtype=vel_a.dtype)
    out_acc = jnp.zeros((capacity,) + acc_a.shape[1:], dtype=acc_a.dtype)
    out_src_tag = jnp.full(capacity, jnp.int32(3), dtype=jnp.int32)
    out_src_idx = jnp.full(capacity, jnp.int32(-1), dtype=jnp.int32)

    out_keys = out_keys.at[pos_a].set(keys_a_filled, mode='drop').at[pos_b].set(keys_b_filled, mode='drop')
    out_pmid = out_pmid.at[pos_a].set(pmid_a, mode='drop').at[pos_b].set(pmid_b, mode='drop')
    out_disp = out_disp.at[pos_a].set(disp_a, mode='drop').at[pos_b].set(disp_b, mode='drop')
    out_vel = out_vel.at[pos_a].set(vel_a, mode='drop').at[pos_b].set(vel_b, mode='drop')
    out_acc = out_acc.at[pos_a].set(acc_a, mode='drop').at[pos_b].set(acc_b, mode='drop')

    src_tag_a_arr = jnp.where(valid_a, jnp.int32(0), jnp.int32(3))
    src_tag_b_arr = jnp.where(valid_b, src_tag_b, jnp.int32(3))
    out_src_tag = out_src_tag.at[pos_a].set(src_tag_a_arr, mode='drop').at[pos_b].set(src_tag_b_arr, mode='drop')
    out_src_idx = (out_src_idx
                   .at[pos_a].set(jnp.arange(Na, dtype=jnp.int32), mode='drop')
                   .at[pos_b].set(jnp.arange(Nb, dtype=jnp.int32), mode='drop'))

    out_valid = jnp.arange(capacity) < total
    out_src_tag = jnp.where(out_valid, out_src_tag, jnp.int32(3))
    out_src_idx = jnp.where(out_valid, out_src_idx, jnp.int32(-1))
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    return (
        out_keys,
        out_pmid,
        out_disp,
        out_vel,
        out_acc,
        out_valid,
        out_src_tag,
        out_src_idx,
    )


def _sorted_merge_three(
    keys_a,
    pmid_a,
    disp_a,
    vel_a,
    acc_a,
    valid_a,
    keys_b,
    pmid_b,
    disp_b,
    vel_b,
    acc_b,
    valid_b,
    keys_c,
    pmid_c,
    disp_c,
    vel_c,
    acc_c,
    valid_c,
    capacity,
    key_fill,
    error_message,
):
    """Merge three sorted fixed-capacity particle streams."""
    count_a = jnp.sum(valid_a)
    count_b = jnp.sum(valid_b)
    count_c = jnp.sum(valid_c)
    total = count_a + count_b + count_c
    _capacity_check(total, capacity, error_message)
    Na = keys_a.shape[0]
    Nb = keys_b.shape[0]
    Nc = keys_c.shape[0]
    keys_a_filled = jnp.where(valid_a, keys_a, key_fill)
    keys_b_filled = jnp.where(valid_b, keys_b, key_fill)
    keys_c_filled = jnp.where(valid_c, keys_c, key_fill)

    pos_a = (
        jnp.arange(Na, dtype=jnp.int32)
        + jnp.searchsorted(keys_b_filled, keys_a_filled, side='left').astype(jnp.int32)
        + jnp.searchsorted(keys_c_filled, keys_a_filled, side='left').astype(jnp.int32)
    )
    pos_b = (
        jnp.arange(Nb, dtype=jnp.int32)
        + jnp.searchsorted(keys_a_filled, keys_b_filled, side='right').astype(jnp.int32)
        + jnp.searchsorted(keys_c_filled, keys_b_filled, side='left').astype(jnp.int32)
    )
    pos_c = (
        jnp.arange(Nc, dtype=jnp.int32)
        + jnp.searchsorted(keys_a_filled, keys_c_filled, side='right').astype(jnp.int32)
        + jnp.searchsorted(keys_b_filled, keys_c_filled, side='right').astype(jnp.int32)
    )

    out_keys = jnp.full(capacity, key_fill, dtype=keys_a.dtype)
    out_pmid = jnp.zeros((capacity,) + pmid_a.shape[1:], dtype=pmid_a.dtype)
    out_disp = jnp.zeros((capacity,) + disp_a.shape[1:], dtype=disp_a.dtype)
    out_vel = jnp.zeros((capacity,) + vel_a.shape[1:], dtype=vel_a.dtype)
    out_acc = jnp.zeros((capacity,) + acc_a.shape[1:], dtype=acc_a.dtype)

    out_keys = (
        out_keys
        .at[pos_a].set(keys_a_filled, mode='drop')
        .at[pos_b].set(keys_b_filled, mode='drop')
        .at[pos_c].set(keys_c_filled, mode='drop')
    )
    out_pmid = out_pmid.at[pos_a].set(pmid_a, mode='drop').at[pos_b].set(pmid_b, mode='drop').at[pos_c].set(pmid_c, mode='drop')
    out_disp = out_disp.at[pos_a].set(disp_a, mode='drop').at[pos_b].set(disp_b, mode='drop').at[pos_c].set(disp_c, mode='drop')
    out_vel = out_vel.at[pos_a].set(vel_a, mode='drop').at[pos_b].set(vel_b, mode='drop').at[pos_c].set(vel_c, mode='drop')
    out_acc = out_acc.at[pos_a].set(acc_a, mode='drop').at[pos_b].set(acc_b, mode='drop').at[pos_c].set(acc_c, mode='drop')

    out_valid = jnp.arange(capacity) < total
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    valid_shape = (out_valid.shape[0],) + (1,) * (out_pmid.ndim - 1)
    out_pmid = jnp.where(out_valid.reshape(valid_shape), out_pmid, jnp.zeros_like(out_pmid))
    out_disp = jnp.where(out_valid.reshape(valid_shape), out_disp, jnp.zeros_like(out_disp))
    out_vel = jnp.where(out_valid.reshape(valid_shape), out_vel, jnp.zeros_like(out_vel))
    out_acc = jnp.where(out_valid.reshape(valid_shape), out_acc, jnp.zeros_like(out_acc))
    return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid


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


def _compact_positions(mask, capacity, error_message):
    """Compact source positions for later route transposes."""
    pos, valid = _compact_sorted_indices(mask, capacity, error_message)
    return jnp.where(valid, pos, jnp.asarray(-1, pos.dtype))


def _sorted_merge_three_with_provenance(
    keys_a,
    pmid_a,
    disp_a,
    vel_a,
    acc_a,
    valid_a,
    keys_b,
    pmid_b,
    disp_b,
    vel_b,
    acc_b,
    valid_b,
    keys_c,
    pmid_c,
    disp_c,
    vel_c,
    acc_c,
    valid_c,
    capacity,
    key_fill,
    error_message,
):
    """Merge three sorted streams and keep source tags for the transpose."""
    count_a = jnp.sum(valid_a)
    count_b = jnp.sum(valid_b)
    count_c = jnp.sum(valid_c)
    total = count_a + count_b + count_c
    _capacity_check(total, capacity, error_message)

    keys_cat = jnp.concatenate((
        jnp.where(valid_a, keys_a, key_fill),
        jnp.where(valid_b, keys_b, key_fill),
        jnp.where(valid_c, keys_c, key_fill),
    ), axis=0)
    pmid_cat = jnp.concatenate((pmid_a, pmid_b, pmid_c), axis=0)
    disp_cat = jnp.concatenate((disp_a, disp_b, disp_c), axis=0)
    vel_cat = jnp.concatenate((vel_a, vel_b, vel_c), axis=0)
    acc_cat = jnp.concatenate((acc_a, acc_b, acc_c), axis=0)

    src_a = jnp.arange(keys_a.shape[0], dtype=jnp.int32)
    src_b = jnp.arange(keys_b.shape[0], dtype=jnp.int32)
    src_c = jnp.arange(keys_c.shape[0], dtype=jnp.int32)
    src_idx = jnp.concatenate((src_a, src_b, src_c), axis=0)
    src_tag = jnp.concatenate((
        jnp.where(valid_a, jnp.int32(0), jnp.int32(3)),
        jnp.where(valid_b, jnp.int32(1), jnp.int32(3)),
        jnp.where(valid_c, jnp.int32(2), jnp.int32(3)),
    ), axis=0)

    order = jnp.argsort(keys_cat, stable=True)[:capacity]
    out_valid = jnp.arange(capacity) < total
    out_keys = keys_cat[order]
    out_pmid = pmid_cat[order]
    out_disp = disp_cat[order]
    out_vel = vel_cat[order]
    out_acc = acc_cat[order]
    out_src_idx = jnp.where(out_valid, src_idx[order], -1)
    out_src_tag = jnp.where(out_valid, src_tag[order], 3)
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    return (
        out_keys,
        out_pmid,
        out_disp,
        out_vel,
        out_acc,
        out_valid,
        out_src_tag,
        out_src_idx,
    )


def _ordered_block_take(values, start, count, slots):
    """Take a contiguous logical block into arbitrary output slots."""
    idx, mask = _ordered_block_take_plan(start, count, slots, values.shape[0])
    return mask, _ordered_block_take_from_plan(values, idx, mask)


def _ordered_block_take_plan(start, count, slots, value_count):
    """Precompute indices and masks for ``_ordered_block_take``."""
    start = start.astype(slots.dtype)
    count = count.astype(slots.dtype)
    idx = jnp.clip(slots - start, 0, value_count - 1)
    mask = (slots >= start) & (slots < (start + count))
    return idx, mask


def _ordered_block_take_from_plan(values, idx, mask):
    """Apply a precomputed ordered-block take plan to one payload array."""
    taken = values[idx]
    mask_shape = (mask.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.where(mask.reshape(mask_shape), taken, jnp.zeros_like(taken))


def _extract_compact_block(values, start, count, out_size):
    """Extract a fixed-capacity compact block from a larger ordered buffer."""
    slots = jnp.arange(out_size, dtype=jnp.int32)
    idx = jnp.clip(start + slots, 0, values.shape[0] - 1)
    taken = values[idx]
    mask = slots < count
    mask_shape = (mask.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.where(mask.reshape(mask_shape), taken, jnp.zeros_like(taken))


def _positions_from_range(start, count, capacity):
    """Return source positions for a compact contiguous range."""
    slots = jnp.arange(capacity, dtype=jnp.int32)
    start = start.astype(slots.dtype)
    return jnp.where(slots < count, start + slots, jnp.asarray(-1, slots.dtype))


def _route_merge_two_topology(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_incoming,
    pmid_incoming,
    disp_incoming,
    vel_incoming,
    acc_incoming,
    valid_incoming,
    capacity,
    key_fill,
    error_message,
    num_gpus,
):
    """Merge stay and one incoming migration stream in slab-topology order."""
    count_stay = jnp.sum(valid_stay)
    count_incoming = jnp.sum(valid_incoming)
    total = count_stay + count_incoming
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    incoming_first = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)
    stay_start = jnp.where(incoming_first, count_incoming, zero)
    incoming_start = jnp.where(incoming_first, zero, count_stay)
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    incoming_idx, incoming_mask = _ordered_block_take_plan(
        incoming_start,
        count_incoming,
        slots,
        keys_incoming.shape[0],
    )

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)

    incoming_keys = _ordered_block_take_from_plan(keys_incoming, incoming_idx, incoming_mask)
    incoming_pmid = _ordered_block_take_from_plan(pmid_incoming, incoming_idx, incoming_mask)
    incoming_disp = _ordered_block_take_from_plan(disp_incoming, incoming_idx, incoming_mask)
    incoming_vel = _ordered_block_take_from_plan(vel_incoming, incoming_idx, incoming_mask)
    incoming_acc = _ordered_block_take_from_plan(acc_incoming, incoming_idx, incoming_mask)

    out_valid = slots < total
    out_keys = stay_keys + incoming_keys
    out_pmid = stay_pmid + incoming_pmid
    out_disp = stay_disp + incoming_disp
    out_vel = stay_vel + incoming_vel
    out_acc = stay_acc + incoming_acc
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid


def _route_merge_two_topology_with_blocks(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_incoming,
    pmid_incoming,
    disp_incoming,
    vel_incoming,
    acc_incoming,
    valid_incoming,
    capacity,
    key_fill,
    error_message,
    num_gpus,
):
    """Two-stream topology merge that also returns compact block metadata."""
    count_stay = jnp.sum(valid_stay)
    count_incoming = jnp.sum(valid_incoming)
    total = count_stay + count_incoming
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    incoming_first = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)
    stay_start = jnp.where(incoming_first, count_incoming, zero)
    incoming_start = jnp.where(incoming_first, zero, count_stay)
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    incoming_idx, incoming_mask = _ordered_block_take_plan(
        incoming_start,
        count_incoming,
        slots,
        keys_incoming.shape[0],
    )

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)

    incoming_keys = _ordered_block_take_from_plan(keys_incoming, incoming_idx, incoming_mask)
    incoming_pmid = _ordered_block_take_from_plan(pmid_incoming, incoming_idx, incoming_mask)
    incoming_disp = _ordered_block_take_from_plan(disp_incoming, incoming_idx, incoming_mask)
    incoming_vel = _ordered_block_take_from_plan(vel_incoming, incoming_idx, incoming_mask)
    incoming_acc = _ordered_block_take_from_plan(acc_incoming, incoming_idx, incoming_mask)

    out_valid = slots < total
    out_keys = stay_keys + incoming_keys
    out_pmid = stay_pmid + incoming_pmid
    out_disp = stay_disp + incoming_disp
    out_vel = stay_vel + incoming_vel
    out_acc = stay_acc + incoming_acc
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    return (
        out_keys,
        out_pmid,
        out_disp,
        out_vel,
        out_acc,
        out_valid,
        stay_start,
        count_stay,
        incoming_start,
        count_incoming,
        zero,
        zero,
    )


def _route_merge_two_topology_with_source_tags(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_incoming,
    pmid_incoming,
    disp_incoming,
    vel_incoming,
    acc_incoming,
    valid_incoming,
    capacity,
    key_fill,
    error_message,
    num_gpus,
    incoming_src_tag=jnp.uint8(1),
):
    """Two-stream topology merge with source tags for custom transposes."""
    count_stay = jnp.sum(valid_stay)
    count_incoming = jnp.sum(valid_incoming)
    total = count_stay + count_incoming
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    incoming_first = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)
    stay_start = jnp.where(incoming_first, count_incoming, zero)
    incoming_start = jnp.where(incoming_first, zero, count_stay)
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    incoming_idx, incoming_mask = _ordered_block_take_plan(
        incoming_start,
        count_incoming,
        slots,
        keys_incoming.shape[0],
    )

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)

    incoming_keys = _ordered_block_take_from_plan(keys_incoming, incoming_idx, incoming_mask)
    incoming_pmid = _ordered_block_take_from_plan(pmid_incoming, incoming_idx, incoming_mask)
    incoming_disp = _ordered_block_take_from_plan(disp_incoming, incoming_idx, incoming_mask)
    incoming_vel = _ordered_block_take_from_plan(vel_incoming, incoming_idx, incoming_mask)
    incoming_acc = _ordered_block_take_from_plan(acc_incoming, incoming_idx, incoming_mask)

    out_valid = slots < total
    out_keys = stay_keys + incoming_keys
    out_pmid = stay_pmid + incoming_pmid
    out_disp = stay_disp + incoming_disp
    out_vel = stay_vel + incoming_vel
    out_acc = stay_acc + incoming_acc
    out_src_tag = incoming_mask.astype(jnp.uint8) * incoming_src_tag
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    out_src_tag = jnp.where(out_valid, out_src_tag, jnp.uint8(3))
    return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid, out_src_tag


def _route_merge_two_topology_with_provenance(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_incoming,
    pmid_incoming,
    disp_incoming,
    vel_incoming,
    acc_incoming,
    valid_incoming,
    capacity,
    key_fill,
    error_message,
    num_gpus,
    incoming_src_tag=jnp.int32(2),
):
    """Two-stream topology merge with explicit source indices and tags."""
    count_stay = jnp.sum(valid_stay)
    count_incoming = jnp.sum(valid_incoming)
    total = count_stay + count_incoming
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    incoming_first = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)
    stay_start = jnp.where(incoming_first, count_incoming, zero)
    incoming_start = jnp.where(incoming_first, zero, count_stay)
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    incoming_idx, incoming_mask = _ordered_block_take_plan(
        incoming_start,
        count_incoming,
        slots,
        keys_incoming.shape[0],
    )

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)
    stay_src_idx = _ordered_block_take_from_plan(
        jnp.arange(keys_stay.shape[0], dtype=jnp.int32),
        stay_idx,
        stay_mask,
    )

    incoming_keys = _ordered_block_take_from_plan(keys_incoming, incoming_idx, incoming_mask)
    incoming_pmid = _ordered_block_take_from_plan(pmid_incoming, incoming_idx, incoming_mask)
    incoming_disp = _ordered_block_take_from_plan(disp_incoming, incoming_idx, incoming_mask)
    incoming_vel = _ordered_block_take_from_plan(vel_incoming, incoming_idx, incoming_mask)
    incoming_acc = _ordered_block_take_from_plan(acc_incoming, incoming_idx, incoming_mask)
    incoming_src_idx = _ordered_block_take_from_plan(
        jnp.arange(keys_incoming.shape[0], dtype=jnp.int32),
        incoming_idx,
        incoming_mask,
    )

    out_valid = slots < total
    out_keys = stay_keys + incoming_keys
    out_pmid = stay_pmid + incoming_pmid
    out_disp = stay_disp + incoming_disp
    out_vel = stay_vel + incoming_vel
    out_acc = stay_acc + incoming_acc
    out_src_idx = stay_src_idx + incoming_src_idx
    out_src_tag = incoming_mask.astype(jnp.int32) * incoming_src_tag
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    out_src_idx = jnp.where(out_valid, out_src_idx, -1)
    out_src_tag = jnp.where(out_valid, out_src_tag, 3)
    return (
        out_keys,
        out_pmid,
        out_disp,
        out_vel,
        out_acc,
        out_valid,
        out_src_tag,
        out_src_idx,
    )


def _route_merge_three_topology(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_left,
    pmid_left,
    disp_left,
    vel_left,
    acc_left,
    valid_left,
    keys_right,
    pmid_right,
    disp_right,
    vel_right,
    acc_right,
    valid_right,
    capacity,
    key_fill,
    error_message,
    num_gpus,
):
    """Merge stay, left-incoming, and right-incoming streams in slab order."""
    count_stay = jnp.sum(valid_stay)
    count_left = jnp.sum(valid_left)
    count_right = jnp.sum(valid_right)
    total = count_stay + count_left + count_right
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    is_first = gpu_pos == 0
    is_last = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)

    stay_start = jnp.where(
        is_first,
        zero,
        jnp.where(is_last, count_right + count_left, count_left),
    )
    left_start = jnp.where(
        is_first,
        count_stay + count_right,
        jnp.where(is_last, count_right, zero),
    )
    right_start = jnp.where(
        is_first,
        count_stay,
        jnp.where(is_last, zero, count_left + count_stay),
    )
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    left_idx, left_mask = _ordered_block_take_plan(left_start, count_left, slots, keys_left.shape[0])
    right_idx, right_mask = _ordered_block_take_plan(right_start, count_right, slots, keys_right.shape[0])

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)

    left_keys = _ordered_block_take_from_plan(keys_left, left_idx, left_mask)
    left_pmid = _ordered_block_take_from_plan(pmid_left, left_idx, left_mask)
    left_disp = _ordered_block_take_from_plan(disp_left, left_idx, left_mask)
    left_vel = _ordered_block_take_from_plan(vel_left, left_idx, left_mask)
    left_acc = _ordered_block_take_from_plan(acc_left, left_idx, left_mask)

    right_keys = _ordered_block_take_from_plan(keys_right, right_idx, right_mask)
    right_pmid = _ordered_block_take_from_plan(pmid_right, right_idx, right_mask)
    right_disp = _ordered_block_take_from_plan(disp_right, right_idx, right_mask)
    right_vel = _ordered_block_take_from_plan(vel_right, right_idx, right_mask)
    right_acc = _ordered_block_take_from_plan(acc_right, right_idx, right_mask)

    out_valid = slots < total
    out_keys = stay_keys + left_keys + right_keys
    out_pmid = stay_pmid + left_pmid + right_pmid
    out_disp = stay_disp + left_disp + right_disp
    out_vel = stay_vel + left_vel + right_vel
    out_acc = stay_acc + left_acc + right_acc
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid


def _route_merge_three_topology_with_blocks(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_left,
    pmid_left,
    disp_left,
    vel_left,
    acc_left,
    valid_left,
    keys_right,
    pmid_right,
    disp_right,
    vel_right,
    acc_right,
    valid_right,
    capacity,
    key_fill,
    error_message,
    num_gpus,
):
    """Three-stream topology merge that also returns compact block metadata."""
    count_stay = jnp.sum(valid_stay)
    count_left = jnp.sum(valid_left)
    count_right = jnp.sum(valid_right)
    total = count_stay + count_left + count_right
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    is_first = gpu_pos == 0
    is_last = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)

    stay_start = jnp.where(
        is_first,
        zero,
        jnp.where(is_last, count_right + count_left, count_left),
    )
    left_start = jnp.where(
        is_first,
        count_stay + count_right,
        jnp.where(is_last, count_right, zero),
    )
    right_start = jnp.where(
        is_first,
        count_stay,
        jnp.where(is_last, zero, count_left + count_stay),
    )
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    left_idx, left_mask = _ordered_block_take_plan(left_start, count_left, slots, keys_left.shape[0])
    right_idx, right_mask = _ordered_block_take_plan(right_start, count_right, slots, keys_right.shape[0])

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)

    left_keys = _ordered_block_take_from_plan(keys_left, left_idx, left_mask)
    left_pmid = _ordered_block_take_from_plan(pmid_left, left_idx, left_mask)
    left_disp = _ordered_block_take_from_plan(disp_left, left_idx, left_mask)
    left_vel = _ordered_block_take_from_plan(vel_left, left_idx, left_mask)
    left_acc = _ordered_block_take_from_plan(acc_left, left_idx, left_mask)

    right_keys = _ordered_block_take_from_plan(keys_right, right_idx, right_mask)
    right_pmid = _ordered_block_take_from_plan(pmid_right, right_idx, right_mask)
    right_disp = _ordered_block_take_from_plan(disp_right, right_idx, right_mask)
    right_vel = _ordered_block_take_from_plan(vel_right, right_idx, right_mask)
    right_acc = _ordered_block_take_from_plan(acc_right, right_idx, right_mask)

    out_valid = slots < total
    out_keys = stay_keys + left_keys + right_keys
    out_pmid = stay_pmid + left_pmid + right_pmid
    out_disp = stay_disp + left_disp + right_disp
    out_vel = stay_vel + left_vel + right_vel
    out_acc = stay_acc + left_acc + right_acc
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    return (
        out_keys,
        out_pmid,
        out_disp,
        out_vel,
        out_acc,
        out_valid,
        stay_start,
        count_stay,
        left_start,
        count_left,
        right_start,
        count_right,
    )


def _route_merge_three_topology_with_source_tags(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_left,
    pmid_left,
    disp_left,
    vel_left,
    acc_left,
    valid_left,
    keys_right,
    pmid_right,
    disp_right,
    vel_right,
    acc_right,
    valid_right,
    capacity,
    key_fill,
    error_message,
    num_gpus,
):
    """Three-stream topology merge with source tags for custom transposes."""
    count_stay = jnp.sum(valid_stay)
    count_left = jnp.sum(valid_left)
    count_right = jnp.sum(valid_right)
    total = count_stay + count_left + count_right
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    is_first = gpu_pos == 0
    is_last = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)

    stay_start = jnp.where(
        is_first,
        zero,
        jnp.where(is_last, count_right + count_left, count_left),
    )
    left_start = jnp.where(
        is_first,
        count_stay + count_right,
        jnp.where(is_last, count_right, zero),
    )
    right_start = jnp.where(
        is_first,
        count_stay,
        jnp.where(is_last, zero, count_left + count_stay),
    )
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    left_idx, left_mask = _ordered_block_take_plan(left_start, count_left, slots, keys_left.shape[0])
    right_idx, right_mask = _ordered_block_take_plan(right_start, count_right, slots, keys_right.shape[0])

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)

    left_keys = _ordered_block_take_from_plan(keys_left, left_idx, left_mask)
    left_pmid = _ordered_block_take_from_plan(pmid_left, left_idx, left_mask)
    left_disp = _ordered_block_take_from_plan(disp_left, left_idx, left_mask)
    left_vel = _ordered_block_take_from_plan(vel_left, left_idx, left_mask)
    left_acc = _ordered_block_take_from_plan(acc_left, left_idx, left_mask)

    right_keys = _ordered_block_take_from_plan(keys_right, right_idx, right_mask)
    right_pmid = _ordered_block_take_from_plan(pmid_right, right_idx, right_mask)
    right_disp = _ordered_block_take_from_plan(disp_right, right_idx, right_mask)
    right_vel = _ordered_block_take_from_plan(vel_right, right_idx, right_mask)
    right_acc = _ordered_block_take_from_plan(acc_right, right_idx, right_mask)

    out_valid = slots < total
    out_keys = stay_keys + left_keys + right_keys
    out_pmid = stay_pmid + left_pmid + right_pmid
    out_disp = stay_disp + left_disp + right_disp
    out_vel = stay_vel + left_vel + right_vel
    out_acc = stay_acc + left_acc + right_acc
    out_src_tag = left_mask.astype(jnp.uint8) + jnp.uint8(2) * right_mask.astype(jnp.uint8)
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    out_src_tag = jnp.where(out_valid, out_src_tag, jnp.uint8(3))
    return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid, out_src_tag


def _route_merge_three_topology_with_provenance(
    keys_stay,
    pmid_stay,
    disp_stay,
    vel_stay,
    acc_stay,
    valid_stay,
    keys_left,
    pmid_left,
    disp_left,
    vel_left,
    acc_left,
    valid_left,
    keys_right,
    pmid_right,
    disp_right,
    vel_right,
    acc_right,
    valid_right,
    capacity,
    key_fill,
    error_message,
    num_gpus,
):
    """Three-stream topology merge with explicit source indices and tags."""
    count_stay = jnp.sum(valid_stay)
    count_left = jnp.sum(valid_left)
    count_right = jnp.sum(valid_right)
    total = count_stay + count_left + count_right
    _capacity_check(total, capacity, error_message)

    gpu_pos = jax.lax.axis_index(AXIS_NAME)
    is_first = gpu_pos == 0
    is_last = gpu_pos == (num_gpus - 1)
    zero = jnp.asarray(0, count_stay.dtype)

    stay_start = jnp.where(
        is_first,
        zero,
        jnp.where(is_last, count_right + count_left, count_left),
    )
    left_start = jnp.where(
        is_first,
        count_stay + count_right,
        jnp.where(is_last, count_right, zero),
    )
    right_start = jnp.where(
        is_first,
        count_stay,
        jnp.where(is_last, zero, count_left + count_stay),
    )
    slots = jnp.arange(capacity, dtype=jnp.int32)

    stay_idx, stay_mask = _ordered_block_take_plan(stay_start, count_stay, slots, keys_stay.shape[0])
    left_idx, left_mask = _ordered_block_take_plan(left_start, count_left, slots, keys_left.shape[0])
    right_idx, right_mask = _ordered_block_take_plan(right_start, count_right, slots, keys_right.shape[0])

    stay_keys = _ordered_block_take_from_plan(keys_stay, stay_idx, stay_mask)
    stay_pmid = _ordered_block_take_from_plan(pmid_stay, stay_idx, stay_mask)
    stay_disp = _ordered_block_take_from_plan(disp_stay, stay_idx, stay_mask)
    stay_vel = _ordered_block_take_from_plan(vel_stay, stay_idx, stay_mask)
    stay_acc = _ordered_block_take_from_plan(acc_stay, stay_idx, stay_mask)
    stay_src_idx = _ordered_block_take_from_plan(
        jnp.arange(keys_stay.shape[0], dtype=jnp.int32),
        stay_idx,
        stay_mask,
    )

    left_keys = _ordered_block_take_from_plan(keys_left, left_idx, left_mask)
    left_pmid = _ordered_block_take_from_plan(pmid_left, left_idx, left_mask)
    left_disp = _ordered_block_take_from_plan(disp_left, left_idx, left_mask)
    left_vel = _ordered_block_take_from_plan(vel_left, left_idx, left_mask)
    left_acc = _ordered_block_take_from_plan(acc_left, left_idx, left_mask)
    left_src_idx = _ordered_block_take_from_plan(
        jnp.arange(keys_left.shape[0], dtype=jnp.int32),
        left_idx,
        left_mask,
    )

    right_keys = _ordered_block_take_from_plan(keys_right, right_idx, right_mask)
    right_pmid = _ordered_block_take_from_plan(pmid_right, right_idx, right_mask)
    right_disp = _ordered_block_take_from_plan(disp_right, right_idx, right_mask)
    right_vel = _ordered_block_take_from_plan(vel_right, right_idx, right_mask)
    right_acc = _ordered_block_take_from_plan(acc_right, right_idx, right_mask)
    right_src_idx = _ordered_block_take_from_plan(
        jnp.arange(keys_right.shape[0], dtype=jnp.int32),
        right_idx,
        right_mask,
    )

    out_valid = slots < total
    out_keys = stay_keys + left_keys + right_keys
    out_pmid = stay_pmid + left_pmid + right_pmid
    out_disp = stay_disp + left_disp + right_disp
    out_vel = stay_vel + left_vel + right_vel
    out_acc = stay_acc + left_acc + right_acc
    out_src_idx = stay_src_idx + left_src_idx + right_src_idx
    out_src_tag = left_mask.astype(jnp.int32) + 2 * right_mask.astype(jnp.int32)
    out_keys = jnp.where(out_valid, out_keys, key_fill)
    out_src_idx = jnp.where(out_valid, out_src_idx, -1)
    out_src_tag = jnp.where(out_valid, out_src_tag, 3)
    return (
        out_keys,
        out_pmid,
        out_disp,
        out_vel,
        out_acc,
        out_valid,
        out_src_tag,
        out_src_idx,
    )


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

    _ = jax.lax.cond(
        jnp.any(dropped_mask),
        lambda _: raise_error(
            "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
            "particles_outside_neighbor_range={x}.",
            x=jnp.sum(dropped_mask),
        ),
        lambda _: None,
        operand=None,
    )

    key_fill = _key_fill_value(conf)
    stay = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, stay_mask, pmid.shape[0], key_fill,
        "[ERROR] Exceeded stay-particle compact capacity. stay_particles={x}, capacity={y}.",
    )
    send_left = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded left-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )

    if num_gpus == 2:
        # 2-GPU fast path: send_right_mask is always zero, skip its compact and ppermute.
        incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)
        merged = _sorted_merge_two(
            *stay,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        max_particles_moved = jnp.sum(send_left[-1])
    else:
        send_right = _compact_sorted_particles(
            keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share, key_fill,
            "[ERROR] Exceeded right-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
        )
        incoming_from_left = jax.lax.ppermute(send_right, axis_name=AXIS_NAME, perm=right_perm)
        incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)
        merged = _sorted_merge_three(
            *stay,
            *incoming_from_left,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        max_particles_moved = jnp.maximum(jnp.sum(send_left[-1]), jnp.sum(send_right[-1]))

    return merged, max_particles_moved


def _canonical_route_authoritative_with_source_tags(
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
    """Route authoritative particles and tag whether they stayed or migrated."""
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

    _ = jax.lax.cond(
        jnp.any(dropped_mask),
        lambda _: raise_error(
            "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
            "particles_outside_neighbor_range={x}.",
            x=jnp.sum(dropped_mask),
        ),
        lambda _: None,
        operand=None,
    )

    key_fill = _key_fill_value(conf)
    stay = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, stay_mask, pmid.shape[0], key_fill,
        "[ERROR] Exceeded stay-particle compact capacity. stay_particles={x}, capacity={y}.",
    )
    send_left = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded left-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )
    incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)
    if num_gpus == 2:
        return _route_merge_two_topology_with_source_tags(
            *stay,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
            num_gpus,
            jnp.uint8(1),
        )

    send_right = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded right-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )
    incoming_from_left = jax.lax.ppermute(send_right, axis_name=AXIS_NAME, perm=right_perm)
    return _route_merge_three_topology_with_source_tags(
        *stay,
        *incoming_from_left,
        *incoming_from_right,
        pmid.shape[0],
        key_fill,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
        num_gpus,
    )


def _canonical_route_authoritative_with_block_metadata(
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
    """Route authoritative particles and return compact block metadata."""
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

    _ = jax.lax.cond(
        jnp.any(dropped_mask),
        lambda _: raise_error(
            "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
            "particles_outside_neighbor_range={x}.",
            x=jnp.sum(dropped_mask),
        ),
        lambda _: None,
        operand=None,
    )

    key_fill = _key_fill_value(conf)
    stay = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, stay_mask, pmid.shape[0], key_fill,
        "[ERROR] Exceeded stay-particle compact capacity. stay_particles={x}, capacity={y}.",
    )
    send_left = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded left-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )
    incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)
    if num_gpus == 2:
        return _route_merge_two_topology_with_blocks(
            *stay,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
            num_gpus,
        )

    send_right = _compact_sorted_particles(
        keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded right-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )
    incoming_from_left = jax.lax.ppermute(send_right, axis_name=AXIS_NAME, perm=right_perm)
    return _route_merge_three_topology_with_blocks(
        *stay,
        *incoming_from_left,
        *incoming_from_right,
        pmid.shape[0],
        key_fill,
        "[ERROR] Exceeded canonical authoritative capacity after migration. "
        "required_particles={x}, max_ptcl_per_slice={y}.",
        num_gpus,
    )


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

    key_fill = _key_fill_value(conf)
    # Use _compact_sorted_particles_with_slots to get both the compacted particles
    # and their original positions (slots) in a single pass, eliminating the
    # duplicate jnp.compress calls that were previously needed for stay_pos / send_left_pos.
    *stay_items, stay_pos = _compact_sorted_particles_with_slots(
        keys, pmid, disp, vel, acc, stay_mask, pmid.shape[0], key_fill,
        "[ERROR] Exceeded stay-particle compact capacity. stay_particles={x}, capacity={y}.",
    )
    stay = tuple(stay_items)

    *send_left_items, send_left_pos = _compact_sorted_particles_with_slots(
        keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share, key_fill,
        "[ERROR] Exceeded left-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
    )
    send_left = tuple(send_left_items)

    if num_gpus == 2:
        # 2-GPU fast path: in a 2-GPU ring send_right_mask is always zero, so we
        # skip the right-side compact, the ppermute of zeros, and use a cheaper
        # 2-way merge (stay + incoming_from_right only).
        incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)
        merged = _sorted_merge_two_with_provenance(
            *stay,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
            src_tag_b=jnp.int32(2),
        )
        send_right_pos = jnp.full((max_values_to_share,), -1, dtype=jnp.int32)
        send_right_valid = jnp.zeros((max_values_to_share,), dtype=jnp.bool_)
    else:
        *send_right_items, send_right_pos = _compact_sorted_particles_with_slots(
            keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share, key_fill,
            "[ERROR] Exceeded right-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
        )
        send_right = tuple(send_right_items)
        incoming_from_left = jax.lax.ppermute(send_right, axis_name=AXIS_NAME, perm=right_perm)
        incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)
        merged = _sorted_merge_three_with_provenance(
            *stay,
            *incoming_from_left,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        send_right_valid = send_right[-1]

    route_aux = (
        stay_pos,
        stay[-1],
        send_left_pos,
        send_left[-1],
        send_right_pos,
        send_right_valid,
        merged[-2],
        merged[-1],
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
):
    """Transpose the authoritative particle migration route."""
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
):
    """2-GPU fast path for _reverse_route_cot.

    In the 2-GPU topology send_right is always zero, so merge_src_tag never
    equals 1 (incoming_from_left). We skip the zero-ppermute on that side
    and the associated scatter.
    """
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
    """Pull cotangents through a canonical ``particle_halo`` move."""
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
    """Pull cotangents through a ``mesh_halo`` authoritative-only move."""
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
    """Move particles across slabs and rebuild duplicated particle-halo storage."""
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
    """Move particles across slabs while storing only authoritative particles."""
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
    """Reconstruct the canonical pre-drift particle-halo state from post-drift data."""
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
    """Reconstruct the pre-drift authoritative-only state from post-drift data."""
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
    """Fused mesh-halo reconstruction plus halo-move pullback for drift adjoints."""
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
    """Fused particle-halo reconstruction plus halo-move pullback for drift adjoints."""
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
    """Compute halo masks from sharded particle positions."""
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
    """Create the sharded forward particle-movement callable."""
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


def initialize_mGPU_reconstruct_pre_drift(conf):
    """Create the sharded pre-drift reconstruction callable."""
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
    """Create the fused reconstruction/pullback callable when available."""
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
    """Create the sharded halo-move transpose callable."""
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
    """Create the sharded halo-mask helper for the active multi-GPU mode."""
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
