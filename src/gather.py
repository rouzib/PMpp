from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental.shard_map import shard_map
from jax.lax import scan
from jax.sharding import NamedSharding, PartitionSpec as P

from .enmesh import _chunk_split, enmesh, _chunk_cat
from .halo_moving import particles_in_slice_mask
from .mesh_halo import (
    exchange_owned_mesh_halo_edges,
    extend_owned_mesh_from_halo_edges,
    extend_owned_mesh_with_halo,
    reduce_mesh_halo_to_owned,
)
from .utils import AXIS_NAME, raise_error


def initialize_mGPU_gather(conf):
    """Create the sharded gather entry point for the configured multi-GPU mode."""
    if conf.multigpu_mode == "mesh_halo":
        return shard_map(
            _gather_mGPU_mesh_halo,
            mesh=conf.compute_mesh,
            in_specs=(
                P(AXIS_NAME, None),  # pmid
                P(AXIS_NAME, None),  # disp
                P(AXIS_NAME),  # unused_index
                None,  # conf
                P(AXIS_NAME, None, None),  # mesh
            ),
            out_specs=P(AXIS_NAME),
            check_rep=False,
        )
    if conf.particle_halo_gather_mesh_halo:
        return shard_map(
            _gather_mGPU_particle_halo_mesh_edges,
            mesh=conf.compute_mesh,
            in_specs=(
                P(AXIS_NAME, None),  # pmid
                P(AXIS_NAME, None),  # disp
                P(AXIS_NAME),  # unused_index
                None,  # conf
                P(AXIS_NAME, None, None),  # mesh
            ),
            out_specs=P(AXIS_NAME),
            check_rep=False,
        )
    return shard_map(
        _gather_mGPU,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),  # pmid
            P(AXIS_NAME, None),  # disp
            P(AXIS_NAME),  # unused_index
            None,  # conf
            P(AXIS_NAME, None, None),  # mesh
        ),
        out_specs=P(AXIS_NAME),
        check_rep=False,
    )


@partial(custom_vjp, nondiff_argnums=(3,))
def _gather_mGPU_mesh_halo(pmid, disp, unused_index, conf, mesh):
    """Gather in ``mesh_halo`` mode from an owned mesh plus exchanged edge cells."""
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
        mesh,
        conf.mesh_halo_width,
        conf.left_perm,
        conf.right_perm,
    )
    mesh_halo = extend_owned_mesh_from_halo_edges(mesh, incoming_left, incoming_right, conf.mesh_halo_width)
    offset = conf.mesh_halo_offsets[gpu_id]
    val = _gather_impl(pmid, disp, conf, mesh_halo, 0, offset, None)
    mask = unused_index.reshape(unused_index.shape + (1,) * (val.ndim - 1))
    return jnp.where(mask, jnp.zeros_like(val), val)


def _gather_mGPU_mesh_halo_fwd(pmid, disp, unused_index, conf, mesh):
    """Forward rule that saves exchanged mesh edges for the mesh-halo gather VJP."""
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
        mesh,
        conf.mesh_halo_width,
        conf.left_perm,
        conf.right_perm,
    )
    mesh_halo = extend_owned_mesh_from_halo_edges(mesh, incoming_left, incoming_right, conf.mesh_halo_width)
    offset = conf.mesh_halo_offsets[gpu_id]
    val = _gather_impl(pmid, disp, conf, mesh_halo, 0, offset, None)
    mask = unused_index.reshape(unused_index.shape + (1,) * (val.ndim - 1))
    val = jnp.where(mask, jnp.zeros_like(val), val)
    return val, (pmid, disp, unused_index, mesh, incoming_left, incoming_right, offset)


def _gather_mGPU_mesh_halo_bwd(conf, res, val_cot):
    """Backward rule that reduces halo-mesh cotangents to owned cells."""
    pmid, disp, unused_index, mesh, incoming_left, incoming_right, offset = res
    mesh_halo = extend_owned_mesh_from_halo_edges(mesh, incoming_left, incoming_right, conf.mesh_halo_width)
    mask = unused_index.reshape(unused_index.shape + (1,) * (val_cot.ndim - 1))
    val_cot = jnp.where(mask, jnp.zeros_like(val_cot), val_cot)
    _, disp_cot, _, mesh_halo_cot, _, _, _ = _gather_bwd(
        (pmid, disp, conf, mesh_halo, offset, None),
        val_cot,
    )
    mesh_cot = reduce_mesh_halo_to_owned(
        mesh_halo_cot,
        conf.mesh_halo_width,
        conf.left_perm,
        conf.right_perm,
    )
    return None, disp_cot, None, mesh_cot


_gather_mGPU_mesh_halo.defvjp(_gather_mGPU_mesh_halo_fwd, _gather_mGPU_mesh_halo_bwd)


@partial(custom_vjp, nondiff_argnums=(3,))
def _gather_mGPU_particle_halo_mesh_edges(pmid, disp, unused_index, conf, mesh):
    """Gather particle-halo slots from exchanged mesh edges instead of exchanged values."""
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
        mesh,
        conf.mesh_halo_width,
        conf.left_perm,
        conf.right_perm,
    )
    mesh_halo = extend_owned_mesh_from_halo_edges(mesh, incoming_left, incoming_right, conf.mesh_halo_width)
    offset = conf.mesh_halo_offsets[gpu_id]
    val = _gather_impl(pmid, disp, conf, mesh_halo, 0, offset, None)
    mask = unused_index.reshape(unused_index.shape + (1,) * (val.ndim - 1))
    return jnp.where(mask, jnp.zeros_like(val), val)


def _gather_mGPU_particle_halo_mesh_edges_fwd(pmid, disp, unused_index, conf, mesh):
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
        mesh,
        conf.mesh_halo_width,
        conf.left_perm,
        conf.right_perm,
    )
    mesh_halo = extend_owned_mesh_from_halo_edges(mesh, incoming_left, incoming_right, conf.mesh_halo_width)
    offset = conf.mesh_halo_offsets[gpu_id]
    val = _gather_impl(pmid, disp, conf, mesh_halo, 0, offset, None)
    mask = unused_index.reshape(unused_index.shape + (1,) * (val.ndim - 1))
    val = jnp.where(mask, jnp.zeros_like(val), val)
    return val, (pmid, disp, unused_index, mesh, incoming_left, incoming_right, offset)


def _gather_mGPU_particle_halo_mesh_edges_bwd(conf, res, val_cot):
    pmid, disp, unused_index, mesh, incoming_left, incoming_right, offset = res
    mesh_halo = extend_owned_mesh_from_halo_edges(mesh, incoming_left, incoming_right, conf.mesh_halo_width)
    mask = unused_index.reshape(unused_index.shape + (1,) * (val_cot.ndim - 1))
    val_cot = jnp.where(mask, jnp.zeros_like(val_cot), val_cot)
    _, disp_cot, _, mesh_halo_cot, _, _, _ = _gather_bwd(
        (pmid, disp, conf, mesh_halo, offset, None),
        val_cot,
    )
    mesh_cot = reduce_mesh_halo_to_owned(
        mesh_halo_cot,
        conf.mesh_halo_width,
        conf.left_perm,
        conf.right_perm,
    )
    return None, disp_cot, None, mesh_cot


_gather_mGPU_particle_halo_mesh_edges.defvjp(
    _gather_mGPU_particle_halo_mesh_edges_fwd,
    _gather_mGPU_particle_halo_mesh_edges_bwd,
)


def _match_exchange_routing(local_left_slot, local_left_valid,
                            local_right_slot, local_right_valid,
                            incoming_valid_left, incoming_valid_right):
    """Match particle-halo exchange buffers to their local destination slots."""
    # Canonical storage builds left-halo copies directly from the neighbor's
    # sorted right-boundary export, and local right-boundary slots are packed in
    # that same sorted packed-key order. The compacted local and incoming halo
    # sequences therefore align slot-for-slot and do not need a key lookup.
    match_left = local_left_valid & incoming_valid_left
    update_indices_left = jnp.where(match_left, local_left_slot, 0)

    match_right = local_right_valid & incoming_valid_right
    update_indices_right = jnp.where(match_right, local_right_slot, 0)

    return (
        update_indices_left,
        match_left,
        update_indices_right,
        match_right,
    )


def _apply_exchange(val_in, pmid, disp, unused_index, conf, gpu_id, return_routing=False):
    """Forward halo exchange for gathered values on duplicated particle slots."""
    halo_start = conf.halo_start[gpu_id]
    halo_end = conf.halo_end[gpu_id]
    global_nMesh = conf.nMesh
    max_values_to_share = conf.max_share_gather_ptcl

    dummy_mask = unused_index
    x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % global_nMesh

    mask_shape = dummy_mask.shape + (1,) * (val_in.ndim - 1)
    val = jnp.where(dummy_mask.reshape(mask_shape), jnp.zeros_like(val_in), val_in)

    to_share_left = particles_in_slice_mask(x_mod, *halo_start) & ~dummy_mask
    to_share_right = particles_in_slice_mask(x_mod, *halo_end) & ~dummy_mask

    check_fraction_and_share = (
            (jnp.sum(to_share_right) > max_values_to_share) |
            (jnp.sum(to_share_left) > max_values_to_share)
    )

    _ = jax.lax.cond(
        check_fraction_and_share,
        lambda _: raise_error(
            "[ERROR] [GPU {a}] Exceeded max_values_to_share: "
            "to_share_right={x}, to_share_left={y}, max_share_gather_ptcl={z}. Some particles may have "
            f"disappeared during the simulation. Consider making 'conf.max_share_gather_ptcl' bigger so that this does not happen again.",
            a=jax.lax.axis_index('gpus'), x=jnp.sum(to_share_right), y=jnp.sum(to_share_left),
            z=max_values_to_share),
        lambda _: None,
        operand=None
    )

    fill_index = jnp.asarray(0, dtype=jnp.int32)
    left_count = jnp.sum(to_share_left)
    right_count = jnp.sum(to_share_right)
    left_idx = jnp.nonzero(to_share_left, size=max_values_to_share, fill_value=fill_index)[0]
    right_idx = jnp.nonzero(to_share_right, size=max_values_to_share, fill_value=fill_index)[0]
    to_share_left_valid = jnp.arange(max_values_to_share) < left_count
    to_share_right_valid = jnp.arange(max_values_to_share) < right_count

    val_valid_shape = (max_values_to_share,) + (1,) * (val.ndim - 1)
    to_share_left_val = jnp.where(
        to_share_left_valid.reshape(val_valid_shape),
        val[left_idx],
        jnp.zeros((max_values_to_share,) + val.shape[1:], dtype=val.dtype),
    )
    to_share_right_val = jnp.where(
        to_share_right_valid.reshape(val_valid_shape),
        val[right_idx],
        jnp.zeros((max_values_to_share,) + val.shape[1:], dtype=val.dtype),
    )
    to_share_left_src = jnp.where(to_share_left_valid, left_idx, jnp.asarray(0, left_idx.dtype))
    to_share_right_src = jnp.where(to_share_right_valid, right_idx, jnp.asarray(0, right_idx.dtype))

    incoming_from_left_val, incoming_valid_left = jax.lax.ppermute(
        (to_share_right_val, to_share_right_valid), axis_name=AXIS_NAME, perm=conf.right_perm)
    incoming_from_right_val, incoming_valid_right = jax.lax.ppermute(
        (to_share_left_val, to_share_left_valid), axis_name=AXIS_NAME, perm=conf.left_perm)

    (
        update_indices_left,
        match_left,
        update_indices_right,
        match_right,
    ) = _match_exchange_routing(
        to_share_left_src,
        to_share_left_valid,
        to_share_right_src,
        to_share_right_valid,
        incoming_valid_left,
        incoming_valid_right,
    )

    left_match_shape = match_left.shape + (1,) * (incoming_from_left_val.ndim - 1)
    right_match_shape = match_right.shape + (1,) * (incoming_from_right_val.ndim - 1)
    val = val.at[update_indices_left].add(incoming_from_left_val * match_left.reshape(left_match_shape).astype(val.dtype))
    val = val.at[update_indices_right].add(incoming_from_right_val * match_right.reshape(right_match_shape).astype(val.dtype))
    if return_routing:
        return val, (
            update_indices_left,
            match_left,
            update_indices_right,
            match_right,
            to_share_left_src,
            to_share_left_valid,
            to_share_right_src,
            to_share_right_valid,
        )
    return val


def _apply_exchange_bwd_from_routing(val_cot_in, routing, unused_index, conf):
    """Transpose the forward halo exchange using routing saved during forward."""
    (
        update_indices_left,
        match_left,
        update_indices_right,
        match_right,
        to_share_left_src,
        to_share_left_valid,
        to_share_right_src,
        to_share_right_valid,
    ) = routing
    dummy_mask = unused_index
    mask_shape = dummy_mask.shape + (1,) * (val_cot_in.ndim - 1)
    val_cot = jnp.where(dummy_mask.reshape(mask_shape), jnp.zeros_like(val_cot_in), val_cot_in)

    left_match_shape = match_left.shape + (1,) * (val_cot.ndim - 1)
    right_match_shape = match_right.shape + (1,) * (val_cot.ndim - 1)
    incoming_from_left_cot = val_cot[update_indices_left] * match_left.reshape(left_match_shape).astype(val_cot.dtype)
    incoming_from_right_cot = val_cot[update_indices_right] * match_right.reshape(right_match_shape).astype(val_cot.dtype)

    to_share_right_cot = jax.lax.ppermute(
        incoming_from_left_cot,
        axis_name=AXIS_NAME,
        perm=conf.left_perm,
    )
    to_share_left_cot = jax.lax.ppermute(
        incoming_from_right_cot,
        axis_name=AXIS_NAME,
        perm=conf.right_perm,
    )

    left_valid_shape = to_share_left_valid.shape + (1,) * (val_cot.ndim - 1)
    right_valid_shape = to_share_right_valid.shape + (1,) * (val_cot.ndim - 1)
    val_cot = val_cot.at[to_share_left_src].add(to_share_left_cot * to_share_left_valid.reshape(left_valid_shape).astype(val_cot.dtype))
    val_cot = val_cot.at[to_share_right_src].add(to_share_right_cot * to_share_right_valid.reshape(right_valid_shape).astype(val_cot.dtype))

    return jnp.where(dummy_mask.reshape(mask_shape), jnp.zeros_like(val_cot), val_cot)


@partial(custom_vjp, nondiff_argnums=(3,))
def _gather_mGPU(pmid, disp, unused_index, conf, mesh):
    """Gather in ``particle_halo`` mode and exchange values for halo slots."""
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]

    val = _gather_impl(pmid, disp, conf, mesh, 0, offset, None)
    return _apply_exchange(val, pmid, disp, unused_index, conf, gpu_id)


def _gather_mGPU_fwd(pmid, disp, unused_index, conf, mesh):
    """Forward rule that saves particle-halo exchange routing for the VJP."""
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]

    val = _gather_impl(pmid, disp, conf, mesh, 0, offset, None)
    val, routing = _apply_exchange(val, pmid, disp, unused_index, conf, gpu_id, return_routing=True)
    return val, (pmid, disp, unused_index, mesh, routing)


def _gather_mGPU_bwd(conf, res, val_cot):
    """Backward rule for particle-halo gather."""
    pmid, disp, unused_index, mesh, routing = res
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]
    local_val_cot = _apply_exchange_bwd_from_routing(val_cot, routing, unused_index, conf)

    _, disp_cot, _, mesh_cot, _, _, _ = _gather_bwd((pmid, disp, conf, mesh, offset, None), local_val_cot)
    return None, disp_cot, None, mesh_cot


_gather_mGPU.defvjp(_gather_mGPU_fwd, _gather_mGPU_bwd)


def gather(ptcl, conf, mesh):
    """Gather particle values from mesh multilinearly in n-D.

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    mesh : ArrayLike
        Input mesh.
    val : ArrayLike, optional
        Input values, can be 0D.
    offset : ArrayLike, optional
        Offset of mesh to particle grid. If 0D, the value is used in each dimension.
    cell_size : float, optional
        Mesh cell size in [L]. Default is ``conf.cell_size``.

    Returns
    -------
    val : jax.Array
        Output values.

    Notes
    -----
    On the mGPU path the output is defined on the duplicated particle-slot state,
    including halo copies. Comparisons against PMWD's unique-particle state should
    therefore map values to unique particles, and should sum PMPP ``disp``
    cotangents across duplicate halo slots.

    """
    pmid = jax.lax.stop_gradient(ptcl.pmid)
    disp = ptcl.disp

    if not conf.use_mGPU or conf.mGPU_gather is None:
        return _gather(pmid, disp, conf, mesh, 0, 0, None)

    unused_index = jax.lax.stop_gradient(ptcl.unused_index)
    return conf.mGPU_gather(pmid, disp, unused_index, conf, mesh)


def gather_stacked_mesh_halo(ptcl, conf, mesh_channels):
    """Gather multiple mesh channels after a single mesh-halo edge exchange."""
    if (
        not conf.use_mGPU
        or conf.compute_mesh is None
        or conf.multigpu_mode != "mesh_halo"
    ):
        return jnp.stack(
            [gather(ptcl, conf, mesh_channels[..., i]) for i in range(mesh_channels.shape[-1])],
            axis=-1,
        )

    pmid = jax.lax.stop_gradient(ptcl.pmid)
    disp = ptcl.disp
    unused_index = jax.lax.stop_gradient(ptcl.unused_index)

    @partial(
        shard_map,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),
            P(AXIS_NAME, None),
            P(AXIS_NAME),
            None,
            P(AXIS_NAME, None, None, None),
        ),
        out_specs=P(AXIS_NAME, None),
        check_rep=False,
    )
    def _gather_stacked_local(pmid_local, disp_local, unused_local, conf_local, mesh_channels_local):
        gpu_id = jax.lax.axis_index(AXIS_NAME)
        incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
            mesh_channels_local,
            conf_local.mesh_halo_width,
            conf_local.left_perm,
            conf_local.right_perm,
        )
        mesh_halo = extend_owned_mesh_from_halo_edges(
            mesh_channels_local,
            incoming_left,
            incoming_right,
            conf_local.mesh_halo_width,
        )
        offset = conf_local.mesh_halo_offsets[gpu_id]
        gathered = [
            _gather(pmid_local, disp_local, conf_local, mesh_halo[..., i], 0, offset, None)
            for i in range(mesh_halo.shape[-1])
        ]
        val = jnp.stack(gathered, axis=-1)
        mask = unused_local.reshape(unused_local.shape + (1,) * (val.ndim - 1))
        return jnp.where(mask, jnp.zeros_like(val), val)

    return _gather_stacked_local(pmid, disp, unused_index, conf, mesh_channels)


def _gather_impl(pmid, disp, conf, mesh, val, offset, cell_size):
    """Local multilinear gather implementation, optionally chunked."""
    ptcl_num, spatial_ndim = pmid.shape

    mesh = jnp.asarray(mesh, dtype=conf.float_dtype)
    channel_shape = mesh.shape[spatial_ndim:]
    val = jnp.zeros((ptcl_num,) + channel_shape, dtype=conf.float_dtype)

    if mesh.shape[spatial_ndim:] != val.shape[1:]:
        raise ValueError('channel shape mismatch: '
                         f'{mesh.shape[spatial_ndim:]} != {val.shape[1:]}')

    carry = mesh, offset, cell_size, conf.cell_size, conf.mesh_shape
    if conf.chunk_size is None or conf.chunk_size >= ptcl_num:
        return _gather_chunk(carry, (pmid, disp, val))[1]

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    val_0 = None
    if remainder is not None:
        val_0 = _gather_chunk(carry, remainder)[1]
    val = scan(_gather_chunk, carry, chunks)[1]

    return _chunk_cat(val_0, val)


@custom_vjp
def _gather(pmid, disp, conf, mesh, val, offset, cell_size):
    """Local gather primitive with custom VJP."""
    return _gather_impl(pmid, disp, conf, mesh, val, offset, cell_size)


def _gather_chunk(carry, chunk):
    """Gather one particle chunk from the carried mesh."""
    mesh, offset, cell_size, conf_cell_size, conf_mesh_shape = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, conf_cell_size, conf_mesh_shape,
                       offset, cell_size, spatial_shape, False)

    # gather
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    # += usually, but since val is always 0, now =
    gathered = mesh.at[ind].get(mode='drop', fill_value=0)
    channel_ndim = gathered.ndim - frac.ndim
    frac_shape = frac.shape + (1,) * channel_ndim
    val = (gathered * frac.reshape(frac_shape)).sum(axis=1)

    return carry, val


def _gather_chunk_adj(carry, chunk):
    """Adjoint of `_gather_chunk`, or equivalently `_gather_adj_chunk`, i.e.
    gather adjoint in chunks

    Gather disp_cot from val_cot and mesh;
    Scatter val_cot to mesh_cot.

    """
    mesh, mesh_cot, offset, cell_size, conf_cell_size, conf_mesh_shape = carry
    pmid, disp, val_cot = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]

    # multilinear mesh indices and fractions
    ind, frac, frac_grad = enmesh(pmid, disp, conf_cell_size, conf_mesh_shape,
                                  offset, cell_size, spatial_shape, True)

    # gather disp_cot from val_cot and mesh, and scatter val_cot to mesh_cot
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    val = mesh.at[ind].get(mode='drop', fill_value=0)

    val_cot = val_cot[:, jnp.newaxis, ...]
    inner = val_cot * val
    if inner.ndim > 2:
        inner = inner.sum(axis=tuple(range(2, inner.ndim)))
    disp_cot = (inner[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size if cell_size is not None else conf_cell_size

    frac_shape = frac.shape + (1,) * (val.ndim - frac.ndim)
    mesh_cot = mesh_cot.at[ind].add(val_cot * frac.reshape(frac_shape))

    carry = mesh, mesh_cot, offset, cell_size, conf_cell_size, conf_mesh_shape
    return carry, disp_cot



def _gather_fwd(pmid, disp, conf, mesh, val, offset, cell_size):
    """Forward rule for the local gather primitive."""
    val_out = _gather_impl(pmid, disp, conf, mesh, val, offset, cell_size)
    return val_out, (pmid, disp, conf, mesh, offset, cell_size)


def _gather_bwd(res, val_cot):
    """Backward rule for the local gather primitive."""
    pmid, disp, conf, mesh, offset, cell_size = res

    ptcl_num = len(pmid)

    mesh = jnp.asarray(mesh, dtype=conf.float_dtype)
    mesh_cot = jnp.zeros_like(mesh)

    carry = mesh, mesh_cot, offset, cell_size, conf.cell_size, conf.mesh_shape
    if conf.chunk_size is None or conf.chunk_size >= ptcl_num:
        carry, disp_cot = _gather_chunk_adj(carry, (pmid, disp, val_cot))
        mesh_cot = carry[1]
    else:
        remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val_cot)

        disp_cot_0 = None
        if remainder is not None:
            carry, disp_cot_0 = _gather_chunk_adj(carry, remainder)
        carry, disp_cot = scan(_gather_chunk_adj, carry, chunks)
        mesh_cot = carry[1]

        disp_cot = _chunk_cat(disp_cot_0, disp_cot)

    return None, disp_cot, None, mesh_cot, None, None, None


_gather.defvjp(_gather_fwd, _gather_bwd)
