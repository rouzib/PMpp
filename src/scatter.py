from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental.shard_map import shard_map
from jax.lax import scan
from jax.sharding import NamedSharding, PartitionSpec as P

from .enmesh import _chunk_split, enmesh, _chunk_cat
from .halo_moving import particles_in_slice_mask
from .utils import AXIS_NAME, raise_error, pmid_to_idx


def reduce_grad_across_gpus(disp_cot, pmid, disp, valid_mask, conf):
    """Sum gradients for halo-duplicated particles across neighboring GPUs only."""
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    halo_start = conf.halo_start[gpu_id]
    halo_end = conf.halo_end[gpu_id]
    global_nmesh = conf.nMesh
    max_values_to_share = conf.max_share_gather_ptcl

    x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % global_nmesh
    to_share_left = particles_in_slice_mask(x_mod, *halo_start) & valid_mask
    to_share_right = particles_in_slice_mask(x_mod, *halo_end) & valid_mask

    check_fraction_and_share = (
        (jnp.sum(to_share_right) > max_values_to_share) |
        (jnp.sum(to_share_left) > max_values_to_share)
    )

    _ = jax.lax.cond(
        check_fraction_and_share,
        lambda _: raise_error(
            "[ERROR] [GPU {a}] Exceeded max_values_to_share in scatter backward: "
            "to_share_right={x}, to_share_left={y}, max_share_gather_ptcl={z}. "
            "Consider making 'conf.max_share_gather_ptcl' bigger.",
            a=gpu_id,
            x=jnp.sum(to_share_right),
            y=jnp.sum(to_share_left),
            z=max_values_to_share,
        ),
        lambda _: None,
        operand=None,
    )

    to_share_left_pmid = jnp.compress(
        to_share_left, pmid, axis=0, size=max_values_to_share, fill_value=jnp.asarray(0, pmid.dtype)
    )
    to_share_right_pmid = jnp.compress(
        to_share_right, pmid, axis=0, size=max_values_to_share, fill_value=jnp.asarray(0, pmid.dtype)
    )
    to_share_left_valid = jnp.compress(
        to_share_left, valid_mask, axis=0, size=max_values_to_share, fill_value=jnp.asarray(False, jnp.bool_)
    )
    to_share_right_valid = jnp.compress(
        to_share_right, valid_mask, axis=0, size=max_values_to_share, fill_value=jnp.asarray(False, jnp.bool_)
    )
    to_share_left_grad = jnp.compress(
        to_share_left, disp_cot, axis=0, size=max_values_to_share, fill_value=jnp.asarray(0, disp_cot.dtype)
    )
    to_share_right_grad = jnp.compress(
        to_share_right, disp_cot, axis=0, size=max_values_to_share, fill_value=jnp.asarray(0, disp_cot.dtype)
    )

    incoming_pmid_left, incoming_grad_left, incoming_valid_left = jax.lax.ppermute(
        (to_share_right_pmid, to_share_right_grad, to_share_right_valid),
        axis_name=AXIS_NAME,
        perm=conf.right_perm,
    )
    incoming_pmid_right, incoming_grad_right, incoming_valid_right = jax.lax.ppermute(
        (to_share_left_pmid, to_share_left_grad, to_share_left_valid),
        axis_name=AXIS_NAME,
        perm=conf.left_perm,
    )

    slot_index = jnp.arange(pmid.shape[0], dtype=jnp.int32)

    local_left_pmid = jnp.compress(
        to_share_left, pmid, axis=0, size=max_values_to_share, fill_value=jnp.asarray(0, pmid.dtype)
    )
    local_right_pmid = jnp.compress(
        to_share_right, pmid, axis=0, size=max_values_to_share, fill_value=jnp.asarray(0, pmid.dtype)
    )
    local_left_slot = jnp.compress(
        to_share_left, slot_index, axis=0, size=max_values_to_share, fill_value=jnp.asarray(-1, slot_index.dtype)
    )
    local_right_slot = jnp.compress(
        to_share_right, slot_index, axis=0, size=max_values_to_share, fill_value=jnp.asarray(-1, slot_index.dtype)
    )

    missing_key = jnp.asarray(conf.mesh_size, dtype=jnp.int32)
    local_left_keys = jnp.where(
        local_left_slot >= 0,
        pmid_to_idx(local_left_pmid, conf),
        missing_key,
    )
    local_right_keys = jnp.where(
        local_right_slot >= 0,
        pmid_to_idx(local_right_pmid, conf),
        missing_key,
    )

    incoming_left_keys = jnp.where(incoming_valid_left, pmid_to_idx(incoming_pmid_left, conf), missing_key)
    incoming_right_keys = jnp.where(incoming_valid_right, pmid_to_idx(incoming_pmid_right, conf), missing_key)

    # The canonical mover packs the local halo subsets and the incoming neighbor
    # exports in the same sorted packed-key order, so the compacted sequences
    # align slot-for-slot. A direct positional match is enough here.
    matched_left = (
        incoming_valid_left
        & (local_left_slot >= 0)
        & (local_left_keys == incoming_left_keys)
    )
    update_indices_left = jnp.where(matched_left, local_left_slot, 0)

    matched_right = (
        incoming_valid_right
        & (local_right_slot >= 0)
        & (local_right_keys == incoming_right_keys)
    )
    update_indices_right = jnp.where(matched_right, local_right_slot, 0)

    disp_cot = disp_cot.at[update_indices_left].add(incoming_grad_left * matched_left[:, None].astype(disp_cot.dtype))
    disp_cot = disp_cot.at[update_indices_right].add(incoming_grad_right * matched_right[:, None].astype(disp_cot.dtype))

    return jnp.where(valid_mask[:, None], disp_cot, 0)


def initialize_mGPU_scatter(conf):
    return shard_map(
        _scatter_mGPU,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),  # pmid
            P(AXIS_NAME, None),  # disp
            None,  # conf
            P(AXIS_NAME, None, None),  # mesh
            P(AXIS_NAME),  # val
            None,  # cell_size
        ),
        out_specs=(P(AXIS_NAME, None, None)),
        check_rep=False
    )


@partial(custom_vjp, nondiff_argnums=(2,))
def _scatter_mGPU(pmid, disp, conf, mesh, val, cell_size):
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]
    return _scatter(pmid, disp, conf, mesh, val, offset, cell_size)


def _scatter_mGPU_fwd(pmid, disp, conf, mesh, val, cell_size):
    result = _scatter_mGPU(pmid, disp, conf, mesh, val, cell_size)
    return result, (pmid, disp, mesh, val, cell_size)


def _scatter_mGPU_bwd(conf, res, mesh_cot):
    pmid, disp, mesh, val, cell_size_res = res
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]

    _, disp_cot, _, mesh_in_cot, val_cot, _, _ = _scatter_bwd(
        (pmid, disp, conf, val, offset, cell_size_res),
        mesh_cot,
    )
    return None, disp_cot, mesh_in_cot, val_cot, None


_scatter_mGPU.defvjp(_scatter_mGPU_fwd, _scatter_mGPU_bwd)


@partial(jax.jit, static_argnames=("mesh_shape", "mesh_dtype", "compute_mesh", "val_shape"))
def _initialize_mesh_on_devices(mesh_shape, mesh_dtype, compute_mesh, val_shape):
    """
    Directly initialize the mesh on the devices. Each device instantiates its local
    fraction of the mesh so that the concatenation of the mesh fractions of all devices
    gives the total mesh.
    :return:
    """
    mesh = jnp.zeros(mesh_shape + val_shape[1:], dtype=mesh_dtype)
    return jax.lax.with_sharding_constraint(mesh, NamedSharding(compute_mesh, P(AXIS_NAME, None, None)))


def scatter(ptcl, conf, mesh=None, val=None, offset=0, cell_size=None):
    """Scatter particle values to mesh multilinearly in n-D.

    Parameters
    ----------
    ptcl : Particles
    conf : Configuration
    mesh : ArrayLike, optional
        Input mesh. Default is a ``zeros`` array of ``conf.mesh_shape + val.shape[1:]``.
    val : ArrayLike, optional
        Input values, can be 0D. Default is ``conf.mesh_size / conf.ptcl_num``.
    offset : ArrayLike, optional
        Offset of mesh to particle grid. If 0D, the value is used in each dimension.
    cell_size : float, optional
        Mesh cell size in [L]. Default is ``conf.cell_size``.

    Returns
    -------
    mesh : jax.Array
        Output mesh.

    """

    if val is None:
        val = conf.mesh_size / conf.ptcl_num
        val = (~ptcl.unused_index).astype(conf.float_dtype) * val

    if mesh is None:
        mesh = _initialize_mesh_on_devices(conf.mesh_shape, conf.float_dtype, conf.compute_mesh, val.shape)
    return conf.mGPU_scatter(ptcl.pmid, ptcl.disp, conf, mesh, val, cell_size)


@custom_vjp
def _scatter(pmid, disp, conf, mesh, val, offset, cell_size):
    ptcl_num, spatial_ndim = pmid.shape

    if val is None:
        val = conf.mesh_size / conf.ptcl_num
    val = jnp.asarray(val, dtype=conf.float_dtype)

    if mesh is None:
        mesh = jnp.zeros(conf.local_mesh_shape + val.shape[1:], dtype=conf.float_dtype)
    mesh = jnp.asarray(mesh, dtype=conf.float_dtype)

    if mesh.shape[spatial_ndim:] != val.shape[1:]:
        raise ValueError('channel shape mismatch: '
                         f'{mesh.shape[spatial_ndim:]} != {val.shape[1:]}')

    carry = mesh, offset, cell_size, conf.cell_size, conf.mesh_shape
    if conf.chunk_size is None or conf.chunk_size >= ptcl_num:
        return _scatter_chunk(carry, (pmid, disp, val))[0][0]

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)
    if remainder is not None:
        carry = _scatter_chunk(carry, remainder)[0]
    return scan(_scatter_chunk, carry, chunks)[0][0]


def _scatter_chunk(carry, chunk):
    mesh, offset, cell_size, conf_cell_size, conf_mesh_shape = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, conf_cell_size, conf_mesh_shape,
                       offset, cell_size, spatial_shape, False)

    if val.ndim != 0:
        val = val[:, jnp.newaxis]  # insert neighbor axis

    # scatter
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    frac = jnp.expand_dims(frac, chan_axis)
    mesh = mesh.at[ind].add(val * frac)

    carry = mesh, offset, cell_size, conf_cell_size, conf_mesh_shape
    return carry, None


def _scatter_chunk_adj(carry, chunk):
    """Adjoint of `_scatter_chunk`, or equivalently `_scatter_adj_chunk`, i.e. scatter
    adjoint in chunks.

    Gather disp_cot from mesh_cot and val;
    Gather val_cot from mesh_cot.

    """
    mesh_cot, offset, cell_size, conf_cell_size, conf_mesh_shape = carry
    pmid, disp, val = chunk

    spatial_ndim = pmid.shape[1]

    spatial_shape = mesh_cot.shape[:spatial_ndim]
    chan_ndim = mesh_cot.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac, frac_grad = enmesh(pmid, disp, conf_cell_size, conf_mesh_shape,
                                  offset, cell_size, spatial_shape, True)

    if val.ndim != 0:
        val = val[:, jnp.newaxis]  # insert neighbor axis

    # gather disp_cot from mesh_cot and val, and gather val_cot from mesh_cot
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    val_cot = mesh_cot.at[ind].get(mode='drop', fill_value=0)

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size if cell_size is not None else conf_cell_size

    frac = jnp.expand_dims(frac, chan_axis)
    val_cot = (val_cot * frac).sum(axis=1)

    return carry, (disp_cot, val_cot)

def _scatter_fwd(pmid, disp, conf, mesh, val, offset, cell_size):
    mesh = _scatter(pmid, disp, conf, mesh, val, offset, cell_size)
    return mesh, (pmid, disp, conf, val, offset, cell_size)


def _scatter_bwd(res, mesh_cot):
    pmid, disp, conf, val, offset, cell_size = res

    local_ptcl_num = len(pmid)

    if val is None:
        val = conf.mesh_size / conf.ptcl_num
    val = jnp.asarray(val, dtype=conf.float_dtype)

    carry = mesh_cot, offset, cell_size, conf.cell_size, conf.mesh_shape
    if conf.chunk_size is None or conf.chunk_size >= local_ptcl_num:
        disp_cot, val_cot = _scatter_chunk_adj(carry, (pmid, disp, val))[1]
    else:
        remainder, chunks = _chunk_split(local_ptcl_num, conf.chunk_size, pmid, disp, val)

        disp_cot_0, val_cot_0 = None, None
        if remainder is not None:
            disp_cot_0, val_cot_0 = _scatter_chunk_adj(carry, remainder)[1]
        disp_cot, val_cot = scan(_scatter_chunk_adj, carry, chunks)[1]

        disp_cot = _chunk_cat(disp_cot_0, disp_cot)
        val_cot = _chunk_cat(val_cot_0, val_cot)

    # The standalone scatter primitive is defined on the duplicated slot state.
    # Keep the backward local to each slot and let higher-level callers decide
    # when duplicate halo-slot cotangents should be aggregated back to unique
    # particles.
    val_nonzero = jnp.asarray(val != 0)
    if val_nonzero.ndim == 0:
        disp_cot = jnp.where(val_nonzero, disp_cot, 0)
    else:
        disp_cot = jnp.where(val_nonzero[:, None], disp_cot, 0)

    return None, disp_cot, None, mesh_cot, val_cot, None, None


_scatter.defvjp(_scatter_fwd, _scatter_bwd)
