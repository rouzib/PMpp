from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.lax import scan
from jax.sharding import PartitionSpec as P

from .enmesh import _chunk_split, enmesh, _chunk_cat
from .particles import Particles
from .utils import AXIS_NAME, raise_error


def initialize_mGPU_gather(conf):
    return shard_map(
        _gather_mGPU,
        mesh=conf.compute_mesh,
        in_specs=(
            P(AXIS_NAME, None),  # pmid
            P(AXIS_NAME, None),  # disp
            P(AXIS_NAME),  # particle_indices
            None,  # conf
            P(AXIS_NAME, None, None),  # mesh
        ),
        out_specs=P(AXIS_NAME),
        # (
        #             P(AXIS_NAME),  # read_values: [n_particles]
        #             P(),  # has_failed
        #             P()  # max_particles_moved
        #         )
        check_rep=True
    )


def _apply_exchange(val_in, pmid_sg, disp_sg, particle_indices_sg, conf_sg, gpu_id):
    """Pure forward halo-exchange on a STOP-GRAD input.
    Returns the exchanged values; no gradients should flow through this.
    """
    # --- local constants / slices ---
    halo_start = conf_sg.halo_start[gpu_id]
    halo_end = conf_sg.halo_end[gpu_id]
    global_nMesh = conf_sg.nMesh
    halo_start_fix = [halo_start[0], (halo_start[1] - 1) % global_nMesh]
    max_values_to_share = conf_sg.max_share_gather_ptcl

    # masks (ints/bools, forward-only)
    dummy_mask = (particle_indices_sg == -1)
    x_mod = (pmid_sg[:, 0] + disp_sg[:, 0] * conf_sg.disp_size) % global_nMesh

    val = jnp.where(dummy_mask, jnp.asarray(0, val_in.dtype), val_in)

    to_share_left = Particles.particles_in_slice_mask(x_mod, *halo_start_fix) & ~dummy_mask
    to_share_right = Particles.particles_in_slice_mask(x_mod, *halo_end) & ~dummy_mask

    # diagnostics (forward-only side effect)
    check_fraction_and_share = (
            (jnp.sum(to_share_right) > max_values_to_share) |
            (jnp.sum(to_share_left) > max_values_to_share)
    )

    _ = jax.lax.cond(
        check_fraction_and_share,
        lambda _: raise_error(
            "[ERROR] [GPU {a}] Exceeded max_values_to_share: "
            "to_share_right={x}, to_share_left={y}, max_share_gather_ptcl={z}. "
            "Some particles may have disappeared during the simulation. "
            "Consider making 'conf.max_share_gather_ptcl' bigger so that this does not happen again.",
            a=jax.lax.axis_index(AXIS_NAME),
            x=jnp.sum(to_share_right), y=jnp.sum(to_share_left), z=max_values_to_share),
        lambda _: None,
        operand=None
    )

    # --- pack share payloads with explicit dtypes for fill_value ---
    to_share_left_idx = jnp.compress(
        to_share_left, particle_indices_sg, axis=0,
        size=max_values_to_share,
        fill_value=jnp.asarray(-1, particle_indices_sg.dtype)
    )
    to_share_right_idx = jnp.compress(
        to_share_right, particle_indices_sg, axis=0,
        size=max_values_to_share,
        fill_value=jnp.asarray(-1, particle_indices_sg.dtype)
    )

    to_share_left_val = jnp.compress(
        to_share_left, val, axis=0, size=max_values_to_share,
        fill_value=jnp.asarray(0, val.dtype)
    )
    to_share_right_val = jnp.compress(
        to_share_right, val, axis=0, size=max_values_to_share,
        fill_value=jnp.asarray(0, val.dtype)
    )

    # --- exchange ---
    incoming_idx_left, incoming_from_left_val = jax.lax.ppermute(
        (to_share_right_idx, to_share_right_val), axis_name=AXIS_NAME, perm=conf_sg.right_perm)
    incoming_idx_right, incoming_from_right_val = jax.lax.ppermute(
        (to_share_left_idx, to_share_left_val), axis_name=AXIS_NAME, perm=conf_sg.left_perm)

    # --- match incoming to local positions ---
    sorted_local_positions = jnp.sort(particle_indices_sg)
    sorted_local_indices = jnp.argsort(particle_indices_sg)

    matching_indices_sorted = jnp.searchsorted(sorted_local_positions, incoming_idx_left, method="sort")
    update_indices_left = sorted_local_indices[matching_indices_sorted]

    matching_indices_sorted = jnp.searchsorted(sorted_local_positions, incoming_idx_right, method="sort")
    update_indices_right = sorted_local_indices[matching_indices_sorted]

    # --- apply updates on a LOCAL copy (val_out) ---
    val_out = val
    val_out = val_out.at[update_indices_left].add(incoming_from_left_val)
    val_out = val_out.at[update_indices_right].add(incoming_from_right_val)

    return val_out


@partial(custom_vjp, nondiff_argnums=(3, ))
def _gather_mGPU(pmid, disp, particle_indices, conf, mesh):
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]

    val = _gather_impl(pmid, disp, conf, mesh, 0.0, offset, None)

    val = _apply_exchange(val, pmid, disp, particle_indices, conf, gpu_id)

    return val


def _gather_mGPU_fwd(pmid, disp, particle_indices, conf, mesh):
    val = _gather_mGPU(pmid, disp, particle_indices, conf, mesh)
    return val, (pmid, disp, particle_indices, mesh)


def _gather_mGPU_bwd(conf, res, val_cot):
    pmid, disp, particle_indices, mesh = res
    gpu_id = jax.lax.axis_index(AXIS_NAME)
    offset = conf.scatter_offsets[gpu_id]

    _, disp_cot, _, mesh_cot, val_cot, _, _ = _gather_bwd((pmid, disp, conf, mesh, offset, None), val_cot)

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

    """
    # return _gather(ptcl.pmid, ptcl.disp, conf, mesh, 0.0, 0.0, conf.cell_size)
    # return _gather_mGPU(ptcl.pmid, ptcl.disp, ptcl.idx, conf, mesh)
    return conf.mGPU_gather(ptcl.pmid, ptcl.disp, ptcl.idx, conf, mesh)


def _gather_impl(pmid, disp, conf, mesh, val, offset, cell_size):
    ptcl_num = len(pmid)
    val = jnp.asarray(val, dtype=conf.float_dtype)

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    carry = mesh, offset, cell_size, conf.cell_size, conf.mesh_shape
    val_0 = None
    if remainder is not None:
        val_0 = _gather_chunk(carry, remainder)[1]
    val = scan(_gather_chunk, carry, chunks)[1]

    val = _chunk_cat(val_0, val)

    return val


def _gather_chunk(carry, chunk):
    mesh, offset, cell_size, conf_cell_size, conf_mesh_shape = carry
    pmid, disp, val = chunk

    spatial_ndim = 3

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac = enmesh(pmid, disp, conf_cell_size, conf_mesh_shape,
                       offset, cell_size, spatial_shape, False)

    # gather
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    frac = jnp.expand_dims(frac, chan_axis)
    # += usually, but since val is always 0, now =
    val = (mesh.at[ind].get(mode='drop', fill_value=0) * frac).sum(axis=1)

    return carry, val


def _gather_chunk_adj(carry, chunk):
    """Adjoint of `_gather_chunk`, or equivalently `_gather_adj_chunk`, i.e.
    gather adjoint in chunks

    Gather disp_cot from val_cot and mesh;
    Scatter val_cot to mesh_cot.

    """
    mesh, mesh_cot, offset, cell_size, conf_cell_size, conf_mesh_shape = carry
    pmid, disp, val_cot = chunk

    spatial_ndim = 3

    spatial_shape = mesh.shape[:spatial_ndim]
    chan_ndim = mesh.ndim - spatial_ndim
    chan_axis = tuple(range(-chan_ndim, 0))

    # multilinear mesh indices and fractions
    ind, frac, frac_grad = enmesh(pmid, disp, conf_cell_size, conf_mesh_shape,
                                  offset, cell_size, spatial_shape, True)

    if val_cot.ndim != 0:
        val_cot = val_cot[:, jnp.newaxis]  # insert neighbor axis

    # gather disp_cot from val_cot and mesh, and scatter val_cot to mesh_cot
    ind = tuple(ind[..., i] for i in range(spatial_ndim))
    val = mesh.at[ind].get(mode='drop', fill_value=0)

    disp_cot = (val_cot * val).sum(axis=chan_axis)
    disp_cot = (disp_cot[..., jnp.newaxis] * frac_grad).sum(axis=1)
    disp_cot /= cell_size if cell_size is not None else conf_cell_size

    frac = jnp.expand_dims(frac, chan_axis)
    mesh_cot = mesh_cot.at[ind].add(val_cot * frac)

    carry = mesh, mesh_cot, offset, cell_size, conf_cell_size, conf_mesh_shape
    return carry, disp_cot


@custom_vjp
def _gather(pmid, disp, conf, mesh, val, offset, cell_size):
    return _gather_impl(pmid, disp, conf, mesh, val, offset, cell_size)


def _gather_fwd(pmid, disp, conf, mesh, val, offset, cell_size):
    val_out = _gather_impl(pmid, disp, conf, mesh, val, offset, cell_size)
    return val_out, (pmid, disp, conf, mesh, offset, cell_size)


def _gather_bwd(res, val_cot):
    pmid, disp, conf, mesh, offset, cell_size = res

    ptcl_num = len(pmid)

    # Make zeros with the same local shard shape…
    mesh_cot = jnp.zeros_like(mesh)
    # …and mark it as "varying" along the manual axis so it matches the scan body output:
    mesh_cot = jax.lax.pvary(mesh_cot, (AXIS_NAME,))

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val_cot)

    carry = mesh, mesh_cot, offset, cell_size, conf.cell_size, conf.mesh_shape
    disp_cot_0 = None
    if remainder is not None:
        carry, disp_cot_0 = _gather_chunk_adj(carry, remainder)
    carry, disp_cot = scan(_gather_chunk_adj, carry, chunks)
    mesh_cot = carry[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)

    return None, disp_cot, None, mesh_cot, val_cot, None, None


_gather.defvjp(_gather_fwd, _gather_bwd)
