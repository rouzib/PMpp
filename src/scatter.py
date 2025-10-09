from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.lax import scan
from jax.sharding import PartitionSpec as P

from .enmesh import _chunk_split, enmesh, _chunk_cat
from .utils import AXIS_NAME


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
            P(AXIS_NAME, None),  # offset
            None,  # cell_size
        ),
        out_specs=(P(AXIS_NAME, None, None)),
        check_rep=False
    )


def _scatter_mGPU(pmid, disp, conf, mesh, val, offset, cell_size):
    result = _scatter(pmid, disp, conf, mesh, val, offset, cell_size)
    return result


@partial(jax.jit, static_argnames=("mesh_shape", "mesh_dtype", "compute_mesh", "val_shape"))
def _initialize_mesh_on_devices(mesh_shape, mesh_dtype, compute_mesh, val_shape):
    """
    Directly initialize the mesh on the devices. Each device instantiates its local
    fraction of the mesh so that the concatenation of the mesh fractions of all devices
    gives the total mesh.
    :param conf: Configuration
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
    return conf.mGPU_scatter(ptcl.pmid, ptcl.disp, conf, mesh, val, conf.scatter_offsets, cell_size)


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

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    carry = mesh, offset, cell_size, conf.cell_size, conf.mesh_shape
    if remainder is not None:
        carry = _scatter_chunk(carry, remainder)[0]
    carry = scan(_scatter_chunk, carry, chunks)[0]

    mesh = carry[0]
    return mesh


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

    ptcl_num = len(pmid)

    if val is None:
        val = conf.mesh_size / conf.ptcl_num
    val = jnp.asarray(val, dtype=conf.float_dtype)

    remainder, chunks = _chunk_split(ptcl_num, conf.chunk_size, pmid, disp, val)

    carry = mesh_cot, offset, cell_size, conf.cell_size, conf.mesh_shape
    disp_cot_0, val_cot_0 = None, None
    if remainder is not None:
        disp_cot_0, val_cot_0 = _scatter_chunk_adj(carry, remainder)[1]
    disp_cot, val_cot = scan(_scatter_chunk_adj, carry, chunks)[1]

    disp_cot = _chunk_cat(disp_cot_0, disp_cot)
    val_cot = _chunk_cat(val_cot_0, val_cot)

    return None, disp_cot, None, mesh_cot, val_cot, None, None


_scatter.defvjp(_scatter_fwd, _scatter_bwd)