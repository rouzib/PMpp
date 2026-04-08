from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .utils import AXIS_NAME


def owned_mesh_partition_spec(ndim: int):
    """Partition a mesh-local array along the owned x-slab axis."""
    if ndim < 1:
        raise ValueError(f"owned_mesh_partition_spec expects ndim >= 1, got {ndim}.")
    return P(AXIS_NAME, *([None] * (ndim - 1)))


def maybe_shard_map_mesh_local_op(local_fn, conf, in_specs, out_specs, check_rep=False):
    """Bind AXIS_NAME for mesh-local operators that use halo collectives internally."""
    if conf.compute_mesh is None or conf.num_devices == 1:
        return local_fn
    return shard_map(
        local_fn,
        mesh=conf.compute_mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=check_rep,
    )


def zero_pad_owned_mesh_halo(mesh_owned, halo_width: int):
    """Embed an owned mesh shard in a zero halo shell along the decomposed x-axis."""
    if halo_width <= 0:
        return mesh_owned
    pad_width = ((halo_width, halo_width),) + ((0, 0),) * (mesh_owned.ndim - 1)
    return jnp.pad(mesh_owned, pad_width)


def exchange_owned_mesh_halo_edges(mesh_owned, halo_width: int, left_perm, right_perm):
    """Exchange the x-edge slabs needed to build a halo shell."""
    if halo_width <= 0:
        empty = mesh_owned[:0]
        return empty, empty

    left_owned = mesh_owned[:halo_width]
    right_owned = mesh_owned[-halo_width:]
    incoming_left = jax.lax.ppermute(right_owned, axis_name=AXIS_NAME, perm=right_perm)
    incoming_right = jax.lax.ppermute(left_owned, axis_name=AXIS_NAME, perm=left_perm)
    return incoming_left, incoming_right


def extend_owned_mesh_from_halo_edges(mesh_owned, incoming_left, incoming_right, halo_width: int):
    """Build a halo-extended mesh from pre-exchanged edge slabs."""
    if halo_width <= 0:
        return mesh_owned

    pad_width = ((halo_width, halo_width),) + ((0, 0),) * (mesh_owned.ndim - 1)
    mesh_halo = jnp.pad(mesh_owned, pad_width)
    mesh_halo = jax.lax.dynamic_update_slice(
        mesh_halo,
        incoming_left,
        (0,) + (0,) * (mesh_owned.ndim - 1),
    )
    mesh_halo = jax.lax.dynamic_update_slice(
        mesh_halo,
        incoming_right,
        (halo_width + mesh_owned.shape[0],) + (0,) * (mesh_owned.ndim - 1),
    )
    return mesh_halo


def extend_owned_mesh_with_halo(mesh_owned, halo_width: int, left_perm, right_perm):
    """Attach static x-halo cells by copying neighbor owned edge cells."""
    incoming_left, incoming_right = exchange_owned_mesh_halo_edges(
        mesh_owned,
        halo_width,
        left_perm,
        right_perm,
    )
    return extend_owned_mesh_from_halo_edges(mesh_owned, incoming_left, incoming_right, halo_width)


def reduce_mesh_halo_to_owned(mesh_with_halo, halo_width: int, left_perm, right_perm):
    """Add halo contributions onto neighbor owned edge cells and drop the halo shell."""
    if halo_width <= 0:
        return mesh_with_halo

    owned = mesh_with_halo[halo_width:-halo_width]
    left_halo = mesh_with_halo[:halo_width]
    right_halo = mesh_with_halo[-halo_width:]

    incoming_left = jax.lax.ppermute(right_halo, axis_name=AXIS_NAME, perm=right_perm)
    incoming_right = jax.lax.ppermute(left_halo, axis_name=AXIS_NAME, perm=left_perm)

    owned = owned.at[:halo_width].add(incoming_left)
    owned = owned.at[-halo_width:].add(incoming_right)
    return owned
