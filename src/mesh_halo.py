from __future__ import annotations

import jax
import jax.numpy as jnp

from .utils import AXIS_NAME


def zero_pad_owned_mesh_halo(mesh_owned, halo_width: int):
    """Embed an owned mesh shard in a zero halo shell along the decomposed x-axis."""
    if halo_width <= 0:
        return mesh_owned
    pad_width = ((halo_width, halo_width),) + ((0, 0),) * (mesh_owned.ndim - 1)
    return jnp.pad(mesh_owned, pad_width)


def extend_owned_mesh_with_halo(mesh_owned, halo_width: int, left_perm, right_perm):
    """Attach static x-halo cells by copying neighbor owned edge cells."""
    if halo_width <= 0:
        return mesh_owned

    left_owned = mesh_owned[:halo_width]
    right_owned = mesh_owned[-halo_width:]
    incoming_left = jax.lax.ppermute(right_owned, axis_name=AXIS_NAME, perm=right_perm)
    incoming_right = jax.lax.ppermute(left_owned, axis_name=AXIS_NAME, perm=left_perm)
    return jnp.concatenate((incoming_left, mesh_owned, incoming_right), axis=0)


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
