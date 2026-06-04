from __future__ import annotations

import math
from functools import partial
from typing import Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .FFT_distributed import create_batched_transposed_real_ffts, create_ffts
from .gather import initialize_mGPU_gather
from .halo_moving import (
    initialize_mGPU_compute_halo_mask,
    initialize_mGPU_halo_move_pullback,
    initialize_mGPU_halo_movement_canonical,
    initialize_mGPU_reconstruct_pre_drift,
    initialize_mGPU_reconstruct_pre_drift_pullback,
)
from .scatter import initialize_mGPU_scatter
from .utils import build_ring_permutations, pytree_dataclass

if TYPE_CHECKING:
    from .configuration import Configuration


def _uninitialized_runtime_callable(*args, **kwargs):
    """Placeholder for runtime helpers before ``Configuration`` initializes them."""
    raise RuntimeError("MultiGPU runtime helper was used before initialization.")


@partial(pytree_dataclass, aux_fields=Ellipsis, frozen=True, eq=False)
class MultiGPUConfiguration:
    """Derived multi-GPU topology and compiled helper functions.

    `Configuration` keeps the physical simulation parameters and user-facing input
    knobs. This object stores the derived distributed runtime state that depends on
    those inputs: slab layout, halo bands, communication permutations, and the
    initialized sharded helper functions.
    """

    compute_mesh: Mesh = None
    num_devices: int = 0
    devices: tuple[jax.Device, ...] = ()
    devices_index: tuple[int, ...] = ()

    local_mesh_shape: tuple[int, ...] = ()
    local_mesh_with_halo_shape: tuple[int, ...] = ()

    mode: str | None = "mesh_halo"
    store_particle_halos: bool = False
    ptcl_halo_width: int = 0
    mesh_halo_width: int = 0
    owned_slice_start: jax.Array = None
    owned_slice_end: jax.Array = None
    slice_start: jax.Array = None
    slice_end: jax.Array = None
    halo_start: jax.Array = None
    halo_end: jax.Array = None
    offsets: jax.Array = None
    scatter_offsets: jax.Array = None
    mesh_halo_offsets: jax.Array = None

    max_ptcl_per_slice: int = 0
    max_share_ptcl: int = 0
    max_halo_share_ptcl: int = 0
    max_share_gather_ptcl: int = 0

    left_perm: tuple[tuple[int, int], ...] = ()
    right_perm: tuple[tuple[int, int], ...] = ()

    halo_moving: Callable = _uninitialized_runtime_callable
    reconstruct_pre_drift: Callable = _uninitialized_runtime_callable
    reconstruct_pre_drift_pullback: Callable | None = None
    halo_move_pullback: Callable = _uninitialized_runtime_callable
    compute_halo_mask: Callable = _uninitialized_runtime_callable
    rfftn: Callable = _uninitialized_runtime_callable
    irfftn: Callable = _uninitialized_runtime_callable
    rfftn_transposed: Callable = _uninitialized_runtime_callable
    irfftn_transposed: Callable = _uninitialized_runtime_callable
    irfftn_transposed_batched: Callable | None = None
    scatter: Callable = _uninitialized_runtime_callable
    gather: Callable = _uninitialized_runtime_callable


def build_multigpu_configuration(conf: "Configuration", runtime_seed: MultiGPUConfiguration | None = None) -> MultiGPUConfiguration | None:
    """Derive static multi-GPU topology from the user-facing configuration.

    This function computes slab ownership, halo extents, communication
    permutations, and static buffer capacities. It intentionally does not
    create jitted helper functions; that happens in ``initialize_multigpu_runtime``
    after the final ``Configuration`` object exists.
    """
    compute_mesh = runtime_seed.compute_mesh if runtime_seed is not None and runtime_seed.compute_mesh is not None else conf.compute_mesh
    if compute_mesh is None:
        return None
    if compute_mesh.size < 1:
        raise ValueError(
            f"mGPU used, but less than 1 device was set in the compute_mesh (actual: {compute_mesh.size})."
        )

    num_devices = compute_mesh.size
    devices = tuple(compute_mesh.devices)
    # Preserve physical device IDs for reporting/debugging, but index all slab
    # topology arrays by logical mesh position so reordered/filtered device lists
    # still produce a contiguous decomposition.
    devices_index = tuple(device.id for device in devices)
    mesh_positions = tuple(range(num_devices))

    mode = (
        runtime_seed.mode
        if runtime_seed is not None and runtime_seed.mode is not None
        else conf.multigpu_mode
    )
    if mode not in {"particle_halo", "mesh_halo"}:
        raise ValueError(f"Unsupported multigpu_mode={mode!r}. Expected 'particle_halo' or 'mesh_halo'.")

    local_mesh_shape = (conf.mesh_shape[0] // num_devices, conf.mesh_shape[1], conf.mesh_shape[2])
    global_nMesh = conf.mesh_shape[0]
    store_particle_halos = mode == "particle_halo"
    ptcl_halo_width = 0 if num_devices == 1 else max(
        1,
        int(round(conf.mesh_shape[0] / conf.ptcl_grid_shape[0])),
    )
    mesh_halo_width = 0 if num_devices == 1 else 1
    local_mesh_with_halo_shape = (
        local_mesh_shape[0] + 2 * mesh_halo_width,
        local_mesh_shape[1],
        local_mesh_shape[2],
    )

    slice_start = [(global_nMesh // num_devices * mesh_pos) % global_nMesh for mesh_pos in mesh_positions]
    slice_end = [(global_nMesh // num_devices * (mesh_pos + 1)) % global_nMesh for mesh_pos in mesh_positions]
    owned_slice_start = list(slice_start)
    owned_slice_end = list(slice_end)
    halo_start = [[(start - ptcl_halo_width) % global_nMesh, start] for start in slice_start]
    halo_end = [[(end - ptcl_halo_width) % global_nMesh, end] for end in slice_end]

    offsets = jnp.array(slice_start)
    scatter_offsets = jnp.array([[offset, 0.0, 0.0] for offset in offsets]) * conf.cell_size
    mesh_halo_offsets = jnp.array(
        [[offset - mesh_halo_width, 0.0, 0.0] for offset in offsets],
        dtype=conf.float_dtype,
    ) * conf.cell_size

    if conf.max_ptcl_per_slice is None:
        scaling = 1.3 + (num_devices.bit_length() - 3) * 0.1
        if num_devices == 1:
            scaling = 1.0
        max_ptcl_per_slice = math.floor(conf.ptcl_num // num_devices * scaling)
    else:
        max_ptcl_per_slice = conf.max_ptcl_per_slice
    if num_devices == 1:
        max_ptcl_per_slice = min(max_ptcl_per_slice, conf.ptcl_num)

    max_share_ptcl = min(conf.max_share_ptcl, max_ptcl_per_slice // 2)
    if conf.max_halo_share_ptcl is None:
        max_halo_share_ptcl = min(
            max_ptcl_per_slice,
            (max_ptcl_per_slice * ptcl_halo_width + local_mesh_shape[0] - 1) // local_mesh_shape[0],
        )
    else:
        max_halo_share_ptcl = min(conf.max_halo_share_ptcl, max_ptcl_per_slice)
    max_share_gather_ptcl = min(conf.max_share_gather_ptcl, max_ptcl_per_slice // 2)
    left_perm, right_perm = build_ring_permutations(num_devices)

    base_runtime = runtime_seed if runtime_seed is not None else MultiGPUConfiguration()
    return base_runtime.replace(
        compute_mesh=compute_mesh,
        num_devices=num_devices,
        devices=devices,
        devices_index=devices_index,
        local_mesh_shape=local_mesh_shape,
        local_mesh_with_halo_shape=local_mesh_with_halo_shape,
        mode=mode,
        store_particle_halos=store_particle_halos,
        ptcl_halo_width=ptcl_halo_width,
        mesh_halo_width=mesh_halo_width,
        owned_slice_start=jnp.array(owned_slice_start),
        owned_slice_end=jnp.array(owned_slice_end),
        slice_start=jnp.array(halo_start)[:, 0],
        slice_end=jnp.array(halo_end)[:, 1] if num_devices > 1 else jnp.array([global_nMesh]),
        halo_start=jnp.array(halo_start),
        halo_end=jnp.array(halo_end),
        offsets=offsets,
        scatter_offsets=scatter_offsets,
        mesh_halo_offsets=mesh_halo_offsets,
        max_ptcl_per_slice=max_ptcl_per_slice,
        max_share_ptcl=max_share_ptcl,
        max_halo_share_ptcl=max_halo_share_ptcl,
        max_share_gather_ptcl=max_share_gather_ptcl,
        left_perm=left_perm,
        right_perm=right_perm,
    )


def initialize_multigpu_runtime(conf: "Configuration", runtime: MultiGPUConfiguration) -> MultiGPUConfiguration:
    """Attach distributed FFT, scatter/gather, and halo-movement callables."""
    rfftn_jit, irfftn_jit, _, _, rfftn_transposed_jit, irfftn_transposed_jit = create_ffts(runtime.compute_mesh)
    _, irfftn_transposed_batched_jit = create_batched_transposed_real_ffts(runtime.compute_mesh)
    return runtime.replace(
        rfftn=rfftn_jit,
        irfftn=irfftn_jit,
        rfftn_transposed=rfftn_transposed_jit,
        irfftn_transposed=irfftn_transposed_jit,
        irfftn_transposed_batched=irfftn_transposed_batched_jit,
        halo_moving=initialize_mGPU_halo_movement_canonical(conf),
        reconstruct_pre_drift=initialize_mGPU_reconstruct_pre_drift(conf),
        reconstruct_pre_drift_pullback=initialize_mGPU_reconstruct_pre_drift_pullback(conf),
        halo_move_pullback=initialize_mGPU_halo_move_pullback(conf),
        compute_halo_mask=initialize_mGPU_compute_halo_mask(conf),
        scatter=initialize_mGPU_scatter(conf),
        gather=initialize_mGPU_gather(conf),
    )
