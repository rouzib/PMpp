"""Tiled Pallas CIC kernels used by PM++.

The kernels are deliberately private implementation details of :mod:`pmpp.cic`.
On the qualified float32 GPU/JAX combination they replace
the materialising ``N x 8`` JAX CIC expressions in both the forward and custom-VJP
paths.  A validity mask is carried into every particle kernel so static-capacity
padding never performs a mesh load, an atomic update, or a gradient write.

The reference JAX implementation remains available by setting the single
``pallas_cic`` configuration flag to ``False``.
"""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

try:  # Keep importing PM++ possible with older JAX installations.
    from jax.experimental import pallas as pl
except Exception:  # pragma: no cover - exercised only by old JAX versions.
    pl = None

try:  # JAX 0.10 moved GPU memory operations to the Triton submodule.
    from jax.experimental.pallas import triton as pl_triton
except Exception:  # pragma: no cover - exercised only by old JAX versions.
    pl_triton = None


def _load(ref, index, *, mask=None, other=None):
    """Load through the Pallas memory API exposed by this JAX version."""

    if hasattr(pl, "load"):
        return pl.load(ref, index, mask=mask, other=other)
    return pl_triton.load(ref.at[index], mask=mask, other=other)


def _store(ref, index, value, *, mask=None):
    """Store through the Pallas memory API exposed by this JAX version."""

    if hasattr(pl, "store"):
        return pl.store(ref, index, value, mask=mask)
    return pl_triton.store(ref.at[index], value, mask=mask)


def _atomic_add(ref, index, value, *, mask=None):
    """Atomic add through the Pallas memory API exposed by this JAX version."""

    if hasattr(pl, "atomic_add"):
        return pl.atomic_add(ref, index, value, mask=mask)
    return pl_triton.atomic_add(ref, index, value, mask=mask)


def pallas_available() -> bool:
    """Return whether the installed JAX exposes the required Triton Pallas API."""

    memory_ops_available = (pl is not None and all(
        hasattr(pl, name) for name in ("load", "store", "atomic_add")
    )) or (pl_triton is not None and all(hasattr(pl_triton, name) for name in ("load", "store", "atomic_add")))
    return (
        pl is not None and hasattr(pl, "pallas_call") and pl_triton is not None
        and hasattr(pl_triton, "CompilerParams") and memory_ops_available
    )


def _supported_pallas_jax_version() -> bool:
    """Return whether JAX meets the minimum version for these kernels."""

    try:
        major, minor, patch = (int(part) for part in jax.__version__.split("+")[0].split(".")[:3])
    except ValueError:  # pragma: no cover - defensive for nonstandard builds.
        return False
    return (major, minor, patch) >= (0, 9, 1)


def pallas_cic_supported(dtype) -> bool:
    """Return whether optimized CIC forward and backward paths are usable here."""

    return (
        pallas_available() and _supported_pallas_jax_version()
        and any(device.platform == "gpu" for device in jax.devices()) and jnp.dtype(dtype) == jnp.dtype(jnp.float32)
    )


def _require_pallas(dtype):
    if not pallas_available():
        raise RuntimeError(
            "Pallas CIC was requested, but this JAX installation does not expose "
            "jax.experimental.pallas."
        )
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Pallas CIC currently targets the GPU backend; no GPU JAX device is visible.")
    dtype = jnp.dtype(dtype)
    if dtype != jnp.dtype(jnp.float32):
        raise RuntimeError(
            f"Pallas CIC currently supports float32 meshes only (got {dtype}). "
            "Use float_dtype=jnp.float32 or disable the Pallas flag."
        )


def _shape_tuple(shape: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in shape)


def _offset_array(offset, dim: int, dtype):
    """Normalize scalar/vector offsets without changing their physical units."""

    return jnp.broadcast_to(jnp.asarray(offset, dtype=dtype), (dim, ))


def _choose_block_size(particle_count: int) -> int:
    """Choose a static tile size without over-unrolling tiny correctness tests."""

    if particle_count <= 1:
        return 1
    block_size = 1
    while block_size < particle_count and block_size < 128:
        block_size *= 2
    return min(block_size, 128)


def _pad_particles(array, padded_count: int, *, value=0):
    """Pad a leading particle axis; padding is bounded by one tile."""

    if array.ndim == 0 or array.shape[0] == padded_count:
        return array
    pad_width = [(0, padded_count - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
    return jnp.pad(array, pad_width, constant_values=value)


def _valid_particles(valid_mask, particle_count: int, padded_count: int):
    if valid_mask is None:
        valid_mask = jnp.ones((particle_count, ), dtype=jnp.bool_)
    else:
        valid_mask = jnp.asarray(valid_mask, dtype=jnp.bool_)
        if valid_mask.ndim != 1 or valid_mask.shape[0] != particle_count:
            raise ValueError(
                "Pallas CIC valid_mask must have shape (particle_count,), got "
                f"{valid_mask.shape} for particle_count={particle_count}."
            )
    return _pad_particles(valid_mask, padded_count, value=False)


def _particle_extent(particle_count: int, block_size: int) -> int:
    """Round only a final incomplete Pallas particle tile for safe masking."""
    return ((particle_count + block_size - 1) // block_size) * block_size


def _particle_block_spec(block_size: int, trailing_shape: tuple[int, ...]):
    """BlockSpec for a leading particle tile (indices are in block units)."""

    return pl.BlockSpec((block_size, ) + trailing_shape, lambda block: (block, ) + (0, ) * len(trailing_shape), )


def _make_cic_coordinate_helper(*, spatial_shape, global_shape, cell_size_is_explicit, cell_dtype, block_size: int):
    """Create the statically unrolled CIC coordinate/weight helper."""

    spatial_shape = _shape_tuple(spatial_shape)
    global_shape = _shape_tuple(global_shape)
    dim = len(spatial_shape)
    if dim != 3:
        raise NotImplementedError("Pallas CIC currently supports three spatial dimensions")

    neighbour_bits = tuple(tuple((bits >> axis) & 1 for axis in range(dim)) for bits in range(2**dim))
    a1 = float(cell_dtype)

    def _axis_weight_grad(delta):
        """CIC weight and derivative in normalized cell coordinates."""

        weight = 1 - jnp.abs(delta)
        # enmesh uses sign(-d2); spelling it with comparisons is more reliable
        # on the installed Mosaic GPU backend than jnp.sign in a Pallas kernel.
        sign = jnp.where(delta < 0, 1.0, jnp.where(delta > 0, -1.0, 0.0))
        return weight, sign

    def _particle_coordinates(pmid_ref, disp_ref, offset_ref, cell_ref, particle, lane_valid):
        work_dtype = jnp.float32
        pmid = tuple(
            _load(pmid_ref, (particle, axis), mask=lane_valid, other=0).astype(jnp.int32) for axis in range(dim)
        )
        disp = tuple(
            _load(disp_ref, (particle, axis), mask=lane_valid, other=0).astype(work_dtype) for axis in range(dim)
        )
        offset = tuple(_load(offset_ref, (axis, )).astype(work_dtype) for axis in range(dim))

        indices = []
        fractions = []
        fraction_grads = []

        if cell_size_is_explicit:
            a2 = _load(cell_ref, ()).astype(work_dtype)
            for bits in neighbour_bits:
                idx_axes = []
                axis_weights = []
                axis_grads = []
                for axis, bit in enumerate(bits):
                    length = jnp.asarray(global_shape[axis], dtype=work_dtype)
                    position = (pmid[axis].astype(work_dtype) * a1 + disp[axis] - offset[axis])
                    position = jnp.mod(position, length * a1)
                    neighbour_position = jnp.mod(position + bit * a2, length * a1)
                    idx = jnp.floor(neighbour_position / a2).astype(jnp.int32)
                    delta = position - idx.astype(work_dtype) * a2
                    period = length * a1
                    # Equivalent to enmesh's periodic rint correction, expressed
                    # without round/rint because those are not lowered reliably by
                    # the installed Pallas GPU backend.
                    delta = jnp.where(delta > 0.5 * period, delta - period, delta)
                    delta = jnp.where(delta < -0.5 * period, delta + period, delta)
                    weight, grad = _axis_weight_grad(delta / a2)
                    idx_axes.append(idx)
                    axis_weights.append(weight)
                    axis_grads.append(grad)
                indices.append(tuple(idx_axes))
                fractions.append(axis_weights[0] * axis_weights[1] * axis_weights[2])
                fraction_grads.append(
                    tuple(
                        axis_grads[axis] * axis_weights[(axis + 1) % dim] * axis_weights[(axis + 2) % dim]
                        for axis in range(dim)
                    )
                )
            return tuple(indices), tuple(fractions), tuple(fraction_grads), a2

        # Common PM++ path (a2=None in enmesh): both grids have the particle
        # cell size, and offset is the physical origin of the local mesh.
        base_indices = []
        base_fractions = []
        for axis in range(dim):
            cell = jnp.asarray(a1, dtype=work_dtype)
            quotient = jnp.floor(offset[axis] / cell).astype(jnp.int32)
            remainder = offset[axis] - quotient.astype(work_dtype) * cell
            reduced = (disp[axis] - remainder) / cell
            lower = jnp.floor(reduced).astype(jnp.int32)
            base_indices.append(pmid[axis] - quotient + lower)
            base_fractions.append(reduced - lower.astype(work_dtype))

        for bits in neighbour_bits:
            idx_axes = []
            axis_weights = []
            axis_grads = []
            for axis, bit in enumerate(bits):
                idx = jnp.mod(base_indices[axis] + bit, global_shape[axis])
                weight, grad = _axis_weight_grad(base_fractions[axis] - bit)
                idx_axes.append(idx)
                axis_weights.append(weight)
                axis_grads.append(grad)
            indices.append(tuple(idx_axes))
            fractions.append(axis_weights[0] * axis_weights[1] * axis_weights[2])
            fraction_grads.append(
                tuple(
                    axis_grads[axis] * axis_weights[(axis + 1) % dim] * axis_weights[(axis + 2) % dim]
                    for axis in range(dim)
                )
            )
        return tuple(indices), tuple(fractions), tuple(fraction_grads), jnp.asarray(a1, dtype=work_dtype)

    return spatial_shape, global_shape, block_size, _particle_coordinates


def _bounds_mask(index, spatial_shape):
    valid = True
    for axis, idx in enumerate(index):
        valid = valid & (idx >= 0) & (idx < spatial_shape[axis])
    return valid


def _make_cic_forward_kernel(
    *, spatial_shape, global_shape, channel_shape, cell_size_is_explicit, cell_dtype, scatter: bool, block_size: int,
    particle_count: int
):
    spatial_shape, global_shape, _, coordinates = _make_cic_coordinate_helper(
        spatial_shape=spatial_shape, global_shape=global_shape, cell_size_is_explicit=cell_size_is_explicit,
        cell_dtype=cell_dtype, block_size=block_size,
    )

    if scatter:

        def kernel(pmid_ref, disp_ref, valid_ref, val_ref, offset_ref, cell_ref, mesh_in_ref, mesh_ref):
            del mesh_in_ref
            block = pl.program_id(0)
            lanes = jnp.arange(block_size, dtype=jnp.int32)
            lane_valid = (block * block_size + lanes) < particle_count
            particle_valid = lane_valid & _load(valid_ref, (lanes, ), mask=lane_valid, other=False)
            indices, fractions, _, _ = coordinates(pmid_ref, disp_ref, offset_ref, cell_ref, lanes, lane_valid)
            scalar_val = val_ref.shape == ()
            for index, fraction in zip(indices, fractions):
                valid = particle_valid & _bounds_mask(index, spatial_shape)
                if channel_shape:
                    for channel in np.ndindex(channel_shape):
                        value = (
                            _load(val_ref, channel) if scalar_val else _load(
                                val_ref, (lanes, ) + channel, mask=particle_valid, other=0,
                            )
                        )
                        _atomic_add(mesh_ref, index + channel, value * fraction, mask=valid)
                else:
                    value = (
                        _load(val_ref, ()) if scalar_val else _load(val_ref, (lanes, ), mask=particle_valid, other=0)
                    )
                    _atomic_add(mesh_ref, index, value * fraction, mask=valid)

        return kernel

    def kernel(pmid_ref, disp_ref, valid_ref, offset_ref, cell_ref, mesh_ref, out_ref):
        block = pl.program_id(0)
        lanes = jnp.arange(block_size, dtype=jnp.int32)
        lane_valid = (block * block_size + lanes) < particle_count
        particle_valid = lane_valid & _load(valid_ref, (lanes, ), mask=lane_valid, other=False)
        indices, fractions, _, _ = coordinates(pmid_ref, disp_ref, offset_ref, cell_ref, lanes, lane_valid)
        for channel in np.ndindex(channel_shape) if channel_shape else [()]:
            result = jnp.zeros((block_size, ), dtype=jnp.float32)
            for index, fraction in zip(indices, fractions):
                valid = _bounds_mask(index, spatial_shape)
                result = result + _load(mesh_ref, index + channel, mask=valid & particle_valid, other=0) * fraction
            _store(out_ref, (lanes, ) + channel, result, mask=lane_valid)

    return kernel


def _make_gather_bwd_kernel(
    *, spatial_shape, global_shape, channel_shape, cell_size_is_explicit, cell_dtype, block_size: int,
    particle_count: int
):
    spatial_shape, _, _, coordinates = _make_cic_coordinate_helper(
        spatial_shape=spatial_shape, global_shape=global_shape, cell_size_is_explicit=cell_size_is_explicit,
        cell_dtype=cell_dtype, block_size=block_size,
    )

    def kernel(
        pmid_ref, disp_ref, valid_ref, mesh_ref, val_cot_ref, offset_ref, cell_ref, mesh_cot_in_ref, disp_cot_ref,
        mesh_cot_ref
    ):
        del mesh_cot_in_ref
        block = pl.program_id(0)
        lanes = jnp.arange(block_size, dtype=jnp.int32)
        lane_valid = (block * block_size + lanes) < particle_count
        particle_valid = lane_valid & _load(valid_ref, (lanes, ), mask=lane_valid, other=False)
        indices, fractions, fraction_grads, cell_scale = coordinates(
            pmid_ref, disp_ref, offset_ref, cell_ref, lanes, lane_valid
        )
        disp_result = [jnp.zeros((block_size, ), dtype=jnp.float32) for _ in range(3)]
        for index, fraction, fraction_grad in zip(indices, fractions, fraction_grads):
            valid = particle_valid & _bounds_mask(index, spatial_shape)
            for channel in np.ndindex(channel_shape) if channel_shape else [()]:
                mesh_index = index + channel
                mesh_value = _load(mesh_ref, mesh_index, mask=valid, other=0)
                val_cot = _load(val_cot_ref, (lanes, ) + channel, mask=particle_valid, other=0)
                _atomic_add(mesh_cot_ref, mesh_index, val_cot * fraction, mask=valid)
                for axis in range(3):
                    disp_result[axis] = disp_result[axis] + (val_cot * mesh_value * fraction_grad[axis])
        for axis in range(3):
            _store(disp_cot_ref, (lanes, axis), disp_result[axis] / cell_scale, mask=lane_valid)

    return kernel


def _make_scatter_bwd_kernel(
    *, spatial_shape, global_shape, channel_shape, cell_size_is_explicit, cell_dtype, block_size: int,
    particle_count: int, scalar_val: bool
):
    spatial_shape, _, _, coordinates = _make_cic_coordinate_helper(
        spatial_shape=spatial_shape, global_shape=global_shape, cell_size_is_explicit=cell_size_is_explicit,
        cell_dtype=cell_dtype, block_size=block_size,
    )

    def kernel(
        pmid_ref, disp_ref, valid_ref, val_ref, offset_ref, cell_ref, mesh_cot_ref, mesh_cot_in_ref, disp_cot_ref,
        val_cot_ref
    ):
        del mesh_cot_in_ref
        block = pl.program_id(0)
        lanes = jnp.arange(block_size, dtype=jnp.int32)
        lane_valid = (block * block_size + lanes) < particle_count
        particle_valid = lane_valid & _load(valid_ref, (lanes, ), mask=lane_valid, other=False)
        indices, fractions, fraction_grads, cell_scale = coordinates(
            pmid_ref, disp_ref, offset_ref, cell_ref, lanes, lane_valid
        )
        disp_result = [jnp.zeros((block_size, ), dtype=jnp.float32) for _ in range(3)]
        channel_indices = list(np.ndindex(channel_shape)) if channel_shape else [()]
        val_result = ([jnp.zeros((block_size, ), dtype=jnp.float32)
                       for _ in channel_indices] if not scalar_val else None)
        for index, fraction, fraction_grad in zip(indices, fractions, fraction_grads):
            valid = particle_valid & _bounds_mask(index, spatial_shape)
            channel_sum = jnp.zeros((block_size, ), dtype=jnp.float32)
            for channel_index, channel in enumerate(channel_indices):
                mesh_value = _load(mesh_cot_ref, index + channel, mask=valid, other=0)
                particle_value = (
                    _load(val_ref,
                          ()) if scalar_val else _load(val_ref, (lanes, ) + channel, mask=particle_valid, other=0)
                )
                weighted = mesh_value * fraction
                if scalar_val:
                    # ``weighted`` is already zero for invalid/padded lanes;
                    # reducing the tile gives one scalar atomic update, which
                    # is the cotangent of a scalar particle value.
                    _atomic_add(val_cot_ref, (), jnp.sum(weighted))
                else:
                    val_result[channel_index] = val_result[channel_index] + weighted
                channel_sum = channel_sum + mesh_value * particle_value
            for axis in range(3):
                disp_result[axis] = disp_result[axis] + channel_sum * fraction_grad[axis]
        for axis in range(3):
            _store(disp_cot_ref, (lanes, axis), disp_result[axis] / cell_scale, mask=lane_valid)
        if not scalar_val:
            for channel_index, channel in enumerate(channel_indices):
                _store(val_cot_ref, (lanes, ) + channel, val_result[channel_index], mask=lane_valid)

    return kernel


def pallas_gather(pmid, disp, mesh, *, offset, particle_cell_size, cell_size=None, global_shape, valid_mask=None):
    """Gather CIC values with a tiled Pallas kernel."""

    _require_pallas(mesh.dtype)
    if pmid.ndim != 2 or pmid.shape[1] != 3:
        raise NotImplementedError("Pallas CIC currently supports (N, 3) particles")
    mesh = jnp.asarray(mesh)
    particle_count = int(pmid.shape[0])
    block_size = _choose_block_size(particle_count)
    padded_count = _particle_extent(particle_count, block_size)
    pmid = _pad_particles(pmid, padded_count)
    disp = _pad_particles(disp, padded_count)
    valid_mask = _valid_particles(valid_mask, particle_count, padded_count)
    offset = _offset_array(offset, 3, mesh.dtype)
    cell_arg = (jnp.asarray(0, dtype=mesh.dtype) if cell_size is None else jnp.asarray(cell_size, dtype=mesh.dtype))
    channel_shape = mesh.shape[3:]
    kernel = _make_cic_forward_kernel(
        spatial_shape=mesh.shape[:3], global_shape=global_shape, channel_shape=channel_shape,
        cell_size_is_explicit=cell_size is not None, cell_dtype=particle_cell_size, scatter=False,
        block_size=block_size, particle_count=particle_count,
    )
    out_shape = jax.ShapeDtypeStruct((padded_count, ) + channel_shape, mesh.dtype)
    call = pl.pallas_call(
        kernel, out_shape=out_shape, grid=(padded_count // block_size, ), in_specs=(
            _particle_block_spec(block_size, (3, )), _particle_block_spec(block_size, (3, )),
            _particle_block_spec(block_size, ()), pl.no_block_spec, pl.no_block_spec, pl.no_block_spec,
        ), out_specs=_particle_block_spec(block_size, channel_shape), compiler_params=pl_triton.CompilerParams(),
        name="pmpp_cic_gather_tiled",
    )
    return call(pmid, disp, valid_mask, offset, cell_arg, mesh)[:particle_count]


def pallas_scatter(pmid, disp, val, mesh, *, offset, particle_cell_size, cell_size=None, global_shape, valid_mask=None):
    """Scatter values into a mesh with a Pallas atomic-add kernel."""

    _require_pallas(mesh.dtype)
    if pmid.ndim != 2 or pmid.shape[1] != 3:
        raise NotImplementedError("Pallas CIC currently supports (N, 3) particles")
    mesh = jnp.asarray(mesh)
    val = jnp.asarray(val, dtype=mesh.dtype)
    particle_count = int(pmid.shape[0])
    block_size = _choose_block_size(particle_count)
    padded_count = _particle_extent(particle_count, block_size)
    pmid = _pad_particles(pmid, padded_count)
    disp = _pad_particles(disp, padded_count)
    valid_mask = _valid_particles(valid_mask, particle_count, padded_count)
    if val.ndim != 0:
        val = _pad_particles(val, padded_count)
    offset = _offset_array(offset, 3, mesh.dtype)
    cell_arg = (jnp.asarray(0, dtype=mesh.dtype) if cell_size is None else jnp.asarray(cell_size, dtype=mesh.dtype))
    channel_shape = val.shape[1:] if val.ndim else ()
    kernel = _make_cic_forward_kernel(
        spatial_shape=mesh.shape[:3], global_shape=global_shape, channel_shape=channel_shape,
        cell_size_is_explicit=cell_size is not None, cell_dtype=particle_cell_size, scatter=True, block_size=block_size,
        particle_count=particle_count,
    )
    out_shape = jax.ShapeDtypeStruct(mesh.shape, mesh.dtype)
    call = pl.pallas_call(
        kernel, out_shape=out_shape, grid=(padded_count // block_size, ), in_specs=(
            _particle_block_spec(block_size, (3, )), _particle_block_spec(block_size,
                                                                          (3, )), _particle_block_spec(block_size, ()),
            pl.no_block_spec if val.ndim == 0 else _particle_block_spec(block_size, val.shape[1:]), pl.no_block_spec,
            pl.no_block_spec, pl.no_block_spec,
        ), out_specs=pl.no_block_spec, input_output_aliases={6: 0}, compiler_params=pl_triton.CompilerParams(),
        name="pmpp_cic_scatter_tiled",
    )
    return call(pmid, disp, valid_mask, val, offset, cell_arg, mesh)


def pallas_gather_bwd(
    pmid, disp, mesh, val_cot, *, offset, particle_cell_size, cell_size=None, global_shape, valid_mask=None
):
    """Hand-written CIC gather adjoint: particle gradients plus mesh atomics."""

    _require_pallas(mesh.dtype)
    if pmid.ndim != 2 or pmid.shape[1] != 3:
        raise NotImplementedError("Pallas CIC currently supports (N, 3) particles")
    mesh = jnp.asarray(mesh)
    val_cot = jnp.asarray(val_cot, dtype=mesh.dtype)
    particle_count = int(pmid.shape[0])
    block_size = _choose_block_size(particle_count)
    padded_count = _particle_extent(particle_count, block_size)
    pmid = _pad_particles(pmid, padded_count)
    disp = _pad_particles(disp, padded_count)
    valid_mask = _valid_particles(valid_mask, particle_count, padded_count)
    val_cot = _pad_particles(val_cot, padded_count)
    offset = _offset_array(offset, 3, mesh.dtype)
    cell_arg = (jnp.asarray(0, dtype=mesh.dtype) if cell_size is None else jnp.asarray(cell_size, dtype=mesh.dtype))
    channel_shape = mesh.shape[3:]
    kernel = _make_gather_bwd_kernel(
        spatial_shape=mesh.shape[:3], global_shape=global_shape, channel_shape=channel_shape,
        cell_size_is_explicit=cell_size is not None, cell_dtype=particle_cell_size, block_size=block_size,
        particle_count=particle_count,
    )
    disp_cot_shape = jax.ShapeDtypeStruct((padded_count, 3), mesh.dtype)
    mesh_cot = jnp.zeros_like(mesh)
    mesh_cot_shape = jax.ShapeDtypeStruct(mesh.shape, mesh.dtype)
    call = pl.pallas_call(
        kernel, out_shape=(disp_cot_shape, mesh_cot_shape), grid=(padded_count // block_size, ), in_specs=(
            _particle_block_spec(block_size, (3, )), _particle_block_spec(block_size,
                                                                          (3, )), _particle_block_spec(block_size, ()),
            pl.no_block_spec, _particle_block_spec(block_size,
                                                   channel_shape), pl.no_block_spec, pl.no_block_spec, pl.no_block_spec,
        ), out_specs=(_particle_block_spec(block_size, (3, )), pl.no_block_spec), input_output_aliases={7: 1},
        compiler_params=pl_triton.CompilerParams(), name="pmpp_cic_gather_bwd_tiled",
    )
    disp_cot, mesh_cot = call(pmid, disp, valid_mask, mesh, val_cot, offset, cell_arg, mesh_cot)
    return disp_cot[:particle_count], mesh_cot


def pallas_scatter_bwd(
    pmid, disp, val, mesh_cot, *, offset, particle_cell_size, cell_size=None, global_shape, valid_mask=None
):
    """Hand-written CIC scatter adjoint: particle/value gradients."""

    _require_pallas(mesh_cot.dtype)
    if pmid.ndim != 2 or pmid.shape[1] != 3:
        raise NotImplementedError("Pallas CIC currently supports (N, 3) particles")
    mesh_cot = jnp.asarray(mesh_cot)
    val = jnp.asarray(val, dtype=mesh_cot.dtype)
    particle_count = int(pmid.shape[0])
    block_size = _choose_block_size(particle_count)
    padded_count = _particle_extent(particle_count, block_size)
    pmid = _pad_particles(pmid, padded_count)
    disp = _pad_particles(disp, padded_count)
    valid_mask = _valid_particles(valid_mask, particle_count, padded_count)
    scalar_val = val.ndim == 0
    if not scalar_val:
        val = _pad_particles(val, padded_count)
    offset = _offset_array(offset, 3, mesh_cot.dtype)
    cell_arg = (
        jnp.asarray(0, dtype=mesh_cot.dtype) if cell_size is None else jnp.asarray(cell_size, dtype=mesh_cot.dtype)
    )
    channel_shape = val.shape[1:] if not scalar_val else ()
    kernel = _make_scatter_bwd_kernel(
        spatial_shape=mesh_cot.shape[:3], global_shape=global_shape, channel_shape=channel_shape,
        cell_size_is_explicit=cell_size is not None, cell_dtype=particle_cell_size, block_size=block_size,
        particle_count=particle_count, scalar_val=scalar_val,
    )
    disp_cot_shape = jax.ShapeDtypeStruct((padded_count, 3), mesh_cot.dtype)
    val_cot_shape = (
        jax.ShapeDtypeStruct((), mesh_cot.dtype) if scalar_val else jax.ShapeDtypeStruct((padded_count, ) +
                                                                                         channel_shape, mesh_cot.dtype)
    )
    val_cot = jnp.zeros_like(val) if not scalar_val else jnp.zeros((), dtype=mesh_cot.dtype)
    call = pl.pallas_call(
        kernel, out_shape=(disp_cot_shape, val_cot_shape), grid=(padded_count // block_size, ), in_specs=(
            _particle_block_spec(block_size, (3, )), _particle_block_spec(block_size,
                                                                          (3, )), _particle_block_spec(block_size, ()),
            pl.no_block_spec if scalar_val else _particle_block_spec(block_size, channel_shape), pl.no_block_spec,
            pl.no_block_spec, pl.no_block_spec, pl.no_block_spec,
        ), out_specs=(
            _particle_block_spec(block_size, (3, )),
            pl.no_block_spec if scalar_val else _particle_block_spec(block_size, channel_shape)
        ), input_output_aliases={7: 1}, compiler_params=pl_triton.CompilerParams(), name="pmpp_cic_scatter_bwd_tiled",
    )
    disp_cot, val_cot = call(pmid, disp, valid_mask, val, offset, cell_arg, mesh_cot, val_cot)
    if scalar_val:
        return disp_cot[:particle_count], val_cot
    return disp_cot[:particle_count], val_cot[:particle_count]
