"""Optional CUDA FFI implementation of local particle routing.

The core :mod:`pmpp` wheel deliberately does not contain a CUDA extension.  This
module is therefore written so importing PM++ never loads ``libcuda`` or
``nvcc``-built code.  When a separately built shared library is present, the
two shard-local handlers exposed here replace only the local pack and merge
parts of ``mesh_halo`` routing.  Neighbor communication remains ordinary JAX
``lax.ppermute`` operations.

The extension ABI is intentionally small:

* ``pmpp_route_pack`` classifies a local authoritative buffer and emits a
  fixed-capacity array of eight ``uint32`` words per record.  The words are a
  32-byte opaque record: raveled pmid, validity, three displacement values, and
  three velocity values (the floating-point values are bit-copied).
* ``pmpp_route_merge`` performs a stable merge against a virtual stay stream
  and emits canonical ``pmid``, displacement, velocity, and validity arrays.
* ``pmpp_route_merge_aux`` has the same operation but additionally emits source
  tags and source indices for callers that need a local transpose plan.

No FFI call is differentiable by itself.  PM++'s N-body custom adjoint already
reconstructs the route plan in JAX, so the CUDA forward path does not change
that public derivative boundary.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


_TARGETS = (
    "pmpp_route_pack",
    "pmpp_route_merge",
    "pmpp_route_merge_aux",
    "pmpp_route_transpose_split",
    "pmpp_route_transpose_scatter",
)
_LIBRARY: ctypes.CDLL | None = None
_REGISTERED = False


def _truthy_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _qualified_jax() -> bool:
    """Return whether the typed FFI ABI is qualified for this prototype."""
    try:
        major, minor, patch = (int(part) for part in jax.__version__.split(".")[:3])
    except (AttributeError, TypeError, ValueError):
        return False
    # The CUDA prototype is compiled and tested against the 0.6 typed FFI
    # ABI.  A later minor line must qualify itself explicitly before reuse.
    return (major, minor, patch) >= (0, 6, 0) and major == 0 and minor == 6


def _candidate_library_paths() -> tuple[Path, ...]:
    explicit = os.environ.get("PMPP_CUDA_ROUTING_LIBRARY")
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))
    package_dir = Path(__file__).resolve().parent
    paths.extend(
        (
            package_dir / "_cuda" / "libpmpp_cuda_routing.so",
            package_dir / "_cuda" / "pmpp_cuda_routing.so",
            package_dir.parent.parent / "cuda" / "build" / "libpmpp_cuda_routing.so",
            package_dir.parent.parent / "cuda" / "build" / "pmpp_cuda_routing.so",
        )
    )
    # Preserve order while avoiding duplicate filesystem probes.
    return tuple(dict.fromkeys(paths))


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    if not _truthy_env("PMPP_CUDA_ROUTING", True):
        return None
    for candidate in _candidate_library_paths():
        if not candidate.is_file():
            continue
        try:
            mode = getattr(ctypes, "RTLD_GLOBAL", 0)
            _LIBRARY = ctypes.CDLL(str(candidate), mode=mode)
            return _LIBRARY
        except OSError:
            # A stale or wrong-platform optional library must behave exactly
            # like an absent library: the canonical JAX route remains usable.
            continue
    return None


def _register_targets() -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    library = _load_library()
    if library is None or not hasattr(jax, "ffi"):
        return False
    try:
        for target in _TARGETS:
            jax.ffi.register_ffi_target(
                target,
                jax.ffi.pycapsule(getattr(library, target)),
                platform="CUDA",
            )
    except (AttributeError, RuntimeError, TypeError, ValueError, OSError):
        _REGISTERED = False
        return False
    _REGISTERED = True
    return True


def extension_status() -> dict[str, Any]:
    """Return non-throwing diagnostics for the optional CUDA extension."""
    library = _load_library()
    return {
        "qualified_jax": _qualified_jax(),
        "jax_version": getattr(jax, "__version__", None),
        "backend": str(jax.default_backend()),
        "library": None if library is None else getattr(library, "_name", None),
        "registered": bool(_REGISTERED),
        "candidates": tuple(str(path) for path in _candidate_library_paths()),
    }


def supported_configuration(
    conf: Any, *, num_devices: int | None = None, mode: str | None = None
) -> bool:
    """Check the static conditions required by the CUDA routing ABI."""
    if not _qualified_jax() or str(jax.default_backend()).lower() not in {"gpu", "cuda"}:
        return False
    if jnp.dtype(conf.float_dtype) != jnp.float32:
        return False
    if jnp.dtype(conf.pmid_dtype) not in {jnp.dtype(jnp.int16), jnp.dtype(jnp.int32)}:
        return False
    resolved_num_devices = (
        int(num_devices)
        if num_devices is not None
        else int(getattr(conf, "num_devices", 0) or 0)
    )
    if resolved_num_devices < 2:
        return False
    if (mode if mode is not None else getattr(conf, "multigpu_mode", None)) != "mesh_halo":
        return False
    try:
        mesh_size = int(np.prod(tuple(int(value) for value in conf.mesh_shape)))
    except (AttributeError, TypeError, ValueError):
        return False
    return mesh_size <= np.iinfo(np.uint32).max and _register_targets()


def enabled_for_configuration(conf: Any) -> bool:
    """Return whether CUDA routing should be selected for a configuration."""
    if getattr(conf, "cuda_routing", False) is not True:
        return False
    return supported_configuration(conf)


def _shape_dtype(shape: tuple[int, ...], dtype: Any) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, jnp.dtype(dtype))


def route_pack(
    pmid: jax.Array,
    disp: jax.Array,
    vel: jax.Array,
    valid: jax.Array,
    x_mod: jax.Array,
    *,
    global_nmesh: int,
    mesh_shape: tuple[int, int, int],
    owned_start: int,
    owned_end: int,
    slice_width: int,
    direction: int,
    num_devices: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Call the shard-local CUDA route-pack handler."""
    n = int(pmid.shape[0])
    outputs = (
        _shape_dtype((capacity, 8), jnp.uint32),
        _shape_dtype((), jnp.int32),
        _shape_dtype((n,), jnp.uint8),
    )
    call = jax.ffi.ffi_call("pmpp_route_pack", outputs)
    return call(
        pmid.astype(jnp.int32),
        disp.astype(jnp.float32),
        vel.astype(jnp.float32),
        valid.astype(jnp.uint8),
        x_mod.astype(jnp.float32),
        jnp.asarray(owned_start, dtype=jnp.int32),
        jnp.asarray(owned_end, dtype=jnp.int32),
        jnp.asarray(slice_width, dtype=jnp.int32),
        global_nmesh=np.int32(global_nmesh),
        mesh_x=np.int32(mesh_shape[0]),
        mesh_y=np.int32(mesh_shape[1]),
        mesh_z=np.int32(mesh_shape[2]),
        direction=np.int32(direction),
        num_devices=np.int32(num_devices),
        capacity=np.int32(capacity),
    )


def route_merge(
    pmid: jax.Array,
    disp: jax.Array,
    vel: jax.Array,
    stay_mask: jax.Array,
    incoming_records: jax.Array,
    incoming_count: jax.Array,
    *,
    mesh_shape: tuple[int, int, int],
    capacity: int,
    auxiliary: bool = False,
) -> tuple[jax.Array, ...]:
    """Call the stable CUDA route-merge handler.

    ``stay_mask`` is deliberately passed instead of a compact stay buffer.  The
    CUDA implementation scans that mask and binary-searches the virtual stream
    when producing each output slot, so outgoing holes never need a second
    full-capacity payload allocation.
    """
    outputs: tuple[jax.ShapeDtypeStruct, ...] = (
        _shape_dtype((capacity, 3), jnp.int32),
        _shape_dtype((capacity, 3), jnp.float32),
        _shape_dtype((capacity, 3), jnp.float32),
        _shape_dtype((capacity,), jnp.uint8),
    )
    if auxiliary:
        outputs += (
            _shape_dtype((capacity,), jnp.uint8),
            _shape_dtype((capacity,), jnp.int32),
        )
        target = "pmpp_route_merge_aux"
    else:
        target = "pmpp_route_merge"
    call = jax.ffi.ffi_call(target, outputs)
    return call(
        pmid.astype(jnp.int32),
        disp.astype(jnp.float32),
        vel.astype(jnp.float32),
        stay_mask.astype(jnp.uint8),
        incoming_records.astype(jnp.uint32),
        incoming_count.astype(jnp.int32),
        mesh_x=np.int32(mesh_shape[0]),
        mesh_y=np.int32(mesh_shape[1]),
        mesh_z=np.int32(mesh_shape[2]),
        capacity=np.int32(capacity),
    )


def route_transpose_split(
    merged_cot: jax.Array,
    source_tag: jax.Array,
    source_idx: jax.Array,
    *,
    auth_size: int,
    share_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Transpose a merge locally into stay/left/right exchange streams."""
    payload_shape = tuple(int(value) for value in merged_cot.shape[1:])
    outputs = (
        _shape_dtype((auth_size,) + payload_shape, jnp.float32),
        _shape_dtype((share_capacity,) + payload_shape, jnp.float32),
        _shape_dtype((share_capacity,) + payload_shape, jnp.float32),
    )
    return jax.ffi.ffi_call("pmpp_route_transpose_split", outputs)(
        merged_cot.astype(jnp.float32),
        source_tag.astype(jnp.uint8),
        source_idx.astype(jnp.int32),
        auth_size=np.int32(auth_size),
        share_capacity=np.int32(share_capacity),
    )


def route_transpose_scatter(
    stay_cot: jax.Array,
    send_left_cot: jax.Array,
    send_right_cot: jax.Array,
    stay_pos: jax.Array,
    stay_valid: jax.Array,
    send_left_pos: jax.Array,
    send_left_valid: jax.Array,
    send_right_pos: jax.Array,
    send_right_valid: jax.Array,
    *,
    auth_size: int,
    share_capacity: int,
) -> jax.Array:
    """Scatter returned route cotangents into authoritative source slots."""
    output = _shape_dtype(tuple(int(value) for value in stay_cot.shape), jnp.float32)
    return jax.ffi.ffi_call("pmpp_route_transpose_scatter", output)(
        stay_cot.astype(jnp.float32),
        send_left_cot.astype(jnp.float32),
        send_right_cot.astype(jnp.float32),
        stay_pos.astype(jnp.int32),
        stay_valid.astype(jnp.uint8),
        send_left_pos.astype(jnp.int32),
        send_left_valid.astype(jnp.uint8),
        send_right_pos.astype(jnp.int32),
        send_right_valid.astype(jnp.uint8),
        auth_size=np.int32(auth_size),
        share_capacity=np.int32(share_capacity),
    )


__all__ = [
    "enabled_for_configuration",
    "extension_status",
    "route_merge",
    "route_pack",
    "route_transpose_scatter",
    "route_transpose_split",
    "supported_configuration",
]
