"""Version-checked optional CUDA FFI compatibility layer."""

from .registration import register_targets
from .status import ffi_status, load_build_manifest

__all__ = ["ffi_status", "load_build_manifest", "register_targets"]
