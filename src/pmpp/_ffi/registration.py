"""Strict, version-aware registration of optional CUDA FFI targets."""

from __future__ import annotations

from .status import RECORD_FORMAT_VERSION, ffi_status


def register_targets(*, strict: bool = False) -> dict:
    """Register the current CUDA targets and optionally reject stale artifacts."""

    status = ffi_status()
    if strict and not status["library_abi_valid"]:
        raise RuntimeError(
            "PM++ CUDA routing library has an incompatible record format: "
            f"expected {RECORD_FORMAT_VERSION}, got {status['record_format_version']}"
        )
    try:
        from .. import cuda_routing

        registered = bool(cuda_routing._register_targets(strict=strict))
    except Exception as exc:
        registered = False
        if strict:
            raise RuntimeError(f"failed to register PM++ CUDA FFI targets: {exc}") from exc
    if strict and not registered:
        raise RuntimeError("requested PM++ CUDA FFI backend is unavailable or not registered")
    return {**ffi_status(), "registered": registered}


__all__ = ["register_targets"]
