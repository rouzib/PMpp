"""Diagnostics for the optional CUDA routing artifact."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from typing import Any


RECORD_FORMAT_VERSION = 2


def _library_candidates() -> tuple[Path, ...]:
    explicit = os.environ.get("PMPP_CUDA_ROUTING_LIBRARY")
    paths = []
    if explicit:
        paths.append(Path(explicit))
    try:
        from ..cuda_routing import _candidate_library_paths

        paths.extend(_candidate_library_paths())
    except Exception:
        pass
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def load_build_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Load the manifest beside a CUDA library or from an explicit path."""

    candidates = [Path(path)] if path else []
    if path is None:
        for library in _library_candidates():
            candidates.extend(
                [
                    library.with_suffix(library.suffix + ".manifest.json"),
                    library.with_name("pmpp_cuda_routing.manifest.json"),
                    library.parent / "pmpp_cuda_routing.manifest.json",
                ]
            )
        explicit_manifest = os.environ.get("PMPP_CUDA_ROUTING_MANIFEST")
        if explicit_manifest:
            candidates.insert(0, Path(explicit_manifest))
    for candidate in dict.fromkeys(candidates):
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class FFIStatus:
    library: str | None
    registered: bool
    qualified_jax: bool
    jax_version: str | None
    jaxlib_version: str | None
    platform: str | None
    build_identifier: str | None
    library_abi_valid: bool
    record_format_version: int
    embedded_architectures: tuple[str, ...]
    manifest: dict[str, Any] | None
    reason: str | None = None

    def to_dict(self):
        return asdict(self)


def ffi_status() -> dict[str, Any]:
    """Return complete non-throwing FFI status for manifests and reports."""

    manifest = load_build_manifest()
    try:
        import jax
        import jaxlib
        from .. import cuda_routing

        extension = cuda_routing.extension_status()
        library = extension.get("library")
        registered = bool(extension.get("registered"))
        qualified = bool(extension.get("qualified_jax"))
        platform = str(jax.default_backend())
        jax_version = getattr(jax, "__version__", None)
        jaxlib_version = getattr(jaxlib, "__version__", None)
        reason = None
    except Exception as exc:  # pragma: no cover - optional runtime.
        library = None
        registered = False
        qualified = False
        platform = None
        jax_version = None
        try:
            import importlib.metadata

            jaxlib_version = importlib.metadata.version("jaxlib")
        except Exception:
            jaxlib_version = None
        reason = str(exc)
    record_version = int((manifest or {}).get("record_format_version", RECORD_FORMAT_VERSION))
    abi_valid = record_version == RECORD_FORMAT_VERSION if manifest else True
    architectures = tuple(str(value) for value in (manifest or {}).get("embedded_cuda_architectures", ()))
    result = FFIStatus(
        library=library,
        registered=registered,
        qualified_jax=qualified,
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
        platform=platform,
        build_identifier=(manifest or {}).get("build_identifier"),
        library_abi_valid=abi_valid,
        record_format_version=record_version,
        embedded_architectures=architectures,
        manifest=manifest,
        reason=reason,
    ).to_dict()
    try:
        from .. import cuda_routing

        extension = cuda_routing.extension_status()
        result["bidir_registered"] = bool(extension.get("bidir_registered"))
        result["bidir_targets"] = tuple(extension.get("bidir_targets", ()))
    except Exception:
        result["bidir_registered"] = False
        result["bidir_targets"] = ()
    return result


__all__ = ["FFIStatus", "ffi_status", "load_build_manifest"]
