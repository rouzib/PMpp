"""Filesystem locations shared by the optional CUDA routing tools."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import re


def package_cuda_directory() -> Path:
    """Return the package-local directory for a compiled routing artifact."""
    return Path(__file__).resolve().parent / "_cuda"


def _distribution_version(distribution: str) -> str:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "source"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", version)


def user_cache_cuda_directory() -> Path:
    """Return a versioned user-cache directory for read-only installations."""
    explicit = os.environ.get("PMPP_CUDA_ROUTING_CACHE")
    if explicit:
        return Path(explicit).expanduser().resolve()

    cache_root = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
    return (
        base / "pmpp" / "cuda-routing" / f"pmpp-{_distribution_version('pmpp')}" /
        f"jaxlib-{_distribution_version('jaxlib')}"
    )


__all__ = ["package_cuda_directory", "user_cache_cuda_directory"]
