"""Explicit, non-production backend policy and resolution diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import os
from typing import Any


ROUTING_BACKENDS = ("auto", "portable", "cuda_current", "cuda_bidir_mergepath")
CIC_BACKENDS = ("auto", "reference", "mosaic_atomic", "cuda_binned")
LOCAL_PAIR_BACKENDS = (
    "auto",
    "dense",
    "shell",
    "shell_fused_halo",
    "shell_fused_halo_coarse",
)
REDUCTION_BACKENDS = ("auto", "separate", "fused")
RECORD_FORMAT_VERSION = 2


@dataclass(frozen=True)
class OptimizationPolicy:
    """Backend choices for development and benchmark workers.

    This is intentionally separate from :class:`pmpp.configuration.Configuration`
    so production configuration and checkpoints do not acquire experimental
    fields before a four-H100 verdict exists.
    """

    routing: str = "auto"
    cic: str = "auto"
    local_pair: str = "auto"
    phase_reductions: str = "auto"
    cic_tile_size: int | None = None
    require_requested_backends: bool = False

    def __post_init__(self):
        choices = {
            "routing": ROUTING_BACKENDS,
            "cic": CIC_BACKENDS,
            "local_pair": LOCAL_PAIR_BACKENDS,
            "phase_reductions": REDUCTION_BACKENDS,
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {allowed}")
        if self.cic_tile_size is not None and int(self.cic_tile_size) not in (128, 256, 512):
            raise ValueError("cic_tile_size must be one of 128, 256, or 512")


@dataclass(frozen=True)
class OptimizationStatus:
    """Fully serializable record of requested and resolved backends."""

    requested: dict[str, Any]
    resolved: dict[str, str]
    reasons: dict[str, str]
    jax_version: str | None
    jaxlib_version: str | None
    platform: str | None
    compute_capability: tuple[str, ...]
    loaded_cuda_library: str | None
    cuda_build_identifier: str | None
    embedded_cuda_architectures: tuple[str, ...]
    cic_tile_size: int | None
    routing_record_format_version: int
    implementation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jax_diagnostics() -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    try:
        import jax  # Imported only when a worker explicitly resolves a policy.

        devices = tuple(jax.devices())
        capabilities = tuple(
            str(getattr(device, "compute_capability", "unknown")) for device in devices
            if device.platform in {"gpu", "cuda"}
        )
        return (
            getattr(jax, "__version__", None),
            _version("jaxlib"),
            str(jax.default_backend()),
            capabilities,
        )
    except Exception as exc:  # pragma: no cover - dependency absence is diagnostic.
        return None, _version("jaxlib"), str(exc), ()


def _cuda_status():
    try:
        from .. import cuda_routing
        from .._ffi.status import ffi_status

        cuda_routing._register_targets(strict=False)
        status = ffi_status()
        extension = cuda_routing.extension_status()
        status.update(
            bidir_registered=bool(extension.get("bidir_registered")),
            bidir_targets=tuple(extension.get("bidir_targets", ())),
        )
        return status
    except Exception as exc:
        return {"reason": f"CUDA diagnostic unavailable: {exc}"}


def _resolve_one(requested: str, available: bool, automatic: str, reason: str) -> tuple[str, str]:
    if requested == "auto":
        return automatic, "automatic selection: " + reason
    if available:
        return requested, "requested backend is qualified"
    return "reference", reason


def resolve_policy(policy: OptimizationPolicy | None = None, *, status_only: bool = False) -> OptimizationStatus:
    """Resolve a policy without changing any PM++ production defaults.

    Explicit requests fail when ``require_requested_backends`` is true.  The
    failure occurs here, before a benchmark constructs or compiles a callable.
    """

    policy = policy or OptimizationPolicy()
    jax_version, jaxlib_version, platform, capability = _jax_diagnostics()
    cuda = _cuda_status()
    cuda_ready = bool(cuda.get("registered") and cuda.get("library_abi_valid", True))
    gpu_ready = platform in {"gpu", "cuda"}

    routing_available = {
        "portable": True,
        "cuda_current": cuda_ready,
        # The current artifact has the legacy pack/merge targets.  The
        # bidirectional merge-path target is deliberately unavailable until a
        # library manifest explicitly advertises it.
        "cuda_bidir_mergepath": bool(cuda.get("bidir_registered", False)),
    }
    cic_available = {
        "reference": True,
        "mosaic_atomic": bool(gpu_ready and jax_version and jax_version.startswith("0.6.")),
        "cuda_binned": False,
    }
    local_available = {name: True for name in LOCAL_PAIR_BACKENDS if name != "auto"}
    reduction_available = {"separate": True, "fused": True}

    resolved: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for field, available, automatic in (
        ("routing", routing_available, "cuda_current" if routing_available["cuda_current"] else "portable"),
        ("cic", cic_available, "mosaic_atomic" if cic_available["mosaic_atomic"] else "reference"),
        ("local_pair", local_available, "shell"),
        ("phase_reductions", reduction_available, "fused"),
    ):
        requested = getattr(policy, field)
        if requested == "auto":
            chosen, reason = automatic, "qualified automatic candidate"
        else:
            chosen, reason = _resolve_one(requested, bool(available.get(requested, False)), automatic, f"{requested} is not qualified")
        resolved[field] = chosen
        reasons[field] = reason
        if policy.require_requested_backends and requested != "auto" and chosen != requested:
            raise RuntimeError(
                f"Requested experimental backend {field}={requested!r} fell back to {chosen!r}: {reason}"
            )

    implementation = "optimized" if any(
        value in {"cuda_bidir_mergepath", "cuda_binned", "shell_fused_halo", "shell_fused_halo_coarse", "fused"}
        for value in resolved.values()
    ) else "current" if any(value not in {"portable", "reference", "dense", "separate"} for value in resolved.values()) else "reference"
    if status_only:
        implementation = implementation
    return OptimizationStatus(
        requested={**policy.__dict__},
        resolved=resolved,
        reasons=reasons,
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
        platform=platform,
        compute_capability=capability,
        loaded_cuda_library=cuda.get("library"),
        cuda_build_identifier=cuda.get("build_identifier"),
        embedded_cuda_architectures=tuple(cuda.get("embedded_architectures", ())),
        cic_tile_size=policy.cic_tile_size or 128,
        routing_record_format_version=RECORD_FORMAT_VERSION,
        implementation=implementation,
    )


__all__ = ["OptimizationPolicy", "OptimizationStatus", "resolve_policy"]
