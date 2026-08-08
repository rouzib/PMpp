"""High-level forward simulation entrypoints."""

from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from .cic import scatter
from .distributed import extension_status
from .initial_conditions import linear_modes, lpt, white_noise, white_noise_nested
from .nbody.solver import nbody, nbody_low_memory_with_telemetry


@dataclass(frozen=True)
class ForwardTelemetry:
    """JSON-serializable timing, routing, capacity, and health diagnostics.

    For ``low_memory``, ``max_authoritative_occupancy`` is sampled initially
    and after every N-body macro-step. Standard runs report final occupancy.
    The low-memory profile reports native migration counts and uses a zero
    gather high-water because ``mesh_halo`` exchanges fixed-width mesh edges,
    not particle gather buffers.
    """

    profile: str
    noise_mode: str
    stage_seconds: dict[str, float]
    num_devices: int
    cuda_routing: bool
    cuda_routing_backend: str | None
    extension_abi: dict[str, Any]
    record_format_version: int | None
    particle_slots: int
    expected_particles: int
    max_ptcl_per_slice: int | None
    max_share_ptcl: int | None
    max_halo_share_ptcl: int | None
    max_share_gather_ptcl: int | None
    max_authoritative_occupancy: int | None
    migration_high_water: int | None
    gather_high_water: int | None
    routing_invalid_count: int | None
    active_slot_fraction: float | None
    density_sum: float | None
    density_mean: float | None
    density_abs_max: float | None
    density_all_finite: bool | None


@dataclass(frozen=True)
class ForwardResult:
    """Density, optional final particles, and inexpensive health telemetry."""

    particles: Any | None
    density: Any
    telemetry: ForwardTelemetry


def _validate_prepared_cosmology(prepared_cosmo, conf):
    """Reject an unprepared or geometrically incompatible cosmology."""
    if prepared_cosmo.transfer is None or prepared_cosmo.growth is None:
        raise ValueError("prepared_cosmo must have transfer and growth tables; call boltzmann first.")
    cosmo_conf = prepared_cosmo.conf
    if tuple(cosmo_conf.ptcl_grid_shape) != tuple(conf.ptcl_grid_shape):
        raise ValueError("prepared_cosmo and conf must use the same particle grid shape.")
    if tuple(cosmo_conf.mesh_shape) != tuple(conf.mesh_shape):
        raise ValueError("prepared_cosmo and conf must use the same force mesh shape.")
    if cosmo_conf.cosmo_dtype != conf.cosmo_dtype:
        raise ValueError("prepared_cosmo and conf must use the same cosmology dtype.")


def _resolve_low_memory_lpt():
    """Resolve the optional low-memory LPT implementation at call time."""
    module = import_module("pmpp.initial_conditions.lpt")
    low_memory_lpt = getattr(module, "lpt_low_memory_with_telemetry", None)
    if low_memory_lpt is None:
        raise RuntimeError(
            "profile='low_memory' requires pmpp.initial_conditions.lpt.lpt_low_memory_with_telemetry, "
            "which is not available in this checkout."
        )
    return low_memory_lpt


def _validate_low_memory_profile(conf):
    """Fail before allocating large fields when the low-memory contract is unmet."""
    errors = []
    status = extension_status()
    if conf.dim != 3:
        errors.append("dim must be 3")
    if jnp.dtype(conf.float_dtype) != jnp.dtype(jnp.float32):
        errors.append("float_dtype must be float32")
    if jnp.dtype(conf.pmid_dtype) != jnp.dtype(jnp.int16):
        errors.append("pmid_dtype must be int16")
    if conf.compute_mesh is None or not conf.use_mGPU:
        errors.append("a distributed compute mesh is required")
    elif conf.multigpu_mode != "mesh_halo":
        errors.append("multigpu mode must be 'mesh_halo'")
    if conf.replicated_mesh:
        errors.append("replicated meshes are not supported")
    if tuple(conf.mesh_shape) != tuple(conf.ptcl_grid_shape):
        errors.append("mesh_shape must equal ptcl_grid_shape (mesh_shape=1)")
    capacity = conf.max_ptcl_per_slice
    local_particles = conf.ptcl_num // int(conf.num_devices or 1)
    if capacity is None or capacity < local_particles or capacity > jnp.iinfo(jnp.int32).max:
        errors.append("max_ptcl_per_slice must explicitly fit the local particles and signed int32 counts")
    for name in ("max_share_ptcl", "max_halo_share_ptcl", "max_share_gather_ptcl"):
        value = getattr(conf, name)
        if value is None or value <= 0 or value > jnp.iinfo(jnp.int32).max:
            errors.append(f"{name} must be an explicit positive signed-int32 capacity")
    if conf.lpt_order != 2:
        errors.append("lpt_order must be 2")
    if conf.lpt_cache_strains:
        errors.append("lpt_cache_strains must be False")
    if not conf.cuda_routing:
        errors.append("qualified native CUDA routing must be active")
    if conf.cuda_routing_backend != "bidir_mergepath":
        errors.append("cuda_routing_backend must be 'bidir_mergepath'")
    if not status.get("registered") or not status.get("bidir_registered"):
        errors.append("the native bidirectional CUDA routing extension must be registered")
    if status.get("record_format_version") != 3:
        errors.append("the CUDA routing extension must use record_format_version=3")
    if not status.get("fused_primal_registered") or not status.get("fused_primal_feature"):
        errors.append("the CUDA routing extension must advertise and register the fused primal drift target")
    if getattr(conf, "mGPU_halo_moving_low_memory", None) is None:
        errors.append("the fused low-memory drift-route mover must be active")
    if conf.mGPU_irfftn_transposed is None:
        errors.append("the scalar distributed inverse FFT must be available")
    if errors:
        raise ValueError("invalid low-memory forward configuration: " + "; ".join(errors) + ".")
    return status


def _contains_tracer(value):
    """Return whether a pytree contains values under a JAX transformation."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(value))


def _timed_stage(function, *, tracing=False):
    """Run and synchronize one stage so its elapsed time is meaningful."""
    started = perf_counter()
    result = function()
    if tracing or _contains_tracer(result):
        return result, 0.0
    jax.block_until_ready(result)
    return result, float(perf_counter() - started)


def _optional_int(value):
    """Return one capacity as a built-in integer or ``None``."""
    return None if value is None else int(value)


def _jsonable(value):
    """Recursively convert extension diagnostics to JSON-native values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


@jax.jit
def _count_active_slots(unused_index):
    """Count valid slots on one addressable particle shard."""
    return jnp.sum(~unused_index, dtype=jnp.int32)


@jax.jit
def _count_authoritative_slots(unused_index, halo_mask):
    """Count non-halo valid slots on one addressable particle shard."""
    return jnp.sum((~unused_index) & (~halo_mask), dtype=jnp.int32)


@jax.jit
def _density_health(density):
    """Fuse density diagnostics so full-size boolean/absolute meshes do not escape."""
    return jnp.sum(density), jnp.mean(density), jnp.max(jnp.abs(density)), jnp.all(jnp.isfinite(density))


def _particle_occupancies(particles):
    """Return active totals and final authoritative maxima via shard-local reductions."""
    disp_shards = particles.disp.addressable_shards
    unused_shards = None if particles.unused_index is None else particles.unused_index.addressable_shards
    halo_shards = None if particles.halo_mask is None else particles.halo_mask.addressable_shards

    if unused_shards is None:
        active_counts = [int(shard.data.shape[0]) for shard in disp_shards]
    else:
        active_counts = [
            int(value) for value in jax.device_get(tuple(_count_active_slots(shard.data) for shard in unused_shards))
        ]

    if halo_shards is None:
        authoritative_counts = active_counts
    elif unused_shards is None:
        authoritative_counts = [
            int(value) for value in jax.device_get(tuple(_count_active_slots(shard.data) for shard in halo_shards))
        ]
    else:
        if len(unused_shards) != len(halo_shards):
            raise RuntimeError("particle validity and halo masks have incompatible shardings.")
        authoritative_counts = [
            int(value) for value in jax.device_get(
                tuple(
                    _count_authoritative_slots(unused.data, halo.data)
                    for unused, halo in zip(unused_shards, halo_shards)
                )
            )
        ]
    return sum(active_counts), max(authoritative_counts, default=0)


def _forward_telemetry(
    particles, density, conf, profile, noise_mode, stage_seconds, routing_extension_status=None,
    max_authoritative_occupancy=None, migration_high_water=None, gather_high_water=None, routing_invalid_count=None,
):
    """Build built-in scalar diagnostics without materializing particle copies."""
    particle_slots = int(particles.disp.shape[0])
    num_devices = int(conf.num_devices or 1)
    if routing_extension_status is None:
        routing_extension_status = extension_status()
    routing_extension_status = _jsonable(routing_extension_status)
    record_format_version = routing_extension_status.get("record_format_version")
    if _contains_tracer((particles, density)):
        return ForwardTelemetry(
            profile=profile, noise_mode=noise_mode, stage_seconds={
                str(name): float(seconds)
                for name, seconds in stage_seconds.items()
            }, num_devices=num_devices, cuda_routing=bool(conf.cuda_routing),
            cuda_routing_backend=None if conf.cuda_routing_backend is None else str(conf.cuda_routing_backend),
            extension_abi=routing_extension_status,
            record_format_version=None if record_format_version is None else int(record_format_version),
            particle_slots=particle_slots, expected_particles=int(conf.ptcl_num), max_ptcl_per_slice=_optional_int(
                conf.max_ptcl_per_slice
            ), max_share_ptcl=_optional_int(conf.max_share_ptcl
                                            ), max_halo_share_ptcl=_optional_int(conf.max_halo_share_ptcl),
            max_share_gather_ptcl=_optional_int(conf.max_share_gather_ptcl), max_authoritative_occupancy=None,
            migration_high_water=None, gather_high_water=None, routing_invalid_count=None, active_slot_fraction=None,
            density_sum=None, density_mean=None, density_abs_max=None, density_all_finite=None,
        )

    active_slots, final_max_occupancy = _particle_occupancies(particles)
    if max_authoritative_occupancy is None:
        max_authoritative_occupancy = final_max_occupancy
    else:
        max_authoritative_occupancy = int(jax.device_get(max_authoritative_occupancy))
    if migration_high_water is not None:
        migration_high_water = int(jax.device_get(migration_high_water))
    if gather_high_water is not None:
        gather_high_water = int(jax.device_get(gather_high_water))
    if routing_invalid_count is not None:
        routing_invalid_count = int(jax.device_get(routing_invalid_count))
    density_sum, density_mean, density_abs_max, finite = jax.device_get(_density_health(density))
    return ForwardTelemetry(
        profile=profile, noise_mode=noise_mode, stage_seconds={
            str(name): float(seconds)
            for name, seconds in stage_seconds.items()
        }, num_devices=num_devices, cuda_routing=bool(conf.cuda_routing),
        cuda_routing_backend=None if conf.cuda_routing_backend is None else str(conf.cuda_routing_backend),
        extension_abi=routing_extension_status,
        record_format_version=None if record_format_version is None else int(record_format_version),
        particle_slots=particle_slots, expected_particles=int(conf.ptcl_num),
        max_ptcl_per_slice=_optional_int(conf.max_ptcl_per_slice), max_share_ptcl=_optional_int(conf.max_share_ptcl),
        max_halo_share_ptcl=_optional_int(conf.max_halo_share_ptcl), max_share_gather_ptcl=_optional_int(
            conf.max_share_gather_ptcl
        ), max_authoritative_occupancy=int(max_authoritative_occupancy), migration_high_water=migration_high_water,
        gather_high_water=gather_high_water, routing_invalid_count=routing_invalid_count,
        active_slot_fraction=float(active_slots / particle_slots), density_sum=float(density_sum),
        density_mean=float(density_mean), density_abs_max=float(density_abs_max), density_all_finite=bool(finite),
    )


def run_forward(
    seed, prepared_cosmo, conf, *, profile="standard", noise_mode="standard", retain_particles=True,
) -> ForwardResult:
    """Run white noise through linear modes, LPT, N-body, and final scatter.

    Parameters
    ----------
    seed : int
        White-noise seed.
    prepared_cosmo : Cosmology
        Cosmology with transfer and growth tables already populated.
    conf : Configuration
        Simulation configuration shared by all stages.
    profile : {"standard", "low_memory"}, optional
        ``standard`` preserves the established differentiable pipeline.
        ``low_memory`` selects low-memory LPT and forward-only streamed-gravity
        N-body implementations and validates their required configuration.
    noise_mode : {"standard", "nested"}, optional
        White-noise generator to use.
    retain_particles : bool, optional
        Whether the final particle state is retained in the result. Setting it
        to False gives the compiler and allocator more freedom after scatter.
    """
    if profile not in {"standard", "low_memory"}:
        raise ValueError("profile must be 'standard' or 'low_memory'.")
    if noise_mode not in {"standard", "nested"}:
        raise ValueError("noise_mode must be 'standard' or 'nested'.")
    if not isinstance(retain_particles, bool):
        raise TypeError("retain_particles must be a bool.")

    _validate_prepared_cosmology(prepared_cosmo, conf)
    tracing = _contains_tracer((seed, prepared_cosmo))
    routing_extension_status = None
    if profile == "low_memory":
        routing_extension_status = _validate_low_memory_profile(conf)
        if tracing:
            raise NotImplementedError("run_forward(profile='low_memory') is forward-only and cannot be transformed.")
        lpt_fn = _resolve_low_memory_lpt()
        nbody_fn = nbody_low_memory_with_telemetry
    else:
        lpt_fn = lpt
        nbody_fn = nbody

    noise_fn = white_noise if noise_mode == "standard" else white_noise_nested
    stage_seconds = {}
    modes, stage_seconds["white_noise"] = _timed_stage(lambda: noise_fn(seed, conf), tracing=tracing)
    tracing = tracing or _contains_tracer(modes)
    if profile == "low_memory" and tracing:
        raise NotImplementedError("run_forward(profile='low_memory') is forward-only and cannot be transformed.")
    modes, stage_seconds["linear_modes"] = _timed_stage(
        lambda: linear_modes(modes, prepared_cosmo, conf), tracing=tracing,
    )
    tracing = tracing or _contains_tracer(modes)
    lpt_result, stage_seconds["lpt"] = _timed_stage(lambda: lpt_fn(modes, prepared_cosmo, conf), tracing=tracing, )
    del modes
    max_authoritative_occupancy = None
    migration_high_water = None
    gather_high_water = None
    routing_invalid_count = None
    if profile == "low_memory":
        particles, lpt_migration, lpt_invalid = lpt_result
        nbody_result, stage_seconds["nbody"] = _timed_stage(
            lambda: nbody_fn(particles, prepared_cosmo, conf), tracing=tracing,
        )
        particles, max_authoritative_occupancy, nbody_migration, nbody_invalid = nbody_result
        migration_high_water = jnp.maximum(lpt_migration, nbody_migration)
        routing_invalid_count = jnp.maximum(lpt_invalid, nbody_invalid)
        gather_high_water = jnp.int32(0)
    else:
        particles = lpt_result
        particles, stage_seconds["nbody"] = _timed_stage(
            lambda: nbody_fn(particles, prepared_cosmo, conf), tracing=tracing,
        )
    density, stage_seconds["scatter"] = _timed_stage(lambda: scatter(particles, conf), tracing=tracing)
    stage_seconds["total"] = float(sum(stage_seconds.values()))
    telemetry = _forward_telemetry(
        particles, density, conf, profile, noise_mode, stage_seconds, routing_extension_status,
        max_authoritative_occupancy, migration_high_water, gather_high_water, routing_invalid_count,
    )
    return ForwardResult(particles=particles if retain_particles else None, density=density, telemetry=telemetry)


__all__ = ["ForwardResult", "ForwardTelemetry", "run_forward"]
