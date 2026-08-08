"""Exact-shape, forward-only qualification probes for the H200 runner.

JAX imports stay inside probe functions so importing the supervisor does not
initialize a CUDA client before allocator and visibility policy are fixed.
"""

from __future__ import annotations

from typing import Any

PROBE_NAMES = ("distributed_fft", "native_route", "fused_grid", "pallas_cic", "nbody_step")


def _host_sum_int32_counts(counts: Any) -> int:
    """Aggregate device-local int32 counts with unbounded host integers."""
    return sum(int(value) for value in counts)


def _require_distributed(conf: Any) -> None:
    if conf.compute_mesh is None or int(conf.num_devices or 0) < 1:
        raise RuntimeError("qualification probes require an explicit distributed compute mesh")


def _require_native_route(conf: Any) -> dict[str, Any]:
    from .distributed import extension_status
    from .distributed.cuda import supported_fused_primal_configuration

    status = extension_status()
    if not bool(conf.cuda_routing) or conf.cuda_routing_backend != "bidir_mergepath":
        raise RuntimeError("native_route probe requires active bidir_mergepath routing")
    if not status.get("registered") or not status.get("bidir_registered"):
        raise RuntimeError("native_route probe requires the registered bidirectional CUDA extension")
    if int(status.get("record_format_version", -1)) != 3:
        raise RuntimeError(f"native_route probe requires record-format ABI v3, got {status}")
    if not status.get("fused_primal_feature") or not status.get("fused_primal_registered"):
        raise RuntimeError(
            "native_route probe requires the manifested and registered "
            "fused_drift_primal_i16_f32 target"
        )
    if not supported_fused_primal_configuration(conf):
        raise RuntimeError("native_route probe configuration is unsupported by the fused primal CUDA ABI")
    return status


def _particle_counts(particles: Any) -> tuple[int, int]:
    """Count valid particles without an overflowing global int32 reduction."""
    import jax
    import jax.numpy as jnp

    if particles.unused_index is None:
        counts = [int(shard.data.shape[0]) for shard in particles.disp.addressable_shards]
    else:
        count_valid = jax.jit(lambda unused: jnp.sum(~unused, dtype=jnp.int32))
        counts = [int(jax.device_get(count_valid(shard.data))) for shard in particles.unused_index.addressable_shards]
    return _host_sum_int32_counts(counts), max(counts, default=0)


def _changed_particle_rows(before: Any, after: Any, unused_index: Any) -> int:
    """Count changed valid rows locally, then aggregate without int32 wrap."""
    import jax
    import jax.numpy as jnp

    count_changed = jax.jit(
        lambda local_before, local_after, local_unused: jnp.sum(
            jnp.any(local_before != local_after, axis=1) & (~local_unused), dtype=jnp.int32,
        )
    )
    counts = [
        int(jax.device_get(count_changed(before_shard.data, after_shard.data, unused_shard.data)))
        for before_shard, after_shard, unused_shard in
        zip(before.addressable_shards, after.addressable_shards, unused_index.addressable_shards)
    ]
    return _host_sum_int32_counts(counts)


def _canonical_grid(conf: Any) -> Any:
    from .initial_conditions.lpt import _low_memory_particle_grid

    particles = _low_memory_particle_grid(conf)
    import jax

    jax.block_until_ready(particles)
    return particles


def probe_distributed_fft(conf: Any, *, fft_tolerance: float = 5e-5, **_: Any) -> dict[str, Any]:
    """Run one exact-global-shape scalar distributed real-FFT round trip."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from .core import AXIS_NAME

    _require_distributed(conf)
    if conf.mGPU_rfftn_transposed is None or conf.mGPU_irfftn_transposed is None:
        raise RuntimeError("scalar distributed FFT functions are unavailable")
    sharding = NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None))
    shape = tuple(int(value) for value in conf.mesh_shape)

    def make_field():
        x = jnp.arange(shape[0], dtype=conf.float_dtype)[:, None, None]
        y = jnp.arange(shape[1], dtype=conf.float_dtype)[None, :, None]
        z = jnp.arange(shape[2], dtype=conf.float_dtype)[None, None, :]
        scale = jnp.asarray(2 * jnp.pi, dtype=conf.float_dtype)
        return (
            jnp.sin(scale * x / shape[0]) + jnp.asarray(0.25, conf.float_dtype) * jnp.cos(scale * y / shape[1]) +
            jnp.asarray(0.125, conf.float_dtype) * jnp.sin(scale * z / shape[2])
        )

    field = jax.jit(make_field, out_shardings=sharding)()
    field.block_until_ready()

    @jax.jit
    def round_trip(value):
        spectrum = conf.mGPU_rfftn_transposed(value)
        recovered = conf.mGPU_irfftn_transposed(spectrum).astype(conf.float_dtype)
        error = recovered - value
        return (
            jnp.max(jnp.abs(error)), jnp.sqrt(jnp.mean(error**2)), jnp.max(jnp.abs(value)),
            jnp.all(jnp.isfinite(recovered)),
        )

    max_abs, rms, reference_max, finite = jax.device_get(round_trip(field))
    max_abs = float(max_abs)
    reference_max = float(reference_max)
    relative_max = max_abs / max(reference_max, 1.0)
    if not bool(finite) or relative_max > float(fft_tolerance):
        raise RuntimeError(
            f"distributed FFT round trip failed: finite={bool(finite)}, relative_max={relative_max:.3e}, "
            f"tolerance={fft_tolerance:.3e}"
        )
    return {
        "probe": "distributed_fft",
        "global_shape": list(shape),
        "dtype": str(conf.float_dtype),
        "scalar_fft": True,
        "max_abs_error": max_abs,
        "rms_error": float(rms),
        "relative_max_error": relative_max,
        "tolerance": float(fft_tolerance),
        "finite": bool(finite),
    }


def probe_fused_grid(conf: Any, **_: Any) -> dict[str, Any]:
    """Create the exact-capacity fused canonical grid and validate its layout."""
    import jax.numpy as jnp

    _require_distributed(conf)
    particles = _canonical_grid(conf)
    valid_particles, max_occupancy = _particle_counts(particles)
    if valid_particles != int(conf.ptcl_num):
        raise RuntimeError(f"fused grid particle count mismatch: {valid_particles} != {conf.ptcl_num}")
    if particles.pmid.dtype != jnp.dtype(conf.pmid_dtype):
        raise RuntimeError(f"fused grid coordinate dtype mismatch: {particles.pmid.dtype} != {conf.pmid_dtype}")
    return {
        "probe": "fused_grid",
        "global_particle_slots": int(particles.disp.shape[0]),
        "capacity_per_device": int(conf.max_ptcl_per_slice),
        "valid_particles": valid_particles,
        "expected_particles": int(conf.ptcl_num),
        "max_device_occupancy": max_occupancy,
        "pmid_dtype": str(particles.pmid.dtype),
        "float_dtype": str(particles.disp.dtype),
    }


def probe_native_route(conf: Any, *, route_migrants: int | None = None, **_: Any) -> dict[str, Any]:
    """Exercise the production fused drift-route at full particle capacity."""
    from functools import partial

    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import PartitionSpec as P

    from .core import AXIS_NAME

    _require_distributed(conf)
    status = _require_native_route(conf)
    mover = getattr(conf, "mGPU_halo_moving_low_memory", None)
    if mover is None:
        raise RuntimeError("native_route probe requires the fused low-memory drift-route mover")
    particles = _canonical_grid(conf)
    plane_size = int(conf.ptcl_grid_shape[1] * conf.ptcl_grid_shape[2])
    if route_migrants is None:
        route_migrants = min(plane_size, max(1, int(conf.max_share_ptcl) // 4))
    route_migrants = int(route_migrants)
    if route_migrants <= 0 or route_migrants > plane_size:
        raise ValueError(f"route_migrants must be in [1, {plane_size}], got {route_migrants}")
    slice_width = int(conf.ptcl_grid_shape[0] // conf.num_devices)
    migration_start = (slice_width - 1) * plane_size

    @partial(
        shard_map, mesh=conf.compute_mesh, in_specs=P(AXIS_NAME, None), out_specs=P(AXIS_NAME, None), check_vma=False,
    )
    def inject_migration_velocity(local_vel):
        return local_vel.at[migration_start:migration_start + route_migrants,
                            0].set(jnp.asarray(1.25 * conf.cell_size, conf.float_dtype))

    inject_migration_velocity = jax.jit(inject_migration_velocity, donate_argnums=(0, ))
    particles = particles.replace(vel=inject_migration_velocity(particles.vel))
    particles.vel.block_until_ready()

    @partial(jax.jit, donate_argnums=(1, 2, 3))
    def route(pmid, reference_disp, vel, unused_index):
        return mover(pmid, reference_disp, vel, jnp.asarray(1, dtype=conf.float_dtype), unused_index)

    pmid, disp, vel, halo_mask, unused_index, has_failed, max_moved, invalid_count = route(
        particles.pmid, particles.disp, particles.vel, particles.unused_index,
    )
    jax.block_until_ready((pmid, disp, vel, halo_mask, unused_index, has_failed, max_moved, invalid_count))
    if bool(jax.device_get(has_failed)):
        raise RuntimeError("native_route probe reported a static-capacity overflow")
    invalid_count = int(jax.device_get(invalid_count))
    if invalid_count != 0:
        raise RuntimeError(f"native_route probe reported {invalid_count} invalid fused route records")
    max_moved = int(jax.device_get(max_moved))
    if max_moved != route_migrants:
        raise RuntimeError(f"native_route probe moved {max_moved} rows per device, expected {route_migrants}")
    routed = particles.replace(pmid=pmid, disp=disp, vel=vel, halo_mask=halo_mask, unused_index=unused_index, )
    valid_particles, max_occupancy = _particle_counts(routed)
    if valid_particles != int(conf.ptcl_num):
        raise RuntimeError(f"native route lost particles: {valid_particles} != {conf.ptcl_num}")

    changed = _changed_particle_rows(particles.pmid, pmid, unused_index)
    if changed <= 0:
        raise RuntimeError("native_route probe did not migrate any particle")
    return {
        "probe": "native_route",
        "capacity_per_device": int(conf.max_ptcl_per_slice),
        "share_capacity": int(conf.max_share_ptcl),
        "requested_migrants_per_device": route_migrants,
        "donated_route_inputs": ["disp", "vel", "unused_index"],
        "changed_particle_rows": changed,
        "max_mover_count": max_moved,
        "invalid_record_count": invalid_count,
        "valid_particles": valid_particles,
        "max_device_occupancy": max_occupancy,
        "routing_backend": str(conf.cuda_routing_backend),
        "record_format_version": int(status["record_format_version"]),
        "fused_primal_feature": bool(status["fused_primal_feature"]),
        "fused_primal_registered": bool(status["fused_primal_registered"]),
        "build_identifier": status.get("build_identifier"),
    }


def probe_pallas_cic(conf: Any, *, cic_tolerance: float = 2e-5, **_: Any) -> dict[str, Any]:
    """Run exact-capacity Pallas scatter and gather on a uniform grid."""
    import jax
    import jax.numpy as jnp

    from .cic import gather, pallas_cic_supported, scatter

    _require_distributed(conf)
    if not conf.pallas_cic or not pallas_cic_supported(conf.float_dtype):
        raise RuntimeError("pallas_cic probe requires the active float32 GPU Pallas path")
    particles = _canonical_grid(conf)

    @jax.jit
    def scatter_gather(state):
        density = scatter(state, conf)
        sampled = gather(state, conf, density)
        valid = jnp.ones(sampled.shape, dtype=jnp.bool_) if state.unused_index is None else ~state.unused_index
        sampled_error = jnp.where(valid, jnp.abs(sampled - 1), 0)
        return (
            jnp.mean(density), jnp.max(jnp.abs(density - 1)), jnp.max(sampled_error),
            jnp.all(jnp.isfinite(density)) & jnp.all(jnp.isfinite(sampled)),
        )

    density_mean, density_error, gather_error, finite = jax.device_get(scatter_gather(particles))
    density_error = float(density_error)
    gather_error = float(gather_error)
    if not bool(finite) or max(density_error, gather_error) > float(cic_tolerance):
        raise RuntimeError(
            f"Pallas CIC probe failed: finite={bool(finite)}, density_error={density_error:.3e}, "
            f"gather_error={gather_error:.3e}, tolerance={cic_tolerance:.3e}"
        )
    return {
        "probe": "pallas_cic",
        "global_mesh_shape": list(conf.mesh_shape),
        "capacity_per_device": int(conf.max_ptcl_per_slice),
        "density_mean": float(density_mean),
        "density_max_error": density_error,
        "gather_max_error": gather_error,
        "tolerance": float(cic_tolerance),
        "finite": bool(finite),
        "pallas_active": True,
    }


def probe_nbody_step(conf: Any, *, prepared_cosmo: Any = None, **_: Any) -> dict[str, Any]:
    """Run one exact-capacity low-memory N-body macro-step."""
    from functools import partial

    import jax
    import jax.numpy as jnp

    from .nbody.solver import nbody_low_memory_with_telemetry

    _require_distributed(conf)
    _require_native_route(conf)
    if prepared_cosmo is None:
        raise ValueError("nbody_step probe requires a prepared cosmology")
    if int(conf.a_nbody_num) != 1:
        raise ValueError(f"nbody_step probe requires exactly one macro-step, got {conf.a_nbody_num}")
    particles = _canonical_grid(conf)

    @partial(jax.jit, donate_argnums=(2, ))
    def perturb(pmid, unused_index, disp):
        phase = jnp.asarray(2 * jnp.pi / conf.ptcl_grid_shape[0], conf.float_dtype)
        amplitude = jnp.asarray(0.05 * conf.ptcl_spacing, conf.float_dtype)
        for axis in range(3):
            x = pmid[:, 0].astype(conf.float_dtype)
            if axis == 0:
                component = jnp.sin(phase * x)
            elif axis == 1:
                component = jnp.sin(phase * (x + pmid[:, 1].astype(conf.float_dtype)))
            else:
                component = jnp.cos(phase * (x + pmid[:, 2].astype(conf.float_dtype)))
            component = amplitude * component
            component = jnp.where(unused_index, 0, component)
            disp = disp.at[:, axis].set(component)
        return disp

    particles = particles.replace(disp=perturb(particles.pmid, particles.unused_index, particles.disp))
    result, max_occupancy, max_migration, invalid_count = nbody_low_memory_with_telemetry(
        particles, prepared_cosmo, conf,
    )

    @jax.jit
    def health(after):
        finite = (
            jnp.all(jnp.isfinite(after.disp)) & jnp.all(jnp.isfinite(after.vel)) & jnp.all(jnp.isfinite(after.acc))
        )
        return jnp.max(jnp.abs(after.vel)), finite

    max_velocity, finite = jax.device_get(health(result))
    valid_particles, final_occupancy = _particle_counts(result)
    if valid_particles != int(conf.ptcl_num):
        raise RuntimeError(f"N-body step lost particles: {valid_particles} != {conf.ptcl_num}")
    if not bool(finite) or float(max_velocity) <= 0:
        raise RuntimeError(
            f"N-body step health check failed: finite={bool(finite)}, max_velocity={float(max_velocity)}"
        )
    invalid_count = int(jax.device_get(invalid_count))
    if invalid_count:
        raise RuntimeError(f"N-body step route reported {invalid_count} invalid particles")
    return {
        "probe": "nbody_step",
        "macro_steps": 1,
        "capacity_per_device": int(conf.max_ptcl_per_slice),
        "valid_particles": valid_particles,
        "final_max_device_occupancy": final_occupancy,
        "trajectory_max_device_occupancy": int(jax.device_get(max_occupancy)),
        "migration_high_water": int(jax.device_get(max_migration)),
        "routing_invalid_count": invalid_count,
        "donated_probe_inputs": ["disp"],
        "max_velocity": float(max_velocity),
        "finite": bool(finite),
    }


_PROBES = {
    "distributed_fft": probe_distributed_fft,
    "native_route": probe_native_route,
    "fused_grid": probe_fused_grid,
    "pallas_cic": probe_pallas_cic,
    "nbody_step": probe_nbody_step,
}


def run_probe(name: str, conf: Any, **kwargs: Any) -> dict[str, Any]:
    """Run one named probe with an injected configuration and dimensions."""
    try:
        function = _PROBES[name]
    except KeyError as error:
        raise ValueError(f"unknown qualification probe {name!r}; expected one of {PROBE_NAMES}") from error
    result = function(conf, **kwargs)
    if result.get("probe") != name:
        raise RuntimeError(f"probe {name!r} returned inconsistent evidence {result!r}")
    return result


__all__ = ["PROBE_NAMES", "run_probe"]
