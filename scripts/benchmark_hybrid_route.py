"""Reproducible four-GPU fused-route latency and compiled-memory benchmark.

Run each policy/traffic case in a fresh process. This driver does not turn an
isolated route result into a full N-body performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.sharding import Mesh

from src.pmpp.core.configuration import Configuration
from src.pmpp.core.utils import AXIS_NAME
from src.pmpp.distributed.configuration import MultiGPUConfiguration
from src.pmpp.distributed.cuda import extension_status


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("neighbor_only", "hybrid"), required=True)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--nmesh", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--particles-per-device", type=int, default=6144)
    parser.add_argument("--neighbor-per-device", type=int, default=64)
    parser.add_argument("--far-per-device", type=int, default=0)
    parser.add_argument("--far-send-capacity", type=int, default=65536)
    parser.add_argument("--far-recv-capacity", type=int, default=65536)
    parser.add_argument("--far-chunk-size", type=int, default=8192)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _memory(analysis):
    if analysis is None:
        return None
    fields = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
              "alias_size_in_bytes")
    return {field: int(getattr(analysis, field)) for field in fields if getattr(analysis, field, None) is not None}


def main():
    args = _args()
    devices = jax.devices("gpu")
    if len(devices) < args.devices:
        raise SystemExit(f"need {args.devices} GPUs, found {len(devices)}")
    if args.devices < 4 or args.nmesh % args.devices:
        raise SystemExit("benchmark requires at least four devices dividing nmesh")
    if not 0 <= args.far_per_device <= args.particles_per_device:
        raise SystemExit("invalid far particle count")
    if args.far_per_device + args.neighbor_per_device > args.particles_per_device:
        raise SystemExit("far and neighbor particle counts exceed the active prefix")
    if args.particles_per_device > args.capacity or args.particles_per_device > args.nmesh**2:
        raise SystemExit("invalid particle or local capacity")
    if args.policy == "neighbor_only" and args.far_per_device:
        raise SystemExit("neighbor_only cannot benchmark far traffic")
    if args.iterations < 1 or args.warmups < 0:
        raise SystemExit("iterations must be positive and warmups non-negative")

    mesh = Mesh(np.asarray(devices[:args.devices]), (AXIS_NAME,))
    hybrid = args.policy == "hybrid"
    conf = Configuration(
        1.0, (args.nmesh,) * 3, mesh_shape=1, float_dtype=jnp.float32,
        multigpu=MultiGPUConfiguration(
            compute_mesh=mesh, mode="mesh_halo", cuda_routing=True,
            cuda_routing_backend="bidir_mergepath", migration_policy=args.policy,
            far_send_capacity=args.far_send_capacity if hybrid else None,
            far_recv_capacity=args.far_recv_capacity if hybrid else None,
            far_chunk_size=args.far_chunk_size if hybrid else None,
        ),
        max_ptcl_per_slice=args.capacity, max_share_ptcl=args.capacity // 2,
        max_halo_share_ptcl=args.capacity // 2,
        max_share_gather_ptcl=args.capacity // 2,
    )
    if conf.mGPU_halo_moving_low_memory is None:
        raise RuntimeError("qualified fused native mover is unavailable")

    pmid = np.zeros((args.devices, args.capacity, 3), np.int16)
    disp = np.zeros((args.devices, args.capacity, 3), np.float32)
    vel = np.zeros_like(disp)
    unused = np.ones((args.devices, args.capacity), bool)
    slab_width = args.nmesh // args.devices
    for source in range(args.devices):
        for row in range(args.particles_per_device):
            pmid[source, row] = (source * slab_width, row // args.nmesh, row % args.nmesh)
        unused[source, :args.particles_per_device] = False
        disp[source, :args.far_per_device, 0] = 2 * slab_width
        first = args.far_per_device
        disp[source, first:first + args.neighbor_per_device, 0] = slab_width
    inputs = (jnp.asarray(pmid.reshape(-1, 3)), jnp.asarray(disp.reshape(-1, 3)),
              jnp.asarray(vel.reshape(-1, 3)), jnp.float32(0),
              jnp.asarray(unused.reshape(-1)))
    route = jax.jit(conf.mGPU_halo_moving_low_memory)
    start = time.perf_counter()
    compiled = route.lower(*inputs).compile()
    compile_seconds = time.perf_counter() - start

    for _ in range(args.warmups):
        jax.block_until_ready(compiled(*inputs))
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = compiled(*inputs)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    if bool(result[5]) or int(result[7]):
        raise RuntimeError(f"routing failed: has_failed={bool(result[5])}, invalid={int(result[7])}")
    output_active = ~np.asarray(result[4])
    if int(output_active.sum()) != args.devices * args.particles_per_device:
        raise RuntimeError("particle count changed in the benchmark")
    result_pmid = np.asarray(result[0]).reshape(args.devices, args.capacity, 3)
    for destination in range(args.devices):
        active = output_active.reshape(args.devices, args.capacity)[destination]
        keys = np.ravel_multi_index(result_pmid[destination, active].T, (args.nmesh,) * 3)
        if np.any(keys[1:] < keys[:-1]):
            raise RuntimeError("routed particle keys are not canonical")

    status = extension_status()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "git_commit": commit,
        "jax_version": jax.__version__, "jaxlib_version": jaxlib.__version__,
        "devices": [str(device) for device in devices[:args.devices]],
        "routing_manifest": status.get("manifest"),
        "policy": args.policy, "nmesh": args.nmesh, "capacity": args.capacity,
        "particles_per_device": args.particles_per_device,
        "neighbor_per_device": args.neighbor_per_device,
        "far_per_device": args.far_per_device,
        "far_send_capacity": conf.multigpu.far_send_capacity,
        "far_recv_capacity": conf.multigpu.far_recv_capacity,
        "far_chunk_size": conf.multigpu.far_chunk_size,
        "compile_seconds": compile_seconds,
        "latency_seconds": times,
        "median_seconds": statistics.median(times),
        "p95_seconds": float(np.percentile(times, 95)),
        "compiled_memory_bytes": _memory(compiled.memory_analysis()),
        "max_particles_moved": int(result[6]),
        "valid_particles": int(output_active.sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("policy", "median_seconds", "p95_seconds",
                                                    "compiled_memory_bytes", "valid_particles")}, indent=2))


if __name__ == "__main__":
    main()
