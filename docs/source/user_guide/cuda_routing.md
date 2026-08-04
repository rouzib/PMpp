# CUDA routing

CUDA routing is an optional shared-library extension that accelerates the
shard-local parts of particle migration in multi-GPU
`MultiGPUConfiguration(mode="mesh_halo")` runs. It replaces the local route
pack and stable route merge with CUDA FFI handlers. Capacity checks, neighbor
exchange, and collective reversal remain in JAX, including the `lax.ppermute`
communication between shards.

It is an acceleration, not a different simulation algorithm. PM++ preserves
the canonical particle ordering and uses its existing N-body custom adjoint to
reconstruct the routing plan during the reverse sweep.

## Why use it

The route-pack and route-merge work appears in every particle migration step.
Moving those shard-local operations to CUDA reduced the measured full AD time
substantially, while leaving the distributed communication and scientific
contract unchanged. In the `512^3`, 63-step float32 benchmark on four H100
80 GiB GPUs, CUDA routing changed full forward time from 4.792 s to 4.707 s
and full AD time from 28.955 s to 22.005 s, with the same reported peak memory
(6.076 GiB/GPU forward and 11.613 GiB/GPU AD). See
[Optimizations](optimizations.md) for the complete comparison.

The gain is topology- and problem-dependent, so measure a production-sized
case on your own system. CUDA routing is most relevant for qualified multi-GPU
`mesh_halo` runs, especially when reverse-mode AD is part of the workload.

## Requirements

The extension is intentionally outside the normal Python wheel. To build and
select it, you need:

- a Linux CUDA development environment with `nvcc`, CUDA runtime headers
  (including CUB), and CMake 3.24 or newer;
- the same Python environment and CUDA-enabled JAX installation that will run
  PM++; and
- a qualified runtime: JAX 0.6.x, GPU backend, float32 simulation fields,
  `int16` or `int32` particle IDs, at least two devices, and `mesh_halo` mode.

The supplied CMake target builds for Ampere SM 8.0/8.6 and embeds SM 8.6 PTX
for forward-compatible compilation. Validate another GPU architecture with the
portable path before adopting the extension for production.

## Build and select the extension

Run the builder from the repository root with the exact Python environment that
will execute PM++:

```bash
python scripts/build_cuda_routing.py
export PMPP_CUDA_ROUTING_LIBRARY="$PWD/cuda/build/libpmpp_cuda_routing.so"
```

The builder configures CMake with that Python executable, obtains the matching
`jaxlib` headers, and prints the resulting library path. The source checkout
also looks in `cuda/build/` automatically, but setting
`PMPP_CUDA_ROUTING_LIBRARY` is useful when the library is built elsewhere or
when an installed package is used.

Request automatic selection in the normal configuration by leaving
`cuda_routing` unset (its default), or request it explicitly after building:

```python
from pmpp.multigpu_configuration import MultiGPUConfiguration

multigpu = MultiGPUConfiguration(
    compute_mesh=compute_mesh,
    mode="mesh_halo",
    cuda_routing=True,
)
```

After configuration construction, `conf.cuda_routing` reports the resolved
choice. The following non-throwing helper reports whether the library was
found and whether the JAX version/backend are qualified:

```python
from pmpp.cuda_routing import extension_status

print(extension_status())
```

For an intentional portable comparison, use `cuda_routing=False` in
`MultiGPUConfiguration`, or set `PMPP_CUDA_ROUTING=0` before PM++ is imported.

## If it is not installed or cannot be used

Nothing needs to be rebuilt to run without CUDA routing. A normal `pip install`
does not invoke CMake or require `nvcc`. If the shared library is missing,
stale, incompatible, disabled, or the configuration does not meet the runtime
requirements, PM++ automatically retains the packed portable JAX router.

That fallback still performs the same capacity checks, routing, collectives,
and custom-adjoint calculation. Its cost is performance rather than a change
in the intended numerical method. Keep the fallback available for CPU runs,
single-GPU runs, non-float32 work, platform qualification, and controlled
regression tests.

For the handler ABI, packed record format, and the boundary between the FFI and
the adjoint, see [Optional CUDA routing](../internals/cuda_routing.md).
