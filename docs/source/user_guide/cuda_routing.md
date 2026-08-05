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
The bidirectional CUDA implementation is typically about twice as fast as the
legacy CUDA or portable JAX implementation for this routing stage. The effect
on the complete simulation depends on how much time it spends moving
particles. CUDA routing is most useful for multi-GPU `mesh_halo` runs,
especially when reverse-mode AD is part of the workload.

## Requirements

The extension is intentionally outside the normal Python wheel. The minimum
requirements are:

- Python 3.10 or newer
- JAX and `jaxlib` 0.6.x with CUDA support
- the CUDA toolkit with `nvcc`, CUDA runtime headers, and CUB
- CMake 3.24 or newer
- at least two GPUs

CUDA routing supports float32 simulation fields, `int16` or `int32` particle
IDs, and `mesh_halo` mode.

## Build and select the extension

Run the builder from a PM++ source checkout with the Python environment that
will execute PM++. This command places the extension in the installed `pmpp`
package's `_cuda` directory, where PM++ finds it automatically:

```bash
python cuda/build_cuda_routing.py \
  --build-dir "$(python -c 'from pathlib import Path; import pmpp; print(Path(pmpp.__file__).resolve().parent / "_cuda")')"
export PMPP_CUDA_ROUTING_BACKEND=bidir_mergepath
```

The builder configures CMake with that Python executable, obtains the matching
`jaxlib` headers, and prints the resulting library and manifest paths. The CUDA
source is distributed with the source package, not the normal wheel, so the
build command must be run from a source checkout.

CUDA routing is optional. Leaving `cuda_routing` unset keeps the portable JAX
route. Request the extension explicitly after building it:

```python
import jax

from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

gpu_devices = jax.devices("gpu")

multigpu = MultiGPUConfiguration(
    compute_mesh=create_compute_mesh(gpu_devices),
    mode="mesh_halo",
    cuda_routing=True,
)
```

The following reproducible check reports the installed artifact and runtime
status without printing the full build manifest:

```python
import json

from pmpp.cuda_routing import extension_status

status = extension_status()
summary = {
    key: status[key]
    for key in (
        "qualified_jax",
        "jax_version",
        "backend",
        "library",
        "bidir_targets",
        "build_identifier",
        "record_format_version",
        "embedded_architectures",
    )
}
print(json.dumps(summary, indent=2))
```

`qualified_jax` should be `true`, `backend` should be `gpu`, `library` should
contain the installed shared-library path, and `bidir_targets` should list both
bidirectional handlers. After a compatible `Configuration` is constructed,
`conf.mGPU.cuda_routing` reports whether PM++ selected the extension.

For an intentional portable comparison, leave `cuda_routing` unset, use
`cuda_routing=False` in `MultiGPUConfiguration`, or set `PMPP_CUDA_ROUTING=0`
before PM++ is imported.

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
