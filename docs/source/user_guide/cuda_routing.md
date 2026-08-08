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

The PM++ wheel contains the CUDA source and build command, but not a precompiled
shared library. Compiling the extension requires:

- Python 3.10 or newer
- JAX and `jaxlib` 0.9.1 or newer with CUDA support
- a CUDA toolkit supported by the active JAX CUDA build, with the matching
  `nvcc`, CUDA runtime headers, and CUB
- CMake 3.24 or newer
- `nvidia-smi` for automatic architecture detection, or an explicit
  `--cuda-architectures` list

The CUDA toolkit is a development dependency and is separate from the CUDA
runtime libraries installed with JAX. The build command reports a clear error
if `cmake`, `nvcc`, JAX, or `jaxlib` is missing. At least two GPUs are required
to use and validate multi-GPU routing, although compilation itself does not
depend on the number of GPUs.

Do not select `nvcc` independently of JAX. For example, a CUDA 12 JAX build
needs a compatible CUDA 12 toolkit, while a CUDA 13 JAX build needs a
compatible CUDA 13 toolkit. See
[CUDA routing build uses an incompatible `nvcc`](troubleshooting.md#cuda-routing-build-uses-an-incompatible-nvcc)
for checks and cluster-module guidance.

CUDA routing supports float32 and float64 simulation fields, `int16` or
`int32` particle mesh indices (`pmid`), and `mesh_halo` mode. Its targets are
specialized by payload and coordinate dtype. The forward-only fused drift route
used by the low-memory solver specifically requires float32 fields, int16
coordinates, and the `bidir_mergepath` backend.

## Install from PyPI and build CUDA routing

Install PM++ first, then run the builder in the same Python environment before
using PM++ for the first time:

```bash
python -m pip install pmpp
pmpp-build-cuda-routing
```

`bidir_mergepath` is the default CUDA routing backend, so no environment
variable is needed for the recommended path.

`python -m pmpp.distributed.build_cuda` is equivalent to
`pmpp-build-cuda-routing` and can be used when the console-script directory is
not on `PATH`.

The command detects every visible GPU compute capability with `nvidia-smi` and
builds native code for each distinct architecture. It also embeds PTX for the
newest detected architecture. If detection is unavailable, it falls back to
the PM++ compatibility set `80;86;90;90-virtual`. An explicit list can be
provided for a login node, cross-machine build, or a GPU hidden by the job
scheduler:

```bash
pmpp-build-cuda-routing --cuda-architectures "80;86;90;90-virtual"
```

The builder configures CMake with the active Python executable and obtains the
matching FFI headers from that environment's `jaxlib`. It compiles in a
temporary build directory, then installs only
`libpmpp_cuda_routing.so` and its ABI manifest under
`pmpp/distributed/_cuda`. PM++ finds that package-local artifact automatically.

Some system installations expose a read-only `site-packages` directory. In
that case, the command installs the artifact in a versioned user cache under
`$XDG_CACHE_HOME/pmpp` or `~/.cache/pmpp`; the PM++ loader searches the same
location automatically. Set `PMPP_CUDA_ROUTING_CACHE` to choose a different
cache directory. Re-running the command skips compilation when the installed
manifest already matches the PM++ version, active `jaxlib`, and detected
architectures. Use `--force` to rebuild it.

### Build from a source checkout

Developers can still call the lower-level builder directly. The target below
places the resulting artifact in the importable package directory:

```bash
python cuda/build_cuda_routing.py \
  --build-dir "$(python -c 'from pathlib import Path; import pmpp.distributed; print(Path(pmpp.distributed.__file__).resolve().parent / "_cuda")')"
```

The installed command is preferred for PyPI users because it selects the
active interpreter, detects GPU architectures, handles read-only package
directories, and avoids leaving CMake files beside the installed artifact.

CUDA routing is optional. Leaving `cuda_routing` unset keeps the portable JAX
route. Request the extension explicitly after building it:

```python
import jax

from pmpp import MultiGPUConfiguration
from pmpp.distributed import create_compute_mesh

gpu_devices = jax.devices("gpu")

multigpu = MultiGPUConfiguration(
    compute_mesh=create_compute_mesh(gpu_devices),
    mode="mesh_halo",
    cuda_routing=True,
)
```

The `cuda_routing_backend` flag defaults to `"bidir_mergepath"`. Select the
older native implementation only for comparison or targeted validation:

```python
legacy_multigpu = multigpu.replace(cuda_routing_backend="cuda_merge")
```

`PMPP_CUDA_ROUTING_BACKEND` remains available as a process-wide override. It
accepts `bidir_mergepath` or `cuda_merge`; the historical value `current` is an
alias for `cuda_merge`. The environment variable takes precedence over the
configuration flag.

The following reproducible check reports the installed artifact and runtime
status without printing the full build manifest:

```python
import json

from pmpp.distributed.cuda import extension_status

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
`conf.multigpu.cuda_routing` reports whether PM++ selected the extension, and
`conf.multigpu.cuda_routing_backend` reports the resolved native backend.

For an intentional portable comparison, leave `cuda_routing` unset, use
`cuda_routing=False` in `MultiGPUConfiguration`, or set `PMPP_CUDA_ROUTING=0`
before PM++ is imported.

## If it is not installed or cannot be used

Nothing needs to be compiled to run without CUDA routing. A normal `pip install`
does not invoke CMake or require `nvcc`. If the shared library is missing,
stale, incompatible, disabled, or the configuration does not meet the runtime
requirements, PM++ automatically retains the packed portable JAX router.

That fallback still performs the same capacity checks, routing, collectives,
and custom-adjoint calculation. Its cost is performance rather than a change
in the intended numerical method. Keep the fallback available for CPU runs,
single-GPU runs, unsupported floating dtypes, platform qualification, and
controlled regression tests.

For the handler ABI, packed record format, and the boundary between the FFI and
the adjoint, see [Optional CUDA routing](../internals/cuda_routing.md).
