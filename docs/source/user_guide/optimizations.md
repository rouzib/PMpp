# Optimizations

PM++ keeps the differentiable solver as its standard implementation. It also
provides an opt-in, forward-only low-memory path for configurations that meet a
narrower runtime contract. Choose between them from the scientific and memory
requirements of the run rather than assuming that one path is always faster.

## Recommended configurations

For a full forward simulation, use:

```python
import jax

from pmpp import Configuration
from pmpp import MultiGPUConfiguration
from pmpp.distributed import create_compute_mesh

gpu_devices = jax.devices("gpu")
n = 64

conf = Configuration(
    ptcl_spacing=1.0,
    ptcl_grid_shape=(n, n, n),
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices),
        mode="mesh_halo",
        cuda_routing=True,
    ),
    pallas_cic=True,
    lpt_cache_strains=True,
    nbody_cosmo_grad=True,
)
```

For a full AD run, use the same configuration. Keep
`nbody_cosmo_grad=True` whenever derivatives with respect to cosmological
parameters are part of the objective. Set it to `False` only for an explicitly
displacement-only objective.

```{note}
CUDA routing is much faster and should be used whenever the compiled extension
is available. Set `cuda_routing=True`; the optimized `bidir_mergepath` backend
is now selected by default. PM++ uses the portable JAX router if the selected
backend, extension, or runtime is unavailable. See
[CUDA routing](cuda_routing.md) for installation and status checks.
```

## Optimization flags

| Flag | Accepted values | Guidance |
| --- | --- | --- |
| `MultiGPUConfiguration.mode` | `"mesh_halo"`, `"particle_halo"`, or `None` | Use `"mesh_halo"`. `"particle_halo"` enables the compatibility path. `None` uses the legacy `Configuration.multigpu_mode` value. |
| `MultiGPUConfiguration.cuda_routing` | `True`, `False`, or `None` | Use `True` when the compiled extension is available. `False` or `None` uses the portable JAX router. An unavailable requested extension also falls back to JAX. |
| `MultiGPUConfiguration.cuda_routing_backend` | `"bidir_mergepath"` or `"cuda_merge"` | Defaults to `"bidir_mergepath"`, the recommended and faster native route. Use `"cuda_merge"` only for legacy comparison or backend-specific validation. The flag matters only when `cuda_routing=True`. |
| `Configuration.pallas_cic` | `True` or `False` | `True` selects Pallas CIC and is the default. Unsupported configurations fall back to reference JAX CIC. `False` selects the reference implementation explicitly. |
| `Configuration.lpt_cache_strains` | `True` or `False` | `True` caches LPT strain arrays and is the default. `False` recomputes them to reduce retained memory. |
| `Configuration.nbody_cosmo_grad` | `True` or `False` | `True` includes cosmology cotangents and is the default. `False` omits them for objectives that need only particle or mode gradients. |
| `Configuration.chunk_size` | Any positive integer | Controls the JAX fallback chunk size. The default is `2**24`. |

See [Pallas CIC kernels](pallas_cic.md) for the CIC backend requirements.

## Forward-only low-memory path

The low-memory path is exposed through separate entry points. Ordinary
`lpt()` and `nbody()` keep their existing differentiable behavior.

```python
def run_low_memory_forward(modes, cosmo, conf):
    from pmpp.initial_conditions import lpt_low_memory_with_telemetry
    from pmpp.nbody import nbody_low_memory_with_telemetry

    particles, lpt_moved, lpt_invalid = lpt_low_memory_with_telemetry(
        modes, cosmo, conf
    )
    particles, max_occupancy, nbody_moved, nbody_invalid = (
        nbody_low_memory_with_telemetry(particles, cosmo, conf)
    )
    telemetry = {
        "lpt_moved": lpt_moved,
        "lpt_invalid": lpt_invalid,
        "max_occupancy": max_occupancy,
        "nbody_moved": nbody_moved,
        "nbody_invalid": nbody_invalid,
    }
    return particles, telemetry
```

The shorter `lpt_low_memory()` and `nbody_low_memory()` entry points return
only the particle state. The telemetry variants are recommended for large
runs because static capacities must be selected from measured high-water
marks. The `modes` input to low-memory LPT and the particle-state buffers
passed to low-memory N-body are donated. Do not reuse them after the call.

### Supported configuration

The currently qualified distributed profile is:

| Property | Required value |
| --- | --- |
| Execution | Forward only, outside an enclosing `jax.jit`, `jax.grad`, or other JAX transformation |
| Dimensions and LPT | Three dimensions and `lpt_order=2` |
| Particle-to-mesh ratio | Constructor `mesh_shape=1`, so the force mesh equals the particle grid |
| Dtypes | `float_dtype=jax.numpy.float32` and `pmid_dtype=jax.numpy.int16` |
| Multi-GPU mode | `MultiGPUConfiguration(mode="mesh_halo")` |
| Native routing | `cuda_routing=True` with `cuda_routing_backend="bidir_mergepath"` |
| Corrections and direction | `correction=None` and `reverse=False` |
| Per-device indexing | Fewer than `2**31` local particles and particle-capacity slots |

The low-memory LPT validator enforces these shape and dtype restrictions. The
distributed N-body implementation additionally requires the fused float32 and
int16 primal route from CUDA routing record format v3. It does not silently use
the portable router when that target is unavailable.

Build the routing extension in the active JAX environment, then fail closed on
its status before allocating a large state:

```python
def require_low_memory_route(conf):
    from pmpp.distributed import extension_status

    status = extension_status()
    if not (
        conf.cuda_routing
        and conf.cuda_routing_backend == "bidir_mergepath"
        and status["record_format_version"] == 3
        and status["fused_primal_feature"]
        and status["fused_primal_registered"]
    ):
        raise RuntimeError("the ABI-v3 fused low-memory route is unavailable")
```

Do not use this profile with a non-unit particle-to-mesh ratio merely because a
small case happens to run. That configuration is not qualified. Use the
standard `lpt()` and `nbody()` path instead.

### What reduces memory

Low-memory 2LPT creates the particle grid from a local row index, builds the
second-order source sequentially, and consumes displacement components one at
a time. It does not retain the standard cached strain set, regardless of
`lpt_cache_strains`.

Low-memory N-body computes one gravity component at a time with a scalar
distributed inverse FFT. Its native drift route computes the drifted position
and ownership in CUDA and avoids particle-sized position, classification, key,
and stay-index arrays. Compatible input and output buffers are donated for
reuse.

These changes reduce live JAX buffers, but they do not guarantee a particular
resident-memory saving. FFT workspaces, communication libraries, allocator
reservation, particle capacity, and device count remain part of the peak.
Inspect compiled memory analysis and measure every device in a fresh process.

### Capacity and routing limits

Set `max_ptcl_per_slice`, `max_share_ptcl`, and the gather capacities
explicitly for a production run. Start from a conservative capacity, record
the telemetry high-water values, and retain a safety margin. Never lower a
capacity merely to make a run fit. Any nonzero invalid count or capacity
failure invalidates the result.

Particle exchange currently supports the same ownership slab or an immediate
left or right neighbor in one route. A failure such as
`particles_outside_neighbor_range` means the displacement crossed more than
one slab. Increasing HBM, the BFC pool, or a share capacity cannot fix it.
Smaller N-body steps can reduce a per-step drift, but they do not fix an LPT
state whose total initial displacement already exceeds the one-hop contract.

### Allocator and performance

Set allocator variables before importing JAX and compare policies in separate
processes. A bounded BFC example is:

```bash
unset XLA_PYTHON_CLIENT_ALLOCATOR
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

With BFC, `nvidia-smi` reports the reserved pool, not only live arrays. A nearly
full resident reading is therefore not itself an out-of-memory diagnosis.
`cuda_malloc_async` has different reservation behavior and does not use the
BFC memory-fraction setting.

Sequential component FFTs may cost more launch and collective time than the
standard batched gravity calculation, while fused routing can recover some of
that cost. Benchmark the complete LPT, N-body, and final scatter stages. Do not
infer full-run speed from a routing or FFT microbenchmark, and do not claim an
AD result from this forward-only profile.
