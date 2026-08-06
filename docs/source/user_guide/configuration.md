# Configuration

`pmpp.core.Configuration` is a frozen dataclass and a static JAX
PyTree. It combines the simulation definition with the shapes and
callables needed by the runtime. Treat a configuration as immutable; use
`conf.replace(...)` to make a related configuration.

## Box, particles, and force mesh

```python
import jax
import jax.numpy as jnp

from pmpp import Configuration
from pmpp import MultiGPUConfiguration
from pmpp.distributed import create_compute_mesh

n = 64
box_size = 250.0
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ example requires at least two GPUs")

conf = Configuration(
    ptcl_spacing=box_size / n,
    ptcl_grid_shape=(n, n, n),
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices),
        mode="mesh_halo",
    ),
    float_dtype=jnp.float32,
)
```

For an isotropic grid,

$$
L = N_p\,\Delta q, \qquad
N_m = r N_p, \qquad
\Delta x = \frac{L}{N_m},
$$

where `ptcl_spacing` is $\Delta q$, `ptcl_grid_shape` contains $N_p$,
`mesh_shape=r` may be a scalar ratio, and `cell_size` is $\Delta x$.
`mesh_shape` cannot make the mesh smaller than the particle grid and the grid
aspect ratios must agree. A ratio of `1` is the recommended starting point.

Three decomposition divisibility rules apply with $D$ devices:

- `ptcl_grid_shape[0] % D == 0`, because particle generation assigns equal
  x-slabs;
- `mesh_shape[0] % D == 0`, because the real mesh is x-sharded;
- `mesh_shape[1] % D == 0`, because the distributed FFT transposes to a
  y-sharded spectral layout.

Configuration construction does not currently reject every incompatible shape
up front, so check these rules explicitly rather than relying on integer
division or a later collective failure.

Choose the smallest particle and mesh resolution that suits the simulation.
The committed simulation notebooks use $256^3$ particles; smaller
grids remain useful for private debugging and smoke tests, not evidence of
production accuracy.
Increasing the force mesh raises FFT memory/traffic and can increase boundary
buffer pressure; validate each resolution before scaling further.

## Units and precision

The default length unit is $h^{-1}\mathrm{Mpc}$, the default mass unit is
$10^{10}h^{-1}M_\odot$, and the default time unit is $H_0^{-1}$. Consequently,
a numerical `box_size=250.0` denotes $250\,h^{-1}\mathrm{Mpc}$ unless `L`, `M`,
or `T` is changed.

PM++ separates three dtypes:

- `cosmo_dtype` for cosmology, growth, and configuration quantities;
- `float_dtype` for particle and mesh fields;
- `pmid_dtype` for signed integer particle/mesh indices.

The default is mixed precision: `cosmo_dtype=float64` for cosmology/configuration
tables and `float_dtype=float32` for particle and mesh work. In the current
dependency stack, importing the `mcfit`-backed configuration module may also
enable JAX x64 globally; record `jax.config.jax_enable_x64` instead of assuming
the process default. Set both float dtypes explicitly when an experiment needs
a different precision policy, and construct the configuration only after the
JAX precision setting is final. `pmid_dtype` must represent every mesh index
safely; do not use `int16` for an axis that can exceed its range.

For the distinction between higher numerical precision, bitwise repeatability,
and gradient accuracy, including the runtime and memory costs of float64 and
deterministic GPU operations, see
[Precision, reproducibility, and gradients](precision_reproducibility.md).

## Time schedule and LPT order

`a_start`, `a_stop`, and `a_nbody_maxstep` define a linearly spaced default
scale-factor schedule. Supply `a_custom` for explicit boundaries. `a_custom`
becomes static tuple data, so changing its values or length recompiles the
solver.

Operational LPT choices are:

| `lpt_order` | Meaning | Status |
|---:|---|---|
| 0 | unperturbed grid | supported |
| 1 | Zel'dovich approximation | supported |
| 2 | second-order LPT | supported, default |

`lpt_cache_strains=True` avoids repeated inverse FFTs during 2LPT at the cost of
keeping diagonal strain arrays live. Set it to `False` only when that memory
trade-off is preferable.

## Static and dynamic values under JAX

The whole `Configuration` is static metadata. Grid shapes, schedules, dtypes,
device topology, correction structure, and capacity values therefore affect
the compilation cache. A different configuration normally means a different
compiled program.

Keep configuration construction outside jitted functions, reuse an object
across repeated calls, and benchmark only after calling
`jax.block_until_ready` on a warm result. Cosmological parameter arrays are
dynamic leaves of `Cosmology`. Use the helpers in `pmpp.cosmology` when selecting
and replacing differentiable parameters.

## Device mesh and capacities

Use the nested runtime form in new code. The following is a shape template,
not a standalone example; replace every named placeholder with values validated
for the intended run:

```text
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ example requires at least two GPUs")

conf = Configuration(
    ptcl_spacing=<particle spacing>,
    ptcl_grid_shape=<particle shape>,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices),
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=<particle slots per device>,
    max_share_ptcl=<migration slots>,
    max_halo_share_ptcl=<particle-halo exchange slots>,
    max_share_gather_ptcl=<gather exchange slots>,
)
```

The older top-level `compute_mesh=` and `multigpu_mode=` arguments remain
compatibility inputs, not the preferred construction. The four capacities mean:

- `max_ptcl_per_slice`: total padded particle slots on each device;
- `max_share_ptcl`: particles that can migrate to one neighbor in a step;
- `max_halo_share_ptcl`: particle-halo rebuild exchange (primarily the legacy
  particle-halo path);
- `max_share_gather_ptcl`: gathered particle-value exchange used by compatible
  particle-halo paths.

Runtime construction clips migration and gather capacities to at most half the
per-slice slots, and clips the halo capacity to the per-slice slots. Inspect and
record the resulting `conf.max_*` values after construction; they are the
effective compiled capacities, which may be smaller than oversized constructor
arguments.

An overflow invokes a host callback that raises a runtime error instead of
silently accepting a truncated fixed-size operation. The run must still be
discarded and repeated with a larger named capacity. See
[Multi-GPU execution](multigpu.md) and [Troubleshooting](troubleshooting.md).

### CIC execution flag

`Configuration.pallas_cic=True` selects the paired Pallas gather/scatter CIC
implementation on qualified float32 GPU backends. It is enabled by default.
Unsupported configurations emit a warning and use the portable reference JAX
implementation. Mesh-halo routing and multi-channel force gather use their
canonical packed/fused implementations unconditionally. See
[Pallas CIC kernels](pallas_cic.md) for operation and qualification, and
[Optimizations](optimizations.md) for measured recommendations.

### CUDA routing flag

`MultiGPUConfiguration(cuda_routing=True)` requests the optional CUDA FFI
route-pack and route-merge implementation. CUDA routing remains opt-in, while
`cuda_routing_backend` defaults to the recommended `"bidir_mergepath"` native
implementation. Set it to `"cuda_merge"` only for legacy comparison. If the
selected backend, extension, or runtime is not qualified, PM++ keeps the
portable JAX router.

```python
cuda_conf = conf.replace(
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices),
        mode="mesh_halo",
        cuda_routing=True,
    )
)

print("CUDA routing enabled:", cuda_conf.multigpu.cuda_routing)
print("CUDA routing backend:", cuda_conf.multigpu.cuda_routing_backend)
```

See [CUDA routing](cuda_routing.md) for compiler requirements, installation,
status checks, and measured performance.

## Record the configuration

At minimum, persist box size, particle and mesh shapes, all three dtypes,
cosmology, seed/noise scheme, LPT order, the complete scale-factor schedule,
device count, multi-GPU mode, and all capacity values. Store the PM++ commit and
JAX version alongside them. Static behavior is not recoverable from a final
density field alone.

See the [configuration API](../api/core.rst) and the configuration
notebook in the [gallery](../notebooks/index.md).
