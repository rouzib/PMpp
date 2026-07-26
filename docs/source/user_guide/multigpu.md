# Multi-GPU execution

PM++ decomposes the periodic domain into equal slabs along the global x axis.
The preferred `mesh_halo` mode keeps one authoritative particle record and
exchanges mesh edge cells for local CIC operations.

## Construct the device mesh

```python
import jax
import jax.numpy as jnp

from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

devices = [d for d in jax.devices() if d.platform == "gpu"]
if len(devices) < 2:
    raise RuntimeError("This run requires at least two GPUs")
selected_devices = devices[:2]

n = 256
compute_mesh = create_compute_mesh(selected_devices)
conf = Configuration(
    ptcl_spacing=1000.0 / n,
    ptcl_grid_shape=(n,) * 3,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=compute_mesh,
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=int((n**3 / len(selected_devices)) * 1.20),
    max_share_ptcl=50_000,
    max_halo_share_ptcl=50_000,
    max_share_gather_ptcl=200_000,
    float_dtype=jnp.float32,
)
```

Device order determines slab order. For global mesh x size $N_x$ and $D$
devices, device $d$ owns

$$
[dN_x/D,\,(d+1)N_x/D).
$$

Require `ptcl_grid_shape[0] % D == 0`, `mesh_shape[0] % D == 0`, and
`mesh_shape[1] % D == 0`. The first rule prevents particle-grid generation
from dropping a remainder, the second creates equal real-space x slabs, and
the third supports the y-sharded spectral layout after the distributed FFT
transpose. Inspect `conf.owned_slice_start`,
`conf.owned_slice_end`, and `conf.local_mesh_shape` rather than inferring a
layout from physical device IDs. Stored slice endpoints are periodic: an end
value of `0` on the final slab represents the wrapped boundary at $N_x$, not an
empty interval. The examples select two devices explicitly; the formulas retain
$D$ so the same constraints can be applied to larger meshes.

## Why `mesh_halo`

In `mesh_halo` mode:

- each valid particle is authoritative on the slab containing its position;
- a drift can transfer ownership, so particles still migrate;
- scatter writes into an owned mesh plus edge halos and reduces halo
  contributions to the neighbor's owned slab;
- gather exchanges owned edge cells and interpolates from the extended local
  mesh;
- custom VJPs reverse those exchanges for gradients.

`particle_halo` retains duplicated particle records and exists for compatibility
and validation. It is not the recommended production mode.

## Static capacity planning

Every device allocates fixed-size particle and communication arrays so JAX can
compile stable shapes. A reasonable initial particle-storage capacity is

$$
C_\mathrm{slice}=\left\lceil f\,N_p/D\right\rceil,
$$

where $f>1$ allows clustering imbalance. The correct factor is experiment
dependent; monitor actual occupancy and migration over representative seeds and
the full schedule. The four capacity fields are described in
[Configuration](configuration.md).

Any printed `Exceeded ... capacity` message means a fixed-size compact/exchange
operation was truncated. Stop, increase the named capacity, recompile, and
rerun from the initial conditions. Do not compare or differentiate the partial
result.

## Distributed FFTs

The real density begins in x slabs. A distributed rFFT performs local transforms
and a collective transpose so a later axis is local; the spectral array is
therefore in a transposed sharded layout. Use PM++'s `conf.mGPU_*fftn*` helpers
inside solver extensions rather than applying a local FFT independently to each
slab.

## Operational checklist

- expose only the intended GPUs before Python starts;
- set `XLA_PYTHON_CLIENT_PREALLOCATE=false` when appropriate;
- run one heavy multi-GPU process at a time;
- create a fresh process after changing imported code;
- record device order/model and every capacity;
- treat all overflow/error prints as failed runs;
- warm up and block results before timing;
- check density mean/mass and boundary-visible projections.

## Cluster and Narval jobs

Request GPU nodes through the scheduler and verify `jax.devices()` inside the
allocated job, never from the login node. Load a CUDA/JAX-compatible module or
environment, activate the project environment, and print the visible devices,
PM++ commit, scheduler resources, mesh/device layout, seed, and capacities into
the run log before starting the simulation.

Narval module names, account names, scratch paths, and allocation policy change
with the site and research group. Keep those details in a private runbook or job
template; the portable contract is the same: the allocated GPUs must be visible
to the CUDA-enabled JAX build, particle x and mesh x/y dimensions must divide
their count, and a small `mesh_halo` smoke run must pass before a production
launch.

For the communication design and diagrams, read
[Distributed runtime](../internals/distributed_runtime.md).
