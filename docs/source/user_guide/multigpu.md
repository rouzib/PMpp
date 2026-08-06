# Multi-GPU execution

PM++ decomposes the periodic domain into equal slabs along the global x axis.
The preferred `mesh_halo` mode keeps one authoritative particle record and
exchanges mesh edge cells for local CIC operations.

## Construct the device mesh

```python
import jax
import jax.numpy as jnp

from pmpp import Configuration
from pmpp import MultiGPUConfiguration
from pmpp.distributed import create_compute_mesh

devices = [d for d in jax.devices() if d.platform == "gpu"]
if len(devices) < 2:
    raise RuntimeError("This run requires at least two GPUs")

n = 256
compute_mesh = create_compute_mesh(devices)
conf = Configuration(
    ptcl_spacing=1000.0 / n,
    ptcl_grid_shape=(n,) * 3,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=compute_mesh,
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=int((n**3 / len(devices)) * 1.20),
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
transpose. Inspect `conf.multigpu.owned_slice_start`,
`conf.multigpu.owned_slice_end`, and `conf.multigpu.local_mesh_shape` rather
than inferring a layout from physical device IDs. Stored slice endpoints are
periodic. An end value of `0` on the final slab represents the wrapped boundary
at $N_x$, not an empty interval. The examples use every visible GPU, and the
formulas retain $D$ so the same constraints apply to any supported device mesh.

## Why `mesh_halo`

`particle_halo` and `mesh_halo` differ in how they provide the neighboring data
needed by CIC scatter and gather near a slab boundary.

In `particle_halo` mode, each GPU stores the particles owned by its slab and
duplicate copies of particles near neighboring slab boundaries. Those halo
particles must be exchanged and rebuilt as particles move. This increases
particle storage and makes boundary work depend on the number of particles in
the halo region.

In `mesh_halo` mode, every physical particle has one authoritative record on
the GPU that owns its current slab. Particles still migrate when they cross a
slab boundary. For scatter and gather, PM++ exchanges narrow edge regions of
the mesh instead of keeping duplicate particle records. Scatter contributions
in mesh halos are reduced into the neighboring owned slab, while gather extends
the local mesh with neighboring edge cells. The custom VJPs reverse these
exchanges when gradients are computed.

`mesh_halo` avoids duplicated particle storage and usually reduces particle
routing and boundary bookkeeping. It is the preferred mode and is typically
faster than `particle_halo` for both smaller and larger simulation boxes.

## Static capacity planning

Every device allocates fixed-size particle and communication arrays so JAX can
compile stable shapes. A reasonable initial particle-storage capacity is

$$
C_\mathrm{slice}=\left\lceil f\,N_p/D\right\rceil,
$$

where $f>1$ allows clustering imbalance. The correct factor is experiment
dependent. Monitor actual occupancy and migration over representative seeds and
the full schedule. The four capacity fields are described in
[Configuration](configuration.md).

Any printed `Exceeded ... capacity` message means a fixed-size compact/exchange
operation was truncated. Stop, increase the named capacity, recompile, and
rerun from the initial conditions. Do not compare or differentiate the partial
result.

## Distributed FFTs

The real density begins in x slabs. A distributed rFFT performs local transforms
and a collective transpose so a later axis is local. The spectral array is
therefore in a transposed sharded layout. Use the FFT helpers owned by
`conf.multigpu`, such as `rfftn_transposed` and `irfftn_transposed`, inside
solver extensions rather than applying a local FFT independently to each slab.

For the communication design and diagrams, read
[Distributed runtime](../internals/distributed_runtime.md).
