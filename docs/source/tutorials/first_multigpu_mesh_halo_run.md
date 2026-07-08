# First multi-GPU `mesh_halo` run

## What you will do

Configure PM++ to run on multiple GPUs using the preferred `mesh_halo` mode.

## Requirements

- At least two visible GPU devices.
- JAX installed with CUDA support.
- PM++ installed in editable or normal mode.

## Complete code

```python
import jax
import jax.numpy as jnp

from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

res = 256
box_size = 1000.0
ptcl_grid_shape = (res, res, res)
ptcl_spacing = box_size / res

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This tutorial requires at least two GPUs.")

compute_mesh = create_compute_mesh(gpu_devices)
num_devices = len(gpu_devices)

conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=compute_mesh,
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=int((res**3 / num_devices) * 1.8),
    max_share_ptcl=50_000,
    max_halo_share_ptcl=50_000,
    max_share_gather_ptcl=200_000,
    float_dtype=jnp.float32,
)

print("GPU devices:", num_devices)
print("Particle grid:", ptcl_grid_shape)
```

## Step-by-step explanation

`mesh_halo` keeps particles authoritatively stored on their owning x-slab and exchanges mesh halo cells before gather operations. This is the current preferred multi-GPU path for serious PM++ runs.

## Expected output

The script should print at least two GPU devices and the configured particle grid. A complete forward pass follows the same modes, LPT, `nbody`, and `scatter` pattern as the serial tutorial.

## Common failures

- CPU-only machines should raise the explicit device-count error.
- If `mesh_shape[0]` or the x-resolution is not divisible by the number of devices, choose a compatible shape unless the code path explicitly supports uneven slabs.
- Capacity overflows mean a static buffer was too small; increase the named capacity and rerun.

## Next steps

Read [static-capacity buffers](../introduction/static_capacity_buffers.md) and [capacity debugging](../how_to/debug_capacity_overflow.md).
