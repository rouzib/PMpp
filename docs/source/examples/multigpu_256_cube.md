# Multi-GPU 256 cube

This example shows the configuration pattern for a practical `mesh_halo` run. It requires at least two CUDA-visible GPUs.

```python
import jax
import jax.numpy as jnp

from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

res = 256
gpus = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpus) < 2:
    raise SystemExit("requires at least two GPUs")

conf = Configuration(
    1000.0 / res,
    (res, res, res),
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(create_compute_mesh(gpus), mode="mesh_halo"),
    max_ptcl_per_slice=int((res**3 / len(gpus)) * 1.8),
    max_share_ptcl=50_000,
    max_halo_share_ptcl=50_000,
    max_share_gather_ptcl=200_000,
    float_dtype=jnp.float32,
)
print("devices", len(gpus))
print("mode", conf.multigpu.mode)
```

Use environment variables such as `CUDA_VISIBLE_DEVICES=0,1,2,3` to choose devices, but do not hard-code a local machine layout in reusable scripts.
