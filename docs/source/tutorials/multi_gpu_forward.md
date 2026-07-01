# Multi-GPU forward run

Multi-GPU runs require multiple visible JAX GPU devices. Do not run this on CPU-only Read the Docs builders.

```python
import jax
from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

gpus = [d for d in jax.devices() if d.platform == "gpu"]
if len(gpus) < 2:
    raise RuntimeError("multi-GPU PM++ run requires at least two GPUs")

conf = Configuration(
    1.0,
    (128, 128, 128),
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpus),
        mode="mesh_halo",
    ),
)
```
