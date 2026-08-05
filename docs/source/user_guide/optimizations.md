# Optimizations

PM++ keeps one optimized implementation for its core multi-GPU path. It is the
best implementation tested, providing the fastest execution and the lowest
memory use.

## Recommended configurations

For a full forward simulation, use:

```python
import jax

from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

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
is available. Set `cuda_routing=True`. PM++ uses the portable JAX router if the
extension or runtime is unavailable. See [CUDA routing](cuda_routing.md) for
installation and status checks.
```

## Optimization flags

| Flag | Accepted values | Guidance |
| --- | --- | --- |
| `MultiGPUConfiguration.mode` | `"mesh_halo"`, `"particle_halo"`, or `None` | Use `"mesh_halo"`. `"particle_halo"` enables the compatibility path. `None` uses the legacy `Configuration.multigpu_mode` value. |
| `MultiGPUConfiguration.cuda_routing` | `True`, `False`, or `None` | Use `True` when the compiled extension is available. `False` or `None` uses the portable JAX router. An unavailable requested extension also falls back to JAX. |
| `Configuration.pallas_cic` | `True` or `False` | `True` selects Pallas CIC and is the default. Unsupported configurations fall back to reference JAX CIC. `False` selects the reference implementation explicitly. |
| `Configuration.lpt_cache_strains` | `True` or `False` | `True` caches LPT strain arrays and is the default. `False` recomputes them to reduce retained memory. |
| `Configuration.nbody_cosmo_grad` | `True` or `False` | `True` includes cosmology cotangents and is the default. `False` omits them for objectives that need only particle or mode gradients. |
| `Configuration.chunk_size` | Any positive integer | Controls the JAX fallback chunk size. The default is `2**24`. |

See [Pallas CIC kernels](pallas_cic.md) for the CIC backend requirements.
