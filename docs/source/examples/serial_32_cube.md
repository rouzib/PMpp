# Serial 32 cube

This is a compact script-style workflow for a deterministic serial run. It mirrors the first tutorial but keeps commentary minimal.

```python
import jax
import jax.numpy as jnp

from pmpp.boltzmann import boltzmann
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise
from pmpp.nbody import nbody
from pmpp.scatter import scatter

res = 32
conf = Configuration(100.0 / res, (res, res, res), mesh_shape=1, float_dtype=jnp.float32)
cosmo = boltzmann(SimpleLCDM(conf), conf)
ptcl = lpt(linear_modes(white_noise(0, conf), cosmo, conf), cosmo, conf)
ptcl = jax.jit(nbody, static_argnames=("conf", "reverse"))(ptcl, cosmo, conf)
density = scatter(ptcl, conf)

print("shape", density.shape)
print("mean", float(density.mean()))
print("finite", bool(jnp.isfinite(density).all()))
```

Expected diagnostics are a mesh-shaped density field, mean density close to `1.0`, and `finite True`.
