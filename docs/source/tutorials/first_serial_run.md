# First serial PM++ simulation

## What you will do

Run a small serial Particle-Mesh simulation, scatter final particles to a density mesh, and check the output shape and mean density.

## Requirements

- PM++ installed with a CPU or GPU JAX backend.
- No multi-GPU setup required.

## Complete code

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
box_size = 100.0

conf = Configuration(
    box_size / res,
    (res, res, res),
    mesh_shape=1,
    float_dtype=jnp.float32,
)

cosmo = boltzmann(SimpleLCDM(conf), conf)
modes = white_noise(0, conf)
modes = linear_modes(modes, cosmo, conf)
ptcl = lpt(modes, cosmo, conf)

nbody_jit = jax.jit(nbody, static_argnames=("conf", "reverse"))
ptcl_final = nbody_jit(ptcl, cosmo, conf)
density = scatter(ptcl_final, conf)

print(density.shape)
print(float(density.mean()))
```

## Step-by-step explanation

PM++ generates deterministic white noise, scales it into linear modes, initializes particles with LPT, evolves them with `nbody`, and deposits final particles onto the mesh.

## Expected output

The density shape should match the mesh shape, and the mean density should be close to `1.0`.

## Common failures

- The first run may be slow because JAX is compiling.
- Import failures usually mean the editable install did not complete or the environment is not activated.

## Next steps

Try [initial conditions](initial_conditions.md) or [differentiating through N-body](differentiating_nbody.md).
