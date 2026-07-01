# Quickstart

A minimal forward workflow imports from `pmpp`, constructs a configuration and cosmology, generates initial modes, initializes particles with LPT, and advances them with the N-body solver.

```python
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.modes import white_noise, linear_modes
from pmpp.lpt import lpt
from pmpp.nbody import nbody
from pmpp.scatter import scatter

conf = Configuration(ptcl_spacing=1.0, ptcl_grid_shape=(32, 32, 32), mesh_shape=1)
cosmo = SimpleLCDM()
wn = white_noise(0, conf)
modes = linear_modes(wn, cosmo, conf)
ptcl = lpt(modes, cosmo, conf)
ptcl = nbody(ptcl, cosmo, conf)
delta = scatter(ptcl, conf)
```

This small example is intended for local CPU or single-device experimentation after installation. Production-scale and multi-GPU runs require appropriate JAX devices and should not be executed in the Read the Docs build.
