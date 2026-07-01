# Single-device forward run

This is the safe starting point for local experimentation. Use small meshes first and import from `pmpp`, not from `src`.

```python
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.modes import white_noise, linear_modes
from pmpp.lpt import lpt
from pmpp.nbody import nbody

conf = Configuration(1.0, (16, 16, 16), mesh_shape=1)
cosmo = SimpleLCDM()
ptcl = lpt(linear_modes(white_noise(0, conf), cosmo, conf), cosmo, conf)
ptcl = nbody(ptcl, cosmo, conf)
```
