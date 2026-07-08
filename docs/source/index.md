# PM++

PM++ is a JAX-based differentiable Particle-Mesh cosmology package for serial and multi-GPU N-body simulations. The display name is **PM++**, the Python distribution is `pmpp`, and code is imported with `import pmpp` rather than `import PM++`.

```{warning}
PM++ is research software. Interfaces, correction models, and multi-GPU runtime details may change until a stable release is declared.
```

::::{grid} 2
:::{grid-item-card} Differentiable Particle-Mesh simulations
Run Particle-Mesh simulations with JAX transformations and custom adjoints for selected stages.
+++
[Learn the model](introduction/particle_mesh.md)
:::

:::{grid-item-card} Multi-GPU slab decomposition
Use x-slab ownership, particle migration, and static-capacity buffers for distributed runs.
+++
[Multi-GPU tutorial](tutorials/first_multigpu_mesh_halo_run.md)
:::

:::{grid-item-card} Distributed FFTs and mesh halos
Use distributed Fourier solves and the preferred `mesh_halo` exchange path for current multi-GPU work.
+++
[Domain decomposition](introduction/domain_decomposition.md)
:::

:::{grid-item-card} Validation and corrections
Compare against PMWD reference behavior and experiment with potential-correction workflows.
+++
[Validation example](examples/pmwd_validation.md)
:::
::::

## Start here

- [Install PM++](install.md)
- [Run a first serial simulation](tutorials/first_serial_run.md)
- [Run a first multi-GPU `mesh_halo` simulation](tutorials/first_multigpu_mesh_halo_run.md)
- [Browse the API reference](api/index.rst)

## Code teaser

```python
from pmpp.configuration import Configuration
from pmpp.modes import white_noise

conf = Configuration(100.0 / 32, (32, 32, 32), mesh_shape=1)
modes = white_noise(0, conf)
print(modes.shape)
```

```{toctree}
:maxdepth: 2
:caption: User guide

getting_started
install
introduction/index
tutorials/index
examples/index
how_to/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
internals/index
glossary
faq
citation
license
```

```{toctree}
:maxdepth: 1
:caption: Development

development/index
```
