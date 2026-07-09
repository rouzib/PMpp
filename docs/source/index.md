<div class="pmpp-hero">
  <div class="pmpp-kicker">Differentiable cosmology • JAX • Multi-GPU</div>
  <h1>PM++ Documentation</h1>
  <p>Build, differentiate, and scale Particle-Mesh N-body simulations with a documentation path that moves from installation to notebooks, internals, and API references.</p>
</div>

```{warning}
PM++ is research software. Interfaces, correction models, and multi-GPU runtime details may change until a stable release is declared.
```

## Start here

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Install PM++
:link: install
:link-type: doc
:class-card: sd-shadow-sm pmpp-card

Prepare a Python/JAX environment and install the package for local experiments or documentation work.
:::

:::{grid-item-card} Explore notebooks
:link: notebooks/index
:link-type: doc
:class-card: sd-shadow-sm pmpp-card

Open the moved showcase and multi-GPU notebooks directly from the docs tree.
:::

:::{grid-item-card} Read internals
:link: internals/index
:link-type: doc
:class-card: sd-shadow-sm pmpp-card

Follow the system diagrams, mesh exchange, adjoint, and runtime architecture notes.
:::

::::

## What PM++ helps with

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Differentiable N-body pipelines
:class-card: sd-shadow-sm pmpp-card

Trace gradients through selected Particle-Mesh stages while keeping long-running simulations memory-aware.
:::

:::{grid-item-card} Multi-GPU decomposition
:class-card: sd-shadow-sm pmpp-card

Use static-capacity buffers, mesh halos, distributed FFT utilities, and clear debugging guidance for slab-decomposed runs.
:::

:::{grid-item-card} Research-friendly examples
:class-card: sd-shadow-sm pmpp-card

Start small with CPU-safe examples, then scale notebooks and tutorials on the hardware that matches your experiment.
:::

:::{grid-item-card} Maintainer references
:class-card: sd-shadow-sm pmpp-card

Dive into API pages, implementation diagrams, and documentation conventions when extending PM++.
:::

::::

## Documentation sections

```{toctree}
:maxdepth: 2
:hidden:

install
notebooks/index
internals/index
api/index
```

- [Installation](install.md)
- [Notebooks](notebooks/index.md)
- [Internals and system diagrams](internals/index.md)
- [API reference](api/index.rst)
