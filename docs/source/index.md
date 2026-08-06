```{image} _static/pmpp-logo.svg
:alt: PM++ logo
:width: 360px
:align: center
:class: pmpp-landing-logo
```

# PM++ documentation

PM++ is a differentiable particle-mesh cosmology simulator built with JAX. It
supports the complete path from Gaussian initial modes to Lagrangian
perturbation theory (LPT), N-body evolution, density fields, summary statistics,
and gradients. Its distributed runtime scales the same model across a
one-dimensional mesh of accelerators.

```{note}
PM++ is designed primarily for multi-accelerator simulations. For single-GPU
work, [PMWD](https://github.com/eelregit/pmwd) is a closely related alternative.
With matched inputs and numerical settings, the two should agree within
validated tolerances.
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Run your first simulation
:link: getting_started/first_simulation
:link-type: doc

Build a multi-device mesh, evolve a small periodic box, and check mass
conservation in about ten minutes.
:::

:::{grid-item-card} Understand the solver
:link: internals/architecture
:link-type: doc

Connect the equations to the arrays, JAX transforms, communication, and custom
adjoints used by PM++.
:::

:::{grid-item-card} Explore notebooks
:link: notebooks/index
:link-type: doc

Study reproducible examples pre-executed on multiple GPUs.
:::

::::

## Choose a path

- **New user:** [install PM++](getting_started/installation.md), then run the
  [first simulation](getting_started/first_simulation.md).
- **Scientific user:** use the [user guide](user_guide/index.md) to choose
  initial conditions, schedules, analysis products, and gradient targets.
- **Method developer:** read [How PM++ works](internals/index.md), then consult
  the [API reference](api/index.rst).

```{toctree}
:maxdepth: 4
:hidden:

getting_started/index
internals/index
notebooks/index
api/index
development/index
```
