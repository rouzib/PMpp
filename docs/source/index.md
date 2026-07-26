# PM++ documentation

PM++ is a differentiable particle-mesh cosmology simulator built with JAX. It
supports the complete path from Gaussian initial modes to Lagrangian
perturbation theory (LPT), N-body evolution, density fields, summary statistics,
and gradients. Its distributed runtime scales the same model across a
one-dimensional mesh of accelerators.

```{important}
For real single-GPU science, use upstream
[PMWD](https://github.com/eelregit/pmwd): with matched inputs and numerical
settings, it should produce the same scientific output within the validated
tolerance. A one-device PM++ mesh is for troubleshooting only.
```

```{warning}
PM++ is research software. Record the package revision, complete configuration,
random seed, JAX version, and hardware with scientific results.
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Run your first simulation
:link: getting_started/first_simulation
:link-type: doc

Build a two-device mesh, evolve a small periodic box, and check mass
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

Study pre-executed, reproducible examples without requiring Read the Docs to
provide GPUs.
:::

::::

## Choose a path

- **New user:** [install PM++](getting_started/installation.md), then run the
  [first simulation](getting_started/first_simulation.md).
- **Scientific user:** use the [user guide](user_guide/index.md) to choose
  initial conditions, schedules, analysis products, and gradient targets.
- **Method developer:** read [How PM++ works](internals/index.md), then consult
  the [API reference](api/index.rst).
- **Contributor:** begin with [development and reference](development/index.md).

```{toctree}
:maxdepth: 4
:hidden:

getting_started/index
internals/index
notebooks/index
api/index
development/index
```
