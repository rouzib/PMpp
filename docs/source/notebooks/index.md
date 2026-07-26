# Notebook Gallery

These notebooks are pre-executed, self-contained research workflows. Read the
Docs renders the committed outputs and never executes notebook code. Every
notebook is regenerated locally with exactly two selected GPUs and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

Every notebook begins with a committed-execution admonition.  Its first executed output
records the JAX version, backend, selected and visible devices, seed, configuration, and
applicable static capacities used to produce the displayed plots.

```{admonition} Reproducing the gallery
:class: tip
Generate stable source cells with `python scripts/generate_docs_notebooks.py`,
execute with `python scripts/execute_docs_notebooks.py --write`,
then run `python scripts/validate_docs_notebooks.py`.
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 01 - First simulation
:link: 01_first_simulation
:link-type: doc

A complete 256-cube two-GPU run to $z=0$ with density and mass checks.
:::

:::{grid-item-card} 02 - Configuration
:link: 02_configuration
:link-type: doc

Grids, units, dtypes, schedules, JIT boundaries, decomposition, and capacities.
:::

:::{grid-item-card} 03 - Nested white noise
:link: 03_nested_white_noise
:link-type: doc

Matched initial modes and evolved $z=0$ fields from 32-cube through 256-cube.
:::

:::{grid-item-card} 04 - Multi-GPU mesh halo
:link: 04_multigpu_mesh_halo
:link-type: doc

A precomputed 256-cube, full-scale-factor two-GPU run with ownership and capacity checks.
:::

:::{grid-item-card} 05 - Observers and analysis
:link: 05_observers_and_analysis
:link-type: doc

Projection observers, compact collectors, final density, and differentiable power.
:::

:::{grid-item-card} 06 - Differentiation
:link: 06_differentiation
:link-type: doc

The full two-GPU adjoint for initial modes and an isolated cosmology sensitivity.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

01_first_simulation
02_configuration
03_nested_white_noise
04_multigpu_mesh_halo
05_observers_and_analysis
06_differentiation
```
