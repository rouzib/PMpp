# Notebooks

These notebooks are source artifacts for hands-on PM++ exploration. They are included in the documentation with execution disabled so the docs remain CPU-safe and reproducible on hosted builders.

```{admonition} Running locally
:class: tip
Install PM++ with the documentation extras, start Jupyter from the repository root, and open the notebooks from `docs/source/notebooks/`. The multi-GPU notebook requires a CUDA/JAX environment with multiple visible devices.
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PM++ showcase
:link: pmpp_showcase
:link-type: doc
:class-card: sd-shadow-sm pmpp-card

A guided tour of the core serial workflow, including initial conditions, PM evolution, and diagnostics.
:::

:::{grid-item-card} Multi-GPU PMWD local comparison
:link: mGPU_pmwd_local
:link-type: doc
:class-card: sd-shadow-sm pmpp-card

A multi-GPU-oriented notebook for comparing local PM++ behavior with PMWD-style workflows.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

pmpp_showcase
mGPU_pmwd_local
```
