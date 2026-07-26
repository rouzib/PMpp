# User Guide

The user guide explains the scientific and runtime choices that define a PM++
experiment. Start with configuration and initial conditions, then add evolution,
analysis, gradients, or multiple devices as needed.

The examples use exactly two GPUs, which is the smallest topology that exercises
PM++'s distributed ownership, communication, and FFT paths.

- [Configuration](configuration.md): geometry, units, precision, schedules,
  compilation, and static capacities.
- [Initial conditions](initial_conditions.md): Gaussian, phase-only, and nested
  noise; transfer functions; and LPT.
- [Evolution and analysis](evolution_and_analysis.md): N-body runs, observers,
  density projections, and spectra.
- [Differentiation](differentiation.md): supported gradient targets and
  validation strategy.
- [Multi-GPU execution](multigpu.md): device meshes, `mesh_halo`, ownership,
  distributed FFTs, and capacity planning.
- [Troubleshooting](troubleshooting.md): overflows, decompositions, compilation,
  and memory failures.

The [notebook gallery](../notebooks/index.md) pairs these explanations with
pre-executed examples.

```{toctree}
:maxdepth: 1
:hidden:

configuration
initial_conditions
evolution_and_analysis
differentiation
multigpu
```
