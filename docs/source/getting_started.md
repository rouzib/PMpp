# Getting started

## Install

Install PM++ first, then return here to choose a path. See [Installation](install.md) for CPU, CUDA, editable, and documentation environments.

## Choose your path

- **I want a first CPU or serial simulation.** Start with [First serial PM++ simulation](tutorials/first_serial_run.md).
- **I want to run multi-GPU.** Read [Domain decomposition](introduction/domain_decomposition.md), then run [First multi-GPU `mesh_halo` run](tutorials/first_multigpu_mesh_halo_run.md).
- **I want gradients.** Read [JAX and differentiability](introduction/jax_and_differentiability.md), then follow [Differentiating through N-body](tutorials/differentiating_nbody.md).
- **I want implementation details.** Start with [Internals overview](internals/index.md).

## Minimal object map

- `Configuration` binds physical scale, mesh shape, runtime flags, capacities, and derived arrays.
- `Cosmology` and `SimpleLCDM` hold cosmological parameters used by transfer, growth, and integration routines.
- `modes` utilities generate white noise and linear modes.
- `Particles` carry particle ids, displacements, velocities, accelerations, and masks.
- `nbody` advances particles through the Particle-Mesh solver.
- `scatter` deposits particles to a mesh for density diagnostics.
