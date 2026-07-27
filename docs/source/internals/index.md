# How PM++ Works

These pages connect the mathematical model to the active implementation in
`src/pmpp`. They explain mechanisms and design trade-offs; private helper names
are not a stability promise.

- [System architecture](architecture.md): the end-to-end dataflow and the
  separation between scientific state and static runtime metadata.
- [Initial modes and LPT](initial_conditions.md): Gaussian modes, transfer and
  growth, exact nested low-frequency matching, and particle initialization.
- [Particle-mesh force](particle_mesh.md): CIC scatter/gather, Poisson solve,
  spectral differentiation, shapes, and adjoints.
- [Integration and discrete adjoint](integration_and_adjoint.md): the symplectic
  step, observers, and reverse-time custom VJP.
- [Distributed runtime](distributed_runtime.md): slabs, ownership migration,
  mesh halos, transposed FFT layouts, and static buffers.

The inherited simulation and adjoint mathematics follow
[Li et al., *Differentiable Cosmological Simulation with the Adjoint Method*,
arXiv:2211.09815v2](https://arxiv.org/abs/2211.09815v2). PM++ extends that
foundation with the distributed data structures and communication paths
described here. The paper is linked rather than copied into the documentation.

## Diagram legend

All diagrams read left to right. Green nodes are scientific state, amber nodes
are numerical operators, indigo nodes are cross-device layout/communication,
and red nodes are validation or loss products. A textual equivalent follows
every diagram for accessibility and non-JavaScript builds.

Every Mermaid diagram is interactive: drag to pan, use the mouse wheel or
trackpad to zoom, use the arrow keys to pan while the diagram is focused, and
press `+`, `-`, or `0` to zoom in, zoom out, or reset. The visible controls and
fullscreen viewer provide the same actions.

```{toctree}
:maxdepth: 1
:hidden:

architecture
initial_conditions
particle_mesh
integration_and_adjoint
distributed_runtime
```
