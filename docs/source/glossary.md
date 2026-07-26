# Glossary

```{glossary}
adjoint
  Reverse recurrence carrying derivatives of an objective with respect to the
  discrete simulation state. PM++ implements a custom N-body adjoint rather
  than retaining every forward step.

authoritative particle
  The unique particle record owned by the slab containing its current periodic
  position. `mesh_halo` stores authoritative particles without duplicated
  particle halo records.

capacity
  Compile-time maximum length of a padded particle or communication buffer.
  Exceeding a capacity truncates required data and invalidates the run.

CIC
  Cloud-in-cell mass assignment/interpolation. A particle contributes to the
  neighboring grid cells with multilinear weights; PM++ uses the same stencil
  for scatter and gather.

collector
  Forward-only callback that updates a caller-provided PyTree during N-body
  evolution without stacking every per-step result.

compute mesh
  One-dimensional `jax.sharding.Mesh` defining the ordered PM++ device axis.
  Its logical order determines x-slab order.

cosmology PyTree
  Dynamic arrays containing cosmological parameters and tabulated transfer,
  growth, and variance information. Unlike `Configuration`, these leaves can
  receive gradients.

custom VJP
  A user-defined vector-Jacobian product registered with JAX. PM++ uses custom
  VJPs for memory-efficient evolution and the exact transpose of distributed
  operators.

displacement
  Floating offset from a particle's integer mesh-cell identifier. Together,
  `pmid` and `disp` determine position.

halo
  Neighboring mesh cells (or, on the legacy path, particle records) made
  available so a local stencil can operate at a slab boundary.

LPT
  Lagrangian perturbation theory used to initialize particle displacement and
  canonical velocity. PM++ supports orders 0, 1 (Zel'dovich), and 2; order 3 is
  not implemented.

mesh_halo
  Preferred PM++ multi-GPU mode: particles are stored authoritatively, while
  mesh edge cells are exchanged for boundary scatter/gather.

nested white noise
  Resolution-consistent Fourier noise whose coefficients are generated from
  signed mode labels. Shared non-Nyquist modes match exactly for fixed box,
  seed, and compatible grids.

observer
  Forward-only function evaluated at schedule boundaries and stacked by
  `nbody_observe`.

particle_halo
  Compatibility mode that stores duplicated particle halo records. It is not
  the recommended production path.

PMWD
  Upstream differentiable particle-mesh reference implementation on which the
  inherited mathematical/adjoint design is based. It is not part of PM++'s
  public API.

slab decomposition
  Division of the periodic global x axis into equal contiguous ranges, one per
  logical device.

transposed spectral layout
  Distributed FFT output whose y axis, rather than x, is sharded. Keeping this
  natural layout avoids an extra collective around spectral operators.

unit-modulus noise
  Fourier noise normalized coefficient-by-coefficient to magnitude one. It
  randomizes phases without ordinary Gaussian amplitude fluctuations.
```
