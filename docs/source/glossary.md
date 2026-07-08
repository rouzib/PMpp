# Glossary

```{glossary}
Particle-Mesh
  A method that deposits particle mass to a mesh, solves gravity on the mesh, and gathers forces back to particles.

x-slab
  A contiguous slab of the simulation domain along the x-axis owned by one device in a multi-GPU run.

static-capacity buffer
  A fixed-shape array allocated large enough for expected particle or halo communication under JAX compilation.

custom VJP
  A hand-written vector-Jacobian product used to define reverse-mode behavior for a function.
```
