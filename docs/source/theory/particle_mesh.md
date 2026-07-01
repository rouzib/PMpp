# Particle-Mesh method

PM++ represents matter with particles and computes long-range gravity on a mesh. Particles are scattered to a density mesh, a potential or force field is solved in Fourier space, and forces are gathered back to particles. This is the standard Particle-Mesh trade-off: mesh operations are efficient and differentiable, while particle state carries the Lagrangian degrees of freedom.

The main implementation pieces are `pmpp.scatter`, `pmpp.gather`, `pmpp.gravity`, and `pmpp.steps`.
