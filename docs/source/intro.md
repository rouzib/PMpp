# PM++

PM++ is a JAX-based, differentiable Particle-Mesh cosmology simulator with active support for multi-GPU domain decomposition. The public Python package is imported as `pmpp` because `PM++` is not a valid Python package identifier.

The active implementation lives in `src/pmpp`. The `pmwd/` directory is retained as a reference implementation used for comparison and regression work; it is not folded into the public `pmpp` API.

PM++ combines white-noise generation, transfer functions, Lagrangian perturbation theory (LPT), particle scatter/gather, a Particle-Mesh force solve, leapfrog-style N-body integration, and custom adjoint code for differentiable simulations.
