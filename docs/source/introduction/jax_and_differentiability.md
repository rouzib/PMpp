# JAX and differentiability

PM++ uses JAX arrays and PyTrees so simulations can be transformed with `jax.jit`, `jax.vmap`, and differentiation utilities. The main runtime constraint is that compiled functions need static shapes. Multi-GPU particle exchange therefore uses fixed-capacity buffers rather than dynamically resizing arrays inside JIT-compiled code.

Long N-body runs are not differentiated by blindly storing every intermediate. PM++ provides custom VJP/adjoint code for selected stages, including the N-body path, distributed FFT helpers, scatter/gather, and halo movement. The goal is to expose useful gradients while controlling memory use and preserving the intended scientific transpose operations.

Key JAX concepts used throughout PM++:

- **PyTrees:** nested particle, cosmology, and runtime containers.
- **`jax.jit`:** compiles fixed-shape simulations.
- **`lax.scan`:** expresses repeated integration steps.
- **`custom_vjp`:** installs hand-written reverse passes where default reverse-mode is not appropriate.

For implementation details, see [N-body adjoint internals](../internals/nbody_adjoint.md).
