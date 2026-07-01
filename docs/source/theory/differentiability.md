# Differentiability

PM++ is designed around JAX transformations. Scatter/gather, force solves, LPT, and N-body integration are written so simulations can participate in automatic differentiation. Where the default JAX transpose is not the desired scientific or memory behavior, PM++ provides custom adjoint/VJP code for specific stages.

Differentiability does not remove the need for static shapes in compiled multi-GPU workflows. Capacity parameters must be set large enough before compilation, and capacity overflow is a correctness failure rather than a recoverable warning.
