# Differentiating through N-body

## What you will do

Define a tiny scalar loss from a final density field and evaluate gradients with JAX.

## Requirements

Use a very small mesh first; gradient runs can have significant compile-time and memory costs.

## Complete code

```python
import jax
import jax.numpy as jnp

# Build a tiny configuration, cosmology, modes, and particles as in the serial tutorial.
# Then define a scalar loss and call jax.value_and_grad on the chosen input.
```

## Step-by-step explanation

PM++ custom VJPs make gradients a deliberate feature rather than an incidental result of tracing every step. Gradients can be taken with respect to differentiable leaves such as initial modes or selected cosmological parameters, depending on the function signature and `nbody_cosmo_grad` setting.

## Expected output

The scalar loss should be finite and gradient leaves should have finite values.

## Common failures

Large runs may fail from memory pressure or long compilation. Start small and scale only after checking finite gradients.

## Next steps

Read [N-body adjoint internals](../internals/nbody_adjoint.md).
