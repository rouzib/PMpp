# Gradient check

Gradient checks should use the smallest problem that exercises the operation under test.

## Pattern

1. Build a tiny deterministic configuration.
2. Define a scalar loss from particles or density.
3. Evaluate `jax.value_and_grad`.
4. Check that all gradient leaves are finite.
5. Compare against finite differences or PMWD reference gradients when feasible.

```python
import jax
import jax.numpy as jnp

# loss = lambda x: ...
# value, grad = jax.value_and_grad(loss)(x0)
# assert jnp.isfinite(value)
# assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grad))
```

Use loose tolerances for finite differences through chaotic or long integrations, and document the perturbation size.
