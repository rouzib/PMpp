# Differentiation

PM++ is designed for gradients through the same discrete solver used in the
forward calculation. Scatter, gather, gravity, distributed FFTs, ownership
movement, and full N-body evolution have explicit reverse-mode rules where the
runtime requires them.

```{warning}
**Current tested limitations (2026-07-26 worktree based on `161dc9b21a7a`):**
tracing through `boltzmann` table construction into cosmology leaves produced
NaN cotangents. For current cosmology-sensitivity work, reuse cached
transfer/growth tables, differentiate the N-body portion on the validated
two-GPU `mesh_halo` path, and validate it independently. This observation
describes the tested revision/environment, not a broader mathematical
limitation.
```

## Start with initial modes

Define a scalar loss and make the floating modes the differentiated argument:

```python
import jax
import jax.numpy as jnp

from pmpp.lpt import lpt
from pmpp.nbody import nbody
from pmpp.scatter import scatter

def loss_from_modes(modes, cosmo, conf):
    particles = lpt(modes, cosmo, conf)
    final_particles = nbody(particles, cosmo, conf)
    density = scatter(final_particles, conf)
    return jnp.mean((density - 1.0) ** 2)

value_and_grad = jax.jit(
    jax.value_and_grad(loss_from_modes),
    static_argnames=("conf",),
)
value, modes_grad = value_and_grad(modes, cosmo, conf)
```

Keep `conf` static, choose a tiny grid and short schedule first, and block the
result before reporting timing or success. The gradient must match `modes.shape`
and contain only finite values.

## Cosmological parameters

`Cosmology` is a PyTree whose core parameter arrays can receive gradients. The
N-body custom adjoint includes cosmology cotangents when
`conf.nbody_cosmo_grad=True` (the default). Set it to `False` only when the loss
requires displacement/mode gradients but not N-body cosmology gradients; this
is a computational choice that changes the returned cotangent.

Use `cosmology_param_names`, `cosmology_param_values`, and
`replace_cosmology_params` from `pmpp.cosmo` to form a stable vector of selected
parameters. Optional parameters such as curvature or dark-energy terms are
differentiable only when their underscored fields are not `None`.

## What is not the gradient path

- `nbody_observe` and `nbody_collect` are forward diagnostic interfaces.
- `nbody(reverse=True)` integrates the equations over a reversed schedule; it
  does not compute a VJP.
- A gradient of a truncated capacity-overflow run is invalid even if every
  returned array is finite.

## Validate before scaling

For a scalar parameter $\theta$, compare the autodiff result to a centered
finite difference,

$$
\frac{\partial L}{\partial\theta}
\approx \frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}.
$$

Choose $\epsilon$ large enough to exceed float32 round-off but small enough to
remain in the local linear regime; repeat at multiple values. For array-valued
inputs, test a directional derivative $\nabla L\cdot v$ rather than perturbing
every element.

Validation order:

1. finite primal loss and gradient;
2. centered finite difference on a tiny case;
3. invariance/mass checks in the differentiated forward path;
4. focused gradient tests for the modified operator;
5. end-to-end gradient test at the intended runtime mode.

The pre-executed differentiation notebook uses the currently validated
two-GPU initial-mode $\rightarrow$ LPT $\rightarrow$ N-body adjoint. Its
directional finite-difference case matched at about $1.7\times10^{-10}$ relative
error; that number documents one test case and is not a universal tolerance.
The notebook's cosmology example is deliberately N-body-only with cached
transfer/growth data because of the limitation above.

The custom recurrence is described in
[Integration and discrete adjoint](../internals/integration_and_adjoint.md).
