# Differentiation

PM++ is designed for gradients through the same discrete solver used in the
forward calculation. Scatter, gather, gravity, distributed FFTs, ownership
movement, and full N-body evolution have explicit reverse-mode rules where the
runtime requires them.

```{important}
Repeatable output is not by itself evidence of an accurate gradient. Long
float32 multi-GPU adjoints can accumulate reconstruction error even when XLA's
deterministic GPU mode makes every repetition bitwise identical. Read
[Precision, reproducibility, and gradients](precision_reproducibility.md)
before choosing float32, float64, or deterministic execution for a simulation
gradient.
```

## Start with initial modes

Define a scalar loss and make the floating modes the differentiated argument:

```python
import jax
import jax.numpy as jnp

from pmpp.cosmology import boltzmann
from pmpp import Configuration
from pmpp.cosmology import SimpleLCDM
from pmpp.initial_conditions import lpt
from pmpp.initial_conditions import linear_modes, white_noise
from pmpp import MultiGPUConfiguration
from pmpp.nbody import nbody
from pmpp.cic import scatter
from pmpp.distributed import create_compute_mesh

n = 16
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ example requires at least two GPUs")

conf = Configuration(
    ptcl_spacing=50.0 / n,
    ptcl_grid_shape=(n,) * 3,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices),
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=4_096,
    max_share_ptcl=2_048,
    max_halo_share_ptcl=2_048,
    max_share_gather_ptcl=2_048,
    float_dtype=jnp.float32,
    a_start=1 / 64,
    a_stop=1 / 32,
    a_nbody_maxstep=1 / 64,
)
cosmo = SimpleLCDM(conf)
noise_real = white_noise(7, conf, real=True)

def loss_from_noise(noise_real, cosmo, conf):
    evolved_cosmo = boltzmann(cosmo, conf)
    modes = linear_modes(noise_real, evolved_cosmo, conf)
    particles = lpt(modes, evolved_cosmo, conf)
    final_particles = nbody(particles, evolved_cosmo, conf)
    density = scatter(final_particles, conf)
    return jnp.mean((density - 1.0) ** 2)

value_and_grad = jax.jit(
    jax.value_and_grad(loss_from_noise),
    static_argnames=("conf",),
)
value, noise_grad = value_and_grad(noise_real, cosmo, conf)
jax.block_until_ready(noise_grad)
assert bool(jnp.isfinite(value))
assert noise_grad.shape == noise_real.shape
assert bool(jnp.isfinite(noise_grad).all())
```

Keep `conf` static, choose a tiny grid and short schedule first, and block the
result before reporting timing or success. The gradient must match the input
noise shape and contain only finite values.

## Cosmological parameters

`Cosmology` is a PyTree whose core parameter arrays can receive gradients. The
N-body custom adjoint includes cosmology cotangents when
`conf.nbody_cosmo_grad=True` (the default). Set it to `False` only when the loss
requires displacement or mode gradients but not N-body cosmology gradients. This
is a computational choice that changes the returned cotangent.

Use `cosmology_param_names`, `cosmology_param_values`, and
`replace_cosmology_params` from `pmpp.cosmology` to form a stable vector of selected
parameters. Optional parameters such as curvature or dark-energy terms are
differentiable only when their underscored fields are not `None`.

## What is not the gradient path

- `nbody_observe` and `nbody_collect` are forward diagnostic interfaces.
- `nbody(reverse=True)` integrates the equations over a reversed schedule. It
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
remain in the local linear regime. Repeat at multiple values. For array-valued
inputs, test a directional derivative $\nabla L\cdot v$ rather than perturbing
every element.

Validation order:

1. finite primal loss and gradient.
2. centered finite difference on a tiny case.
3. invariance and mass checks in the differentiated forward path.
4. focused gradient tests for the modified operator.
5. end-to-end gradient test at the intended runtime mode.

The pre-executed differentiation notebook uses the currently validated
multi-GPU initial-mode $\rightarrow$ LPT $\rightarrow$ N-body adjoint. Its
float64 directional finite-difference case has a relative error of order
$10^{-10}$.
That result documents one test case and is not a universal tolerance. The
notebook's cosmology example differentiates N-body evolution with cached
transfer and growth data.

The custom recurrence is described in
[Integration and discrete adjoint](../internals/integration_and_adjoint.md).
