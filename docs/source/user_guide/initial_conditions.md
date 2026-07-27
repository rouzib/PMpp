# Initial conditions

PM++ separates a random realization, cosmological scaling, and particle
initialization:

```python
import jax
import jax.numpy as jnp

from pmpp.boltzmann import boltzmann
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise, white_noise_nested
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh

n = 32
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ example requires at least two GPUs")

conf = Configuration(
    ptcl_spacing=100.0 / n,
    ptcl_grid_shape=(n,) * 3,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices[:2]),
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=32_768,
    max_share_ptcl=16_384,
    max_halo_share_ptcl=16_384,
    max_share_gather_ptcl=16_384,
    float_dtype=jnp.float32,
    a_start=1 / 64,
    a_stop=1 / 32,
    a_nbody_maxstep=1 / 64,
)

@jax.jit
def make_initial_particles(seed):
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    noise = white_noise(seed, conf)
    modes = linear_modes(noise, cosmo, conf)
    return lpt(modes, cosmo, conf)

particles = make_initial_particles(7)
jax.block_until_ready(particles.disp)
assert bool(jnp.isfinite(particles.disp).all())
```

## Ordinary Gaussian noise

`white_noise(seed, conf)` draws a real-space standard-normal field and returns
its rFFT by default. A fixed seed and identical configuration are deterministic.
Use `real=True` when a real-space white-noise array is specifically needed.

`white_noise` is appropriate for independent simulations at one resolution.
Changing resolution changes the PRNG draw layout, so corresponding low-frequency
modes are not promised to match.

## Phase-only noise

Set `unit_abs=True` to normalize each Fourier coefficient to unit modulus:

```python
make_phase_only = jax.jit(lambda seed: white_noise(seed, conf, unit_abs=True))
phase_only = make_phase_only(7)
jax.block_until_ready(phase_only)
assert bool(jnp.isfinite(phase_only).all())
```

This preserves randomized phases but removes Rayleigh amplitude fluctuations.
It is a different ensemble from ordinary Gaussian noise and must be recorded in
the simulation metadata. With `real=True`, PM++ transforms the normalized
spectrum back to a real-space field.

## Nested, resolution-consistent noise

For a convergence study at fixed box size, use
`white_noise_nested(seed, conf)`. It hashes `(seed, kx, ky, kz)` using signed
integer Fourier labels. Therefore the same non-Nyquist physical mode receives
the exact same coefficient at different resolutions.

The comparison contract is precise:

- use the same seed, box size, dtype, and compatible periodic grids;
- map signed mode labels to the storage index at each resolution;
- exclude the coarse-grid Nyquist planes, which are self-conjugate and have no
  unambiguous counterpart on the finer grid;
- compare Fourier coefficients, not equal-shaped slices chosen by raw index.

Shared coefficients are tested with exact array equality, not a tolerance. The
real-space fields have different sample grids and bandwidths; they are **not**
expected to be elementwise equal.

## Linear modes

`linear_modes` applies the linear power spectrum,

$$
\delta_\mathrm{lin}(\mathbf{k})
= \sqrt{V P_\mathrm{lin}(k)}\,\omega(\mathbf{k}),
$$

where $V$ is the box volume and $\omega$ is the normalized white-noise mode.
Pass `a=` to include growth to a requested scale factor, and `real=True` to
return the inverse-transformed field. Normally LPT consumes the Fourier result.

Call `boltzmann` first: it supplies the transfer and growth tables used by the
linear spectrum and LPT. PM++ currently uses the configured analytic transfer
fit; record changes such as `transfer_fit_nowiggle` as part of the model.

## Lagrangian perturbation theory

`lpt(modes, cosmo, conf)` constructs a uniform particle grid and adds the
configured perturbative displacement and canonical velocity at `a_start`.
Order 1 is the Zel'dovich approximation and order 2 adds the quadratic tidal
source. Order 0 leaves the grid unperturbed. Order 3 is not implemented and
raises at runtime.

On a device mesh, LPT returns particles already routed into the runtime's
authoritative ownership layout. Capacity errors during this stage invalidate
the result just as they do during N-body evolution.

For the equations and exact resolution-matching design, see
[Initial modes and LPT internals](../internals/initial_conditions.md). The
[initial-condition notebooks](../notebooks/index.md) show all schemes side by
side.
