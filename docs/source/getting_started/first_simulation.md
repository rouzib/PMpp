# Your first simulation

This example evolves a periodic $256^3$ particle load to $a=1$ in a
$100\,h^{-1}\mathrm{Mpc}$ box on exactly two GPUs. The run therefore
exercises slab ownership, mesh halos, and the distributed FFT layout.

## Complete example

```python
import jax
import jax.numpy as jnp

from pmpp.boltzmann import boltzmann
from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.lpt import lpt
from pmpp.modes import linear_modes, white_noise
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.nbody import nbody
from pmpp.scatter import scatter
from pmpp.utils import create_compute_mesh

resolution = 256
box_size = 100.0  # Mpc/h with the default PM++ length unit
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ simulation requires at least two GPUs")
selected_devices = gpu_devices[:2]
compute_mesh = create_compute_mesh(selected_devices)

conf = Configuration(
    ptcl_spacing=box_size / resolution,
    ptcl_grid_shape=(resolution,) * 3,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=compute_mesh,
        mode="mesh_halo",
    ),
    max_ptcl_per_slice=10_066_329,
    max_share_ptcl=1_600_000,
    max_halo_share_ptcl=800_000,
    max_share_gather_ptcl=1_800_000,
    float_dtype=jnp.float32,
    a_start=1 / 64,
    a_stop=1.0,
    a_nbody_maxstep=1 / 64,
)

@jax.jit
def run(seed):
    cosmo = boltzmann(SimpleLCDM(conf), conf)
    noise = white_noise(seed=seed, conf=conf)
    modes = linear_modes(noise, cosmo, conf)
    particles = lpt(modes, cosmo, conf)
    final_particles = nbody(particles, cosmo, conf)
    return final_particles, scatter(final_particles, conf)

final_particles, density = run(0)
jax.block_until_ready(density)

valid = ~final_particles.unused_index
expected_mass = conf.mesh_size
print("density shape:", density.shape)
print("finite:", bool(jnp.isfinite(density).all()))
print("mean density:", float(density.mean()))
print("deposited mass:", float(density.sum()))
print("valid particle slots:", int(valid.sum()))

assert density.shape == conf.mesh_shape
assert bool(jnp.isfinite(density).all())
assert jnp.allclose(density.mean(), 1.0, rtol=2e-5, atol=2e-5)
assert jnp.allclose(density.sum(), expected_mass, rtol=2e-5, atol=2e-5)
```

The single outer `jax.jit` boundary covers cosmology setup, random modes, LPT,
N-body evolution, and the final density scatter. The first call compiles this
complete pipeline; block a later call before measuring steady-state execution.

Mesh-halo routing always uses the packed collective and canonical sparse merge
plan. These preserve particle ordering and adjoint provenance without exposing
legacy routing switches.

## What each stage does

1. `Configuration` fixes box/grid geometry, precision, the scale-factor
   schedule, and the two-device runtime.
2. `boltzmann(SimpleLCDM(...))` tabulates transfer and growth quantities.
3. `white_noise` produces a deterministic Gaussian realization and
   `linear_modes` scales it by the linear matter power spectrum.
4. `lpt` maps the initial density modes to particle displacement and canonical
   velocity at `a_start`.
5. `nbody` performs the configured kick-drift-kick evolution; its custom VJP is
   used if this calculation is differentiated.
6. `scatter` deposits particles with cloud-in-cell (CIC) weights. Its default
   normalization makes a complete periodic particle load have mean density one.

The output mesh has shape `(256, 256, 256)`, is finite, and conserves the normalized
mass `conf.mesh_size`. The two devices own the global x intervals `[0, 128)` and
`[128, 256)` and exchange the boundary data required by the PM operators.

## Where to go next

- Change box, grid, schedule, and precision in [Configuration](../user_guide/configuration.md).
- Compare Gaussian, phase-only, and resolution-consistent noise in
  [Initial conditions](../user_guide/initial_conditions.md).
- Add projections and power spectra in
  [Evolution and analysis](../user_guide/evolution_and_analysis.md).
- Scale the same pipeline with [Multi-GPU execution](../user_guide/multigpu.md).
