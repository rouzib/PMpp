# Evolution and analysis

This section follows a simulation after its initial particle state has been
created. It shows how to evolve the particles, construct density and projected
maps, collect forward diagnostics, and compute power spectra or
cross-correlations.

## Evolve particles

`nbody` advances an LPT particle state over `conf.a_nbody`:

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

n = 32
gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if len(gpu_devices) < 2:
    raise RuntimeError("This PM++ example requires at least two GPUs")

conf = Configuration(
    ptcl_spacing=100.0 / n,
    ptcl_grid_shape=(n,) * 3,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=create_compute_mesh(gpu_devices),
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
cosmo = boltzmann(SimpleLCDM(conf), conf)
noise = white_noise(3, conf)
modes = linear_modes(noise, cosmo, conf)
particles = lpt(modes, cosmo, conf)

@jax.jit
def evolve_and_scatter(initial_particles):
    final_particles = nbody(initial_particles, cosmo, conf)
    return final_particles, scatter(final_particles, conf)

final_particles, density = evolve_and_scatter(particles)
jax.block_until_ready(density)
assert bool(jnp.isfinite(density).all())
```

`reverse=False` follows the scale-factor schedule from `a_start` to `a_stop`.
`reverse=True` traverses it in reverse. It is not a substitute for the custom
reverse-mode adjoint used by `jax.grad`.

## Particle and density products

A `Particles` object carries integer mesh-cell identifiers (`pmid`), floating
displacements (`disp`), canonical velocities (`vel`), acceleration when
initialized, and masks for unused/halo slots. Obtain wrapped physical positions
with `particles.pos()` rather than reconstructing them from storage details.

`scatter(particles, conf)` uses CIC weights and defaults to a normalized density
whose volume mean is one. Useful checks are:

```python
assert density.shape == conf.mesh_shape
assert bool(jax.numpy.isfinite(density).all())
assert jax.numpy.allclose(density.mean(), 1.0, rtol=2e-5, atol=2e-5)
assert jax.numpy.allclose(density.sum(), conf.mesh_size, rtol=2e-5, atol=2e-5)
```

For a projected map, sum over one spatial axis. Projections along `y` or `z`
retain the decomposed global x axis and are useful for inspecting multi-GPU slab
boundaries. Projection along `x` hides those boundaries.

## Observers and collectors

Observers keep diagnostic work out of the core adjoint solver:

```python
from pmpp.nbody import nbody_observe
from pmpp.nbody_observers import density_projection_observer

observer = density_projection_observer(axis=2, normalize=True)
observe = jax.jit(
    lambda initial_particles: nbody_observe(
        initial_particles,
        cosmo,
        conf,
        observer,
        include_start=True,
        return_final=True,
    )
)
final_particles, images = observe(particles)
```

`nbody_observe` stacks `observer(a, particles, cosmo, conf)` once per integration
boundary. `nbody_collect` instead updates a caller-provided PyTree with a pure
function of the previous state and current step. Both are forward-only
diagnostic interfaces. Use `nbody` when differentiating the simulation.

Large per-step images can dominate memory. Prefer a collector when only a
running statistic or selected outputs are required.

## Power spectra and cross-correlations

```python
from pmpp.power_spectrum import density_to_pk, particles_to_pk

analyze_density = jax.jit(lambda field: density_to_pk(field, conf, mas="CIC"))
analyze_particles = jax.jit(lambda state: particles_to_pk(state, conf, mas="CIC"))
k, pk, nmodes = analyze_density(density)
k2, pk2, nmodes2 = analyze_particles(final_particles)
```

The `mas` argument describes the mass-assignment scheme and controls Fourier
window deconvolution. It must match how the field was constructed. Use `None`
to disable deconvolution. PM++ also provides density/particle cross-correlation
functions that return $r(k)$, the cross spectrum, both auto spectra, and mode
counts.

The isotropic shell estimator assumes a cubic box. Treat low-mode-count shells
and modes near the mesh Nyquist limit carefully. Save `nmodes` with every
spectrum so downstream fits can identify poorly sampled bins.

See [Particle-mesh force internals](../internals/particle_mesh.md), the
[particles and evolution API](../api/particles_evolution.rst), and the notebook gallery for
pre-executed projections and spectra.
