# Troubleshooting

## Capacity overflow

**Symptom:** output contains `[ERROR] Exceeded ... capacity` or reports a
particle/share count larger than its maximum.

**Meaning:** a fixed-shape compact or communication buffer was truncated. This
is a correctness failure, not a warning about performance.

**Action:** identify the named limit, increase it, start a fresh process if
needed, and rerun from initial conditions. Increase `max_ptcl_per_slice` for
authoritative storage, `max_share_ptcl` for drift migration,
`max_halo_share_ptcl` for halo rebuilds, or `max_share_gather_ptcl` for gathered
value exchange. If a particle crosses more than a neighboring slab in one
drift, reduce the step size; a larger buffer does not fix that routing contract.

## Incompatible decomposition

**Symptom:** local shapes are wrong, a collective fails, or an x dimension is
silently shorter than expected.

**Action:** for $D$ devices, require particle-grid x, mesh-grid x, and mesh-grid
y dimensions to divide $D$ exactly. Particle generation splits x, while the
distributed FFT changes from x sharding to y sharding. Keep particle and mesh
aspect ratios compatible, and ensure a requested mesh halo is no wider than one
local x slab. Recreate the configuration after changing visible devices.

## Cluster job sees no GPUs

Check inside the allocation: scheduler GPU resources, loaded CUDA module,
driver/JAX compatibility, and `CUDA_VISIBLE_DEVICES`. Print `jax.devices()` at
job start. A login-node device check does not describe the compute node. On
managed clusters, keep site-specific module, account, and path commands out of
portable PM++ examples.

## Unexpected recompilation

**Symptom:** repeated calls compile again.

**Action:** reuse the exact `Configuration`, schedule, correction structure,
and input shapes. Configuration fields are static JAX metadata. Do not construct
a new schedule or change dtypes/capacities inside a hot loop. Separate compile
and execution timing with a warm-up and `jax.block_until_ready`.

## Out of memory

Reduce the particle/mesh resolution, shorten the schedule while debugging,
reduce unnecessarily generous capacities, set `lpt_cache_strains=False` to
trade extra 2LPT FFTs for memory, or use `float32`. Gradient runs include a
forward solve and adjoint/recomputation work; a forward run fitting in memory
does not prove its gradient will fit. Run only one heavy multi-GPU process at a
time.

## Mean density is not one

Check for an overflow first. Then verify that `scatter` used its default value,
the particle state contains all authoritative particles exactly once, and the
density was not already converted to overdensity. A normal default scatter has
`density.sum() ~= conf.mesh_size` and `density.mean() ~= 1`.

## Nested modes do not match

Use `white_noise_nested` on both grids with the same seed, box, and dtype. Map
signed Fourier labels, exclude the coarse Nyquist planes, and compare Fourier
coefficients. Raw array slices and different-resolution real-space arrays are
not the matching contract.

## Gradient disagrees with finite differences

Start with a tiny grid and one short step, check several finite-difference
scales, and test a directional derivative. Confirm `nbody_cosmo_grad` is enabled
for cosmology targets. Eliminate overflows and nondifferentiable diagnostics
from the loss. Then run the focused gradient test for the operator closest to
the mismatch before the full N-body test.

## Cosmology gradient through `boltzmann` is NaN

NaN cosmology cotangents can occur when reverse mode includes transfer and
growth table construction in `boltzmann`. Cache those tables, differentiate the
N-body sensitivity separately, and verify it with a directional finite
difference. Do not report that narrower check as an end-to-end gradient through
Boltzmann initialization.
