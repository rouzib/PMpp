# Pallas CIC kernels

PM++ uses cloud-in-cell (CIC) operations to deposit particle quantities on the
mesh and to interpolate mesh values back to particles. On qualified GPU runs,
the optional Pallas implementation replaces the materialising JAX CIC
expressions with tiled GPU kernels for the paired scatter and gather operations.
The same selection also covers their hand-written custom-VJP paths, so a full
AD run does not silently return to a slower forward-only implementation.

## Normal use

`Configuration.pallas_cic` is the one public CIC performance switch. It is
`True` by default, and a normal float32 GPU configuration needs no additional
Pallas setup:

```python
from pmpp.configuration import Configuration

conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=1,
    pallas_cic=True,
)
```

PM++ uses Pallas for both scatter and gather only when all of the following are
true:

- JAX exposes `jax.experimental.pallas` and is on the tested JAX 0.6 minor
  line;
- at least one JAX GPU device is visible; and
- `float_dtype` is `jax.numpy.float32`.

When a condition is not met, construction emits a `RuntimeWarning` and PM++
uses the portable JAX CIC implementation instead. The simulation remains
usable; the warning tells you that the Pallas optimization was not selected.
For a deliberate reference comparison, set `pallas_cic=False`.

```python
reference_conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=1,
    pallas_cic=False,
)
```

## What the kernels do

Each tiled kernel handles a block of particle IDs, displacements, and a validity
mask. The forward kernels perform CIC scatter with mesh atomic additions and
CIC gather from all eight neighboring cells. The custom VJPs provide the
corresponding particle, value, and mesh cotangents for reverse-mode AD.

Static particle buffers can include invalid capacity rows. The validity mask is
applied to every load, atomic update, and gradient write. If the last Pallas
tile is incomplete, PM++ pads that tile internally and masks its added rows;
there is no padded-versus-unpadded user option. This keeps the implementation
correct for arbitrary static capacities while retaining one tested fast path.

The kernels operate on the paired CIC operations. There are deliberately no
independent public gather and scatter switches: the paired path is what was
benchmarked and retained. Likewise, multi-channel mesh-halo force gather stays
fused rather than offering a lower-memory-benefit unfused branch.

## Performance and trade-offs

In the `512^3`, 63-step, float32 four-H100 benchmark described in
[Optimizations](optimizations.md), the qualified CUDA-routing plus Pallas-CIC
configuration gave the fastest measured full forward run (4.644 s) and was
statistically tied for the fastest full AD run (21.938 s). It used about
0.225 GiB/GPU more peak memory in the forward than the portable path, while its
AD peak was slightly lower in that measurement.

Treat those figures as a measured example, not a portability guarantee. Pallas
uses GPU atomics for scatter, so compare scientific observables and gradients
with the project's tolerances rather than requiring bitwise-identical arrays
from a different implementation or GPU architecture. The reference JAX CIC
path remains the appropriate baseline for new platform qualification and
debugging.

## Recommended choice

Use the default `pallas_cic=True` for a float32 GPU production forward or AD
run. Use `pallas_cic=False` only when comparing against the portable reference,
investigating a numerical issue, or running a platform outside the qualified
Pallas envelope. Do not treat disabling it as a required compatibility step:
PM++ performs that fallback automatically when it is needed.
