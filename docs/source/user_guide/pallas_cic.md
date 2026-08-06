# Pallas CIC kernels

PM++ uses cloud-in-cell (CIC) operations to deposit particle quantities on the
mesh and to interpolate mesh values back to particles. On qualified GPU runs,
the optional Pallas implementation replaces the materialising JAX CIC
expressions with tiled GPU kernels for the paired scatter and gather operations.
The same selection also covers their hand-written custom-VJP paths.

## Normal use

`Configuration.pallas_cic` is the one public CIC performance switch. It is
`True` by default, and a normal float32 GPU configuration needs no additional
Pallas setup:

```python
from pmpp import Configuration

n = 64
conf = Configuration(
    ptcl_spacing=1.0,
    ptcl_grid_shape=(n, n, n),
    mesh_shape=1,
    pallas_cic=True,
)
```

PM++ uses Pallas for both scatter and gather only when all of the following are
true:

- A JAX minor line qualified for these kernels is installed, currently 0.6 or
  0.10, and `jax.experimental.pallas` is available. Other supported JAX minor
  lines use reference CIC until they are qualified independently.
- At least one JAX GPU device is visible.
- `float_dtype` is `jax.numpy.float32`.

When a condition is not met, construction emits a `RuntimeWarning` and PM++
uses the portable JAX CIC implementation instead. The simulation remains
usable. The warning tells you that the Pallas optimization was not selected.
For a deliberate reference comparison, set `pallas_cic=False`.

```python
reference_conf = Configuration(
    ptcl_spacing=1.0,
    ptcl_grid_shape=(n, n, n),
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
tile is incomplete, PM++ pads that tile internally and masks its added rows.
there is no padded-versus-unpadded user option. This keeps the implementation
correct for arbitrary static capacities while retaining one tested fast path.

The kernels operate on the paired CIC operations. There are deliberately no
independent public gather and scatter switches. The paired path is what was
benchmarked and retained. Likewise, multi-channel mesh-halo force gather stays
fused rather than offering a lower-memory-benefit unfused branch.

## Performance and trade-offs

Pallas CIC can make the individual gather and scatter operations about four to
six times faster. The end-to-end simulation benefit is smaller because CIC is
only one part of the full pipeline.

Pallas uses GPU atomics for scatter, so compare simulation observables and
gradients with the project's tolerances rather than requiring bitwise-identical
arrays across implementations or GPU architectures. The reference JAX CIC path
remains available for platform qualification and debugging.
