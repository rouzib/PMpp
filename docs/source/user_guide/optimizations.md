# Optimizations

PM++ keeps one canonical mesh-halo routing implementation: packed migration
collectives, sparse canonical merge, and fused multi-channel force gather.
Those implementation choices are not user switches because the alternatives
were slower, consumed more memory, or could not differentiate.

The remaining settings are deliberately few. The recommendations below come
from a fresh-process `512^3`, 63-step, float32 benchmark on four H100 80 GiB
GPUs (five steady-state samples; no capacity warnings). They are useful
defaults, not a substitute for measuring a different topology or resolution.

## Recommended configurations

For a full forward simulation, use:

```python
from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration

conf = Configuration(
    ptcl_spacing,
    ptcl_grid_shape,
    mesh_shape=1,
    multigpu=MultiGPUConfiguration(
        compute_mesh=compute_mesh,
        mode="mesh_halo",
    ),
    pallas_cic=True,
    lpt_cache_strains=True,
    nbody_cosmo_grad=True,
)
```

For a full AD run, use the same configuration. In particular, keep
`nbody_cosmo_grad=True` whenever derivatives with respect to cosmological
parameters are part of the objective. Set it to `False` only for an explicitly
displacement-only objective; in the benchmark it did not provide a meaningful
full-AD gain.

Omitting `cuda_routing` selects it automatically: a qualified CUDA FFI build
is used when available, while missing or incompatible CUDA support falls back
to the portable JAX router. Set `cuda_routing=False` only for diagnosis or a
controlled comparison. See [CUDA routing](cuda_routing.md) for the optional
build and qualification details.

## Measured result

| Setting | Full forward | Full AD | Peak memory per GPU (forward / AD) |
| --- | ---: | ---: | ---: |
| Portable canonical path | 4.792 s | 28.955 s | 6.076 / 11.613 GiB |
| CUDA routing | 4.707 s | 22.005 s | 6.076 / 11.613 GiB |
| CUDA routing + Pallas CIC | 4.644 s | 21.938 s | 6.301 / 11.567 GiB |

The Pallas-CIC configuration was the fastest measured full forward and was
statistically tied for the fastest full AD. Its forward memory cost was about
0.225 GiB per GPU relative to the portable baseline; the AD peak was slightly
lower. The current implementation masks and internally pads an incomplete
final tile only as needed for correctness; this is not a user setting. See
[Pallas CIC kernels](pallas_cic.md) for its runtime requirements.

## Remaining flags

- `MultiGPUConfiguration.mode="mesh_halo"` is the preferred distributed path.
  It keeps only authoritative particles on each slab and exchanges mesh halos.
- `MultiGPUConfiguration.cuda_routing` enables the optional CUDA FFI
  route-pack/merge implementation. It gives the large AD reduction above when
  qualified and otherwise automatically falls back.
- `Configuration.pallas_cic` selects Pallas for both CIC gather and scatter.
  It is `True` by default. On unsupported dtype, backend, or JAX versions, PM++
  warns before JIT and uses reference JAX CIC instead.
- `Configuration.lpt_cache_strains=True` avoids redundant LPT FFT work at the
  cost of retaining strain arrays. The benchmark found no meaningful full-run
  runtime difference, so choose it according to whether the additional retained
  arrays fit your memory budget.
- `Configuration.nbody_cosmo_grad=True` retains complete cosmology cotangents.
  This is required for a full scientific AD result.
- `Configuration.chunk_size` controls the ordinary JAX scatter/gather chunk
  size used by the portable fallback. The default `2**24` was adequate in the
  benchmark; altering it did not improve the tested full pipeline.

## Removed choices

The following are intentionally not configuration options:

- compact particle merge: slower forward and AD and much higher AD memory;
- chunked migration exchange: no forward benefit and its dynamic loop cannot
  be reverse-mode differentiated;
- unpacked mesh-halo exchange: no material speed benefit and higher memory;
- separate Pallas gather and scatter controls: using both is the useful path;
- non-fused mesh gather: no speed benefit and substantially higher AD memory.

Any result with a capacity overflow warning is invalid. Increase the named
capacity and rerun before treating timing, memory, density, or gradients as a
comparison.
