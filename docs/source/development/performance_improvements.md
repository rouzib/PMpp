# Performance optimizations

The `performance_imrpovements` branch implements the seven proposals from the
review of commit `1406f91`.

| Change | Scope and compatibility |
| --- | --- |
| Shared force inverse FFT | The ordinary spectral force redistributes potential and x-force intermediates, then forms y/z forces locally. The transpose combines y/z cotangents before redistribution. Existing Nyquist factors and the large-array scalar fallback are retained. |
| Half-spectrum rFFT transpose | Real FFT pullbacks use a weighted inverse real FFT with the original shape. Interior compressed frequencies get half weight; zero and even-grid Nyquist frequencies get full weight. Odd lengths and arbitrary complex cotangents are tested. |
| CUDA merge | Each source calculates its insertion position directly using the stable order stay, left, right. Active outputs are fully overwritten and only the inactive suffix is cleared. Both lean and metadata-producing merges use this algorithm. |
| Separate halo gather | Float32 Pallas CIC reads owned, left-halo and right-halo allocations directly. Its backward rule accumulates those three cotangents separately. The portable and float64 paths retain their existing implementation. |
| Differentiable fused drift | Ordinary force-bound drifts use the existing fused native primal when its int16/float32 feature is available. A custom VJP reconstructs routing during the backward pass and differentiates displacement, velocity and the drift factor. Phase corrections and unsupported configurations retain their established path. |
| Streamed 2LPT | A running diagonal sum evaluates each diagonal once, then subtracts the three off-diagonal squares: six strain inverse FFTs instead of nine. |
| Routing diagnostics and capacities | Independent diagnostics share small collectives; invalid counts retain saturating arithmetic. A host-side capacity report sizes buffers from supplied observations without reducing defaults or bypassing failure checks. |

These changes use the configured device mesh and ring permutations. Four- and
eight-device CPU tests exercise topology and numerical behavior; they do not
measure performance on larger GPU machines.

## Reproducing numerical checks

Use a supported GPU JAX environment. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
and include `src`, the repository root, and `tests` in `PYTHONPATH`.

```bash
python -m pytest -c pyproject.toml tests/test_performance_improvements.py -q
python -m pytest -c pyproject.toml tests/test_grad_fft_distributed.py \
  tests/test_mesh_halo_scatter_gather.py tests/test_grad_gather.py \
  tests/test_grad_gravity.py tests/test_grad_lpt.py \
  tests/test_grad_nbody_mesh_halo.py -q
```

For the native checks, build the extension with a compatible CUDA toolkit and
CMake 3.24 or newer, then select both the library and manifest:

```bash
python cuda/build_cuda_routing.py --build-dir cuda/build-performance
export PMPP_CUDA_ROUTING_LIBRARY="$PWD/cuda/build-performance/libpmpp_cuda_routing.so"
export PMPP_CUDA_ROUTING_MANIFEST="$PWD/cuda/build-performance/pmpp_cuda_routing.manifest.json"
export PMPP_REQUIRE_CUDA_ROUTING_TESTS=1
python -m pytest -c pyproject.toml tests/test_cuda_bidir_mergepath.py \
  tests/test_cuda_routing_equivalence.py tests/test_performance_improvements.py -q
```

Four- and eight-device topology checks require separate fresh processes:

```bash
for devices in 4 8; do
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=$devices \
  PMPP_TEST_DEVICES=$devices python -m pytest -c pyproject.toml \
    tests/test_performance_improvements.py -q
done
```
