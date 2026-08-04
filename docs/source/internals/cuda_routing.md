# Optional CUDA routing

PM++ keeps the portable JAX routing implementation as the correctness
fallback.  An optional CMake-built shared library in `cuda/` provides typed
JAX FFI handlers for the local parts of `mesh_halo` routing:

For build, selection, fallback, and performance guidance, see the user-facing
[CUDA routing guide](../user_guide/cuda_routing.md). This page describes the
extension ABI and adjoint boundary.

```text
CUDA route-pack -> JAX capacity checks and ppermute -> CUDA stable route-merge
```

The handlers are shard-local.  They do not create a communicator, access a
remote device, or replace XLA collectives.  The pack output is a fixed-capacity
array of eight `uint32` words per particle (32 bytes): a raveled `pmid`, a
validity word, and bit-copied displacement and velocity triples.  Counts are
returned separately, so a migration overflow can be checked in JAX before a
collective is issued; writes beyond the static capacity are predicated.

The extension is optional and is selected only for the qualified JAX 0.6 CUDA
float32 path.  Build it with the PMPP virtual environment's Python:

```bash
python scripts/build_cuda_routing.py
export PMPP_CUDA_ROUTING_LIBRARY=$PWD/cuda/build/libpmpp_cuda_routing.so
```

The normal Hatchling wheel does not run CMake and does not require `nvcc`.
Set `PMPP_CUDA_ROUTING=0` to force the canonical JAX route even when a shared
library is present.  The hand-written N-body adjoint remains the derivative
boundary and recomputes the route plan during the reverse sweep.  With the
extension enabled, the auxiliary merge emits source tags/indices for the
CUDA local transpose; the acceleration-only payload and all collective
reversal remain in JAX.  No route maps are saved for the full trajectory.
