# Hybrid nonlocal particle migration

Implementation base: `730e5c7f8922fad6168c27ca29e27119e5b8aa47` on the
`non-local-exchange` branch. The user-provided implementation plan informed the
design; this page records the implementation and its qualification status.

## Contract and configuration

`migration_policy="neighbor_only"` retains strict one-hop behavior. Opt-in
`migration_policy="hybrid"` is supported for dynamic `mesh_halo` with at least
four logical devices and a qualified float32 displacement/velocity, int16
`pmid`, native `bidir_mergepath` library. It requires independent positive
`far_send_capacity`, `far_recv_capacity`, and `far_chunk_size` values. Other
modes or a stale/missing native ABI fail during configuration construction.
One to three devices have no possible non-neighbor destination and require no
far buffers. The existing `max_share_ptcl` still bounds neighbor streams;
`max_ptcl_per_slice` bounds the final authoritative state.

```python
multigpu=MultiGPUConfiguration(
    compute_mesh=compute_mesh,
    mode="mesh_halo",
    cuda_routing=True,
    cuda_routing_backend="bidir_mergepath",
    migration_policy="hybrid",
    far_send_capacity=65536,
    far_recv_capacity=65536,
    far_chunk_size=8192,
)
```

These capacities are initial benchmark values, not production recommendations.
Measure the largest per-device send and receive counts before choosing them.

The fused native pack and both neighbor packet exchanges remain on the ordinary
path. One small replicated reduction checks capacity and the presence of
exceptional destinations before any final merge. With no exception, the
original fused merge runs. With an exception, the JAX branch compacts far
source slots, gathers a small destination-count table, exchanges bounded
packets with static `ppermute` permutations, sorts compact arrivals, and calls
the new asymmetric-capacity fused CUDA merge. The final merge keeps stay,
ordinary-left, ordinary-right, then far tie order; far ties preserve ascending
logical source rank and original slot order. Invalid coordinates and any
capacity or native/portable count disagreement fail closed. Successful far
traffic does not count as an invalid route.

The generic JAX route handles LPT, auxiliary fields, saved-input pullbacks,
and reverse reconstruction with the same source/destination schedule. The
ordinary differentiable N-body adjoint uses those reverse helpers. The
dedicated `nbody_low_memory` solver remains forward-only. An old CUDA library
may still serve strict routing, but cannot enable hybrid.

## Local evidence and limits

Four- and eight-logical-device CPU tests cover direct exceptional packets,
mixed neighbor/far traffic, empty ranks, several chunk sizes, capacity
failure, duplicate-key ordering, route transpose, reverse reconstruction, and
near/far/near dispatch through one compiled callable. Test instrumentation
observed no exceptional callback in the two near calls and one on each of four
shards in the far call. Shard-map replication checking was enabled. The focused
hybrid suite passed 19 tests; the existing routing/capacity suite passed 21.
The native four-GPU tests are present but have not run on this machine: it has
two RTX 3090s and no `nvcc` on PATH. CPU tests do not establish native CUDA
correctness, no-far overhead, scratch usage, or H100 performance. Keep hybrid
opt-in until all target-node gates below pass. The source changes preserve
record format v3; the new manifest feature and target prevent a stale library
from qualifying.

## Four-H100 node: build and correctness gates

Use the same synced checkout and Python/JAX environment for building and
testing. Run one GPU process at a time. Keep compilation and execution logs.
The commands below assume a shell at the repository root and a virtualenv
already active with JAX/jaxlib 0.9.1 or newer.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD:$PWD/tests"
git rev-parse HEAD
git status --short
python -c 'import jax,jaxlib; print(jax.__version__,jaxlib.__version__,jax.devices())'
nvcc --version
python -m src.pmpp.distributed.build_cuda --force --cuda-architectures '90;90-virtual' --target-dir "$PWD/cuda/build-hybrid-h100"
export PMPP_CUDA_ROUTING_LIBRARY="$PWD/cuda/build-hybrid-h100/libpmpp_cuda_routing.so"
export PMPP_CUDA_ROUTING_MANIFEST="$PWD/cuda/build-hybrid-h100/pmpp_cuda_routing.manifest.json"
export PMPP_REQUIRE_HYBRID_NATIVE=1
export PMPP_HYBRID_TEST_PLATFORM=gpu
python -m pytest tests/test_hybrid_configuration.py tests/test_hybrid_far_exchange.py tests/test_hybrid_fused_dispatch.py tests/test_hybrid_native.py -q
python -m pytest tests/test_cuda_bidir_mergepath.py tests/test_deep_routing_invariants.py tests/test_capacity_failure.py tests/test_routing_configuration.py -q
python -m pytest tests/test_grad_lpt.py tests/test_deep_lpt_orders.py tests/test_grad_gather.py tests/test_grad_gravity.py -q
python tests/test_grad_nbody_mesh_halo.py
python -m pytest tests/test_performance_improvements.py -q
```

Require zero failures and zero unexpected skips in the native hybrid tests.
Check that the manifest advertises `hybrid_fused_drift_i16_f32`, the new merge
symbol registers, and the physical devices are H100. The native test includes
strict-versus-hybrid ordinary equivalence, long-jump rescue, a tiny independent
global oracle, a particle-specific saved-input VJP, multiple periodic wraps,
exact slab endpoints and adjacent float32 values, invalid coordinates, and
far-send/receive/final capacity exhaustion. It also compares a small complete
hybrid LPT, N-body, scatter, and gradient run against strict routing with no
far particles. Extend native qualification with
asymmetric neighbor stream limits and intentionally stale or absent libraries.
A failure must be global and must not yield a usable partial state.

Run the actual four-process smoke job in a four-H100 Slurm allocation, with one
visible GPU per task. It injects exceptional traffic on rank 0 only and checks
the final state on every rank. Export the library and manifest paths shown
above in the batch job, then run:

```bash
export PMPP_COORDINATOR_ADDRESS="$(hostname -s):12355"
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
  timeout 300 python scripts/test_hybrid_distributed.py
```

Use a free coordinator port and adjust the GPU binding flag if the site's
Slurm configuration requires it. A single process over four GPUs cannot
establish matching collectives across processes. Run the relevant full N-body
gradient and LPT tests in that distributed setup as well. If the
scientific workflow uses 16 or 32 GPUs, repeat the small sparse case there
before the 1024-cubed run. Test a reordered logical device mesh and asymmetric
source/receiver counts at the target topology.

## Four-H100 node: performance and memory gates

Run these in separate fresh processes, with identical input shapes and the
same integration state for strict and no-far hybrid. Save JSON results.

```bash
python scripts/benchmark_hybrid_route.py --policy neighbor_only --far-per-device 0 --output results/hybrid/strict.json
python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 0 --output results/hybrid/hybrid-zero.json
python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 64 --output results/hybrid/hybrid-sparse.json
python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 4096 --output results/hybrid/hybrid-dense.json
```

Use additional cases varying `--far-chunk-size`, active offsets, per-device
concentration, and `--particles-per-device`. The driver records compilation,
median and p95 synchronized route latency, compiled buffer estimates, JAX
versions, device identities, manifest, capacities, and particle conservation.
Compiled memory estimates do not include every CUDA scratch or allocator peak.
Collect device peak memory with Nsight Systems or an equivalent profiler during
the steady-state run, including FFI scratch and communication buffers. Also
compare complete forward steps and full value-and-gradient simulations in
fresh processes; isolated routing speed is not sufficient. Record valid and
padded bytes, collective counts, fallback frequency, and maximum per-device
send and receive traffic. Aggregate multi-process time by the slowest rank.

The acceptance target is approximately at most 1–2 percent regression on
ordinary steps, with no unexplained peak-memory increase. Sparse exceptional
traffic must conserve particles and beat or justify the alternatives at the
measured production distribution. Report any missed target; do not change the
workload or capacities between baseline and candidate to hide it.
