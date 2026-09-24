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
hybrid suite passed 21 logical CPU tests, including checks that native-like FFI
merge outputs satisfy both strict and hybrid conditional axis types.
An initial four-H100 run with JAX 0.10.2 built and registered the native ABI,
but stopped during tracing on a manual-axis type mismatch in the FFI merge
result. The follow-up explicitly marks that result varying. A subsequent H100
run passed 16 hybrid tests, failed one capacity test because its requested chunk
exceeded its send capacity, and failed the full N-body test with an NCCL
`ncclAlltoAll` runtime error. The capacity test setup is now corrected. The
same NCCL error affected separate LPT, gravity, N-body, and FFT tests, so its
cause remains unresolved. The four routing microbenchmarks completed; their
single-run medians were 3.12 ms for strict, 2.43 ms for hybrid with no far
particles, 3.42 ms for sparse far traffic, and 2.83 ms for dense far traffic.
Compiled temporary memory was 0.40 MB for strict and 5.16 MB for hybrid in this
small case. These measurements do not establish full-simulation performance or
peak device memory. The four-process smoke test did not start because its Slurm
step requested an untyped GPU within a typed H100 allocation; the command below
now uses `h100:1`.

On the same four-H100 node, JAX 0.9.1 passed the standalone JAX all-to-all check
that failed with JAX 0.10.2. The rerun passed 18 tests in the hybrid group
(12 skipped), 42 routing tests, six gradient tests, the N-body gradient script, and
14 performance regression tests. The four-process smoke script stopped before
routing because it queried FFI registration before configuration construction;
that ordering is now corrected, but the distributed run remains unqualified.
The JAX 0.9.1 route-only medians were 1.58 ms strict, 2.05 ms no-far hybrid,
3.89 ms sparse hybrid, and 2.78 ms dense hybrid. The no-far case was about 30%
slower than strict in this run and used 5.16 MB versus 0.40 MB of compiled
temporary memory. This does not meet the ordinary-step speed target. Repeat
measurements and full forward/gradient comparisons are needed before considering
hybrid a production default. A follow-up keeps the no-far route at one
conditional and places the second conditional only on exceptional or failed
steps. It passed 45 focused logical CPU tests. On a different four-H100 node,
100-iteration medians were 2.73 ms strict and 2.01 ms no-far hybrid. The strict
median changed substantially between nodes, so the speed difference is not yet
a stable conclusion. Compiled temporary memory remained 0.40 MB strict versus
5.16 MB hybrid. The four-process smoke now reaches routing, but fails in NCCL
with `invalid device ordinal`. A standalone four-process JAX collective
diagnostic is provided below to isolate the launcher/runtime from native routing.

The local machine has two RTX 3090s and no `nvcc` on PATH. CPU tests do not
establish native CUDA correctness, no-far overhead, scratch usage, or H100
performance. Keep hybrid opt-in until all target-node gates below pass. The
source changes preserve record format v3; the new manifest feature and target
prevent a stale library from qualifying.

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
srun --ntasks=4 --gpus-per-task=h100:1 --gpu-bind=single:1 --kill-on-bad-exit=0 \
  timeout 300 python scripts/test_hybrid_distributed.py
```

If that step fails in NCCL, run the same four-process launcher without PM++:

```bash
srun --ntasks=4 --gpus-per-task=h100:1 --gpu-bind=single:1 --kill-on-bad-exit=0 \
  timeout 300 python scripts/diagnose_jax_distributed.py
```

The diagnostic prints each rank's visible devices and tests a JAX ring
permutation and all-to-all. A failure there indicates that the launcher or JAX
collective runtime needs attention before the PM++ distributed smoke can
qualify. A pass narrows the remaining issue to the PM++ route or native ABI.

Use a free coordinator port and adjust the GPU binding flag if the site's
Slurm configuration requires it. A single process over four GPUs cannot
establish matching collectives across processes. Run the relevant full N-body
gradient and LPT tests in that distributed setup as well. If the
scientific workflow uses 16 or 32 GPUs, repeat the small sparse case there
before the 1024-cubed run. Test a reordered logical device mesh and asymmetric
source/receiver counts at the target topology.

If the LPT, gravity, or FFT tests fail with an NCCL `ncclAlltoAll` error, run a
standalone four-device JAX collective in the same allocation to separate the
collective runtime from PM++ code:

```bash
export NCCL_DEBUG=INFO
srun --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  python scripts/diagnose_jax_alltoall.py
```

Keep the NCCL log and the test log. A failing standalone collective does not
qualify the FFT or full N-body path; a passing one narrows the issue to the
distributed FFT or its sharding configuration.

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
