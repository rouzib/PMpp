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
diagnostic fails at its first ring permutation with the same error, before
importing PM++. In that Slurm step, every task reports `CUDA_VISIBLE_DEVICES=0`;
the failure therefore belongs to the four-process GPU visibility/collective
setup, not the hybrid route. The target single-node qualification uses one task
that sees all four H100s. On `tg11101`, that layout with JAX/jaxlib 0.9.1
listed CUDA devices 0 through 3, passed the standalone four-device JAX
all-to-all, and passed all nine native hybrid tests in 122.84 seconds. This
qualifies the tested small native routing, capacity, and N-body gradient cases
on one four-H100 node; full-scale speed and peak device memory remain unmeasured.
With the benchmark's initial far capacities of 65,536 records and 100 timed
iterations on the same node, strict no-far routing took 2.150 ms median and
hybrid no-far took 2.912 ms median, a 35.4 percent regression in that run.
Compiled temporary storage was 0.397 MB versus 5.156 MB. The highest reported
JAX allocator peak was 2.411 MB for strict and 34.585 MB for hybrid; on each
other device it was 1.322 MB versus 9.313 MB. These peaks include input setup,
compilation, execution, and validation. Final bytes in use were much closer,
so the large peaks are transient. This run misses the ordinary-path speed and
memory targets. Earlier timings differed substantially, so repeat on the same
node before claiming a stable latency ratio. The 65,536-record far buffer is
over ten times the 6,144 active particles per device in this synthetic case;
capacity sensitivity and the phase of the allocator peak need measurement.
With 256-record far buffers and 64-record chunks on the same node, the no-far
hybrid median was 2.122 ms and the 64-far-particle median was 2.839 ms.
Compiled temporary storage dropped to 0.422 MB. Devices 1 through 3 peaked
at 1.346 MB, close to the previous strict 1.322 MB. GPU 0 still reported a
34.585 MB cumulative peak, reached immediately after compilation; it did not
increase during warmups, timed execution, or validation. That peak therefore
does not measure steady-state routing memory. The smaller capacity removes the
large persistent compiler buffer estimate, but the near-equal no-far latency
needs paired repeat runs to establish a stable ratio.

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

For the target single-node mode, give one task all four H100s. Do not bind each
task to a single GPU. Export the library and manifest paths shown above, then
run the standalone JAX collective and the native hybrid tests from the same
interactive allocation:

```bash
srun --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  timeout 300 python scripts/diagnose_jax_alltoall.py
srun --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  timeout 900 python -m pytest tests/test_hybrid_native.py -q
```

The native tests include far migration, capacity failure, and a complete small
N-body/gradient comparison in this one-process topology. The four-process
scripts exercise a different runtime mode and are only needed if that mode is
deployed. Its current Slurm launcher fails in a pure JAX collective, so it does
not provide evidence about the PM++ route. If the scientific workflow uses 16
or 32 GPUs, qualify that separate process topology before the 1024-cubed run.
Test a reordered logical device mesh and asymmetric source/receiver counts at
the target topology.

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
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  python scripts/benchmark_hybrid_route.py --policy neighbor_only --far-per-device 0 --output results/hybrid/strict.json
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 0 --output results/hybrid/hybrid-zero.json
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 64 --output results/hybrid/hybrid-sparse.json
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 4096 --output results/hybrid/hybrid-dense.json
```

Use additional cases varying `--far-chunk-size`, active offsets, per-device
concentration, and `--particles-per-device`. The driver records compilation,
median and p95 synchronized route latency, compiled buffer estimates, JAX
allocator peak bytes per device, JAX versions, device identities, manifest,
capacities, and particle conservation. The allocator peak includes compilation
and execution in each fresh process. It may not include all external CUDA
allocations, so collect total device peak memory with Nsight Systems or an
equivalent profiler during the steady-state run. Also
compare complete forward steps and full value-and-gradient simulations in
fresh processes; isolated routing speed is not sufficient. Record valid and
padded bytes, collective counts, fallback frequency, and maximum per-device
send and receive traffic. Aggregate multi-process time by the slowest rank.

For a capacity sensitivity check, keep the same no-far input and rerun hybrid
with 256-record far send/receive buffers and 64-record chunks. The benchmark
now records allocator peaks after input setup, compilation, warmups, timed
runs, and validation, so the phase where the peak rises is visible. A separate
sparse case with 64 far particles per device checks that this smaller capacity
can actually carry the synthetic exceptional traffic. These values are for
diagnosis; production capacities must cover observed worst-case traffic.

```bash
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  timeout 1800 python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 0 \
  --far-send-capacity 256 --far-recv-capacity 256 --far-chunk-size 64 \
  --iterations 100 --output results/hybrid/hybrid-zero-cap256.json
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  timeout 1800 python scripts/benchmark_hybrid_route.py --policy hybrid --far-per-device 64 \
  --far-send-capacity 256 --far-recv-capacity 256 --far-chunk-size 64 \
  --iterations 100 --output results/hybrid/hybrid-sparse-cap256.json
```

The acceptance target is approximately at most 1–2 percent regression on
ordinary steps, with no unexplained peak-memory increase. Sparse exceptional
traffic must conserve particles and beat or justify the alternatives at the
measured production distribution. Report any missed target; do not change the
workload or capacities between baseline and candidate to hide it.

## Full 256-cubed forward run in a 5 Mpc/h box

`scripts/run_hybrid_256_box5.py` runs 2LPT followed by 63 N-body steps from
`a=0.1` to `a=1.0` with seed 0 and four H100s in one task. It uses hybrid
native routing, Pallas CIC, and the forward-only low-memory LPT/N-body paths.
The initial capacities are 1.5 times the mean local particle count for final
storage, 2,000,000 neighbor shares, 1,000,000 far sends and receives, and
16,384-record far chunks. These are starting bounds, not claims about the
traffic in a 5 Mpc/h box. The runner fails if routing reports invalid traffic,
an overflow is raised, the density is nonfinite, or mass is not conserved.
N-body routing returns one replicated failure status to the host. On failure,
all devices skip the remaining solver work and the host raises a
`ParticleRoutingFailure` with the failing step, migration count, invalid
candidate count, last good occupancy, and configured capacities. A migration
count alone does not identify which bound failed.
It writes a phase-by-phase JSON report, a full density `.npy`, three axis
projections `.npy`, and a projection PNG. The density copy and plotting happen
after the timed simulation stages. `nbody_compile` and `nbody` are measured
separately. Use `--execution-runs 2` to warm the remaining forward stages and
record `forward_execution_seconds` for a second execution; this includes
white noise, linear modes, LPT, N-body, and scatter, but excludes cosmology
setup, density transfer, and plotting. With one execution, the other forward
stage timers still include first-call compilation.

```bash
cd /home/r/rouzib/links/scratch/pmpp_repo/PMpp
source ../ENV_2/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$PWD:$PWD/tests"
export PMPP_CUDA_ROUTING_LIBRARY="$PWD/cuda/build-hybrid-h100/libpmpp_cuda_routing.so"
export PMPP_CUDA_ROUTING_MANIFEST="$PWD/cuda/build-hybrid-h100/pmpp_cuda_routing.manifest.json"
mkdir -p results/hybrid
srun --export=ALL --ntasks=1 --gpus-per-task=h100:4 --kill-on-bad-exit=0 \
  timeout 7200 python -u scripts/run_hybrid_256_box5.py \
  --npart 256 --box-size 5 --nbody-steps 63 --seed 0 \
  --sigma8 0.80 --n-s 0.96 --omega-m 0.30 --omega-b 0.05 --h 0.70 \
  --max-ptcl-per-slice 9000000 --max-share-ptcl 2000000 \
  --far-send-capacity 1000000 --far-recv-capacity 1000000 \
  --far-chunk-size 16384 --execution-runs 2 \
  --plot-dir /project/6112408/rouzib/PMpp/results \
  --output results/hybrid/full256-box5-hybrid.json \
  2>&1 | tee results/hybrid/full256-box5-hybrid.log
```

All values above are command-line options. To specify an exact per-GPU
particle slot count, use `--max-ptcl-per-slice N`; it overrides
`--max-ptcl-factor`. Other adjustable bounds are `--max-halo-share-ptcl`,
`--max-share-gather-ptcl`, and `--lpt-share-multiplier`. The scale-factor
interval (`--a-start`, `--a-stop`), `--lpt-order`, and `--mesh-shape` are also
configurable. Keep the output name unique between runs.
The runner rejects a per-GPU particle capacity larger than the total number
of particles; this catches an extra digit before allocating large buffers.

To plot the already saved successful 256-cubed density without repeating the
simulation:

```bash
python scripts/plot_hybrid_density.py \
  results/hybrid/full256-box5-hybrid-cap8m_density.npy \
  --box-size 5 --output-dir /project/6112408/rouzib/PMpp/results
```

The 256-cubed H100 report supplied on 2026-09-23 recorded a peak occupancy of
8,438,611 particles per GPU, but its configured slot count was 89,388,608,
not approximately nine million. Its 12.56-second N-body phase included JIT
compilation. Do not extrapolate that phase time or its 13.88-GB allocator peak
to 1024 cubed without a correctly sized, compile-separated run.
For the same 5 Mpc/h box on four GPUs, naive particle-count scaling predicts
about 540 million occupied slots per GPU at 1024 cubed and roughly 576 million
slots with the 256-cubed headroom. Scaling the observed allocator peak by slot
count alone gives about 89 GB/GPU, above the reported 63.8-GB H100 allocator
limit; the larger mesh adds further memory. This is an estimate, not a
qualified 1024-cubed run. A 512-cubed measurement with correct capacities is
the next useful memory check before considering a memory rewrite or more GPUs.

The result must report `"status": "ok"`, zero LPT and N-body invalid counts,
finite density, and mass within tolerance. A capacity failure requires a new
run with larger explicit bounds; do not use its partial density. The small
four-logical-CPU portable smoke completed with one N-body step and conserved
512 particles to float32 precision. That check does not qualify the CUDA
hybrid run or its full 256-cubed performance.
