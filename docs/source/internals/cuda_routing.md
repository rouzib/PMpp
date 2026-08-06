# Optional CUDA routing

PM++ can replace the shard-local classification, packing, stable merge, and
transpose-scatter parts of `mesh_halo` particle migration with a typed
[JAX FFI extension][jax-ffi] {cite:p}`openxlaFfiDocs`. Device-to-device
communication remains a JAX collective. The extension changes how the
canonical route is evaluated, not the route's mathematical contract.

```{mermaid}
flowchart LR
  A["Canonical local particle slots"] --> B["CUDA classify and pack"]
  B --> C["JAX ppermute left and right"]
  C --> D["CUDA stable merge"]
  D --> E["Canonical slots on new owner"]
```

## Boundary between CUDA and JAX

[JAX FFI][jax-ffi] and XLA FFI provide a CUDA stream to a device handler, and
CUDA streams support asynchronous kernel enqueueing
{cite:p}`openxlaFfiDocs,nvidiaCudaStreamsDocs`. PM++'s shard-local handlers
enqueue their kernels on that stream and return without a private device
synchronization. Temporary scan storage comes from the XLA FFI scratch
allocator used in `route_kernels.cu`.

The ring exchanges remain [`jax.lax.ppermute`][jax-ppermute]:

```python
jax.lax.ppermute(...)
```

This leaves the logical topology and collective ordering visible to JAX. The
CUDA library never selects a remote device or performs peer communication
itself.

## Route record

An outgoing float32 particle is encoded as eight `uint32` words:

| Word | Contents |
| --- | --- |
| 0 | periodic raveled `pmid` key |
| 1 | validity word |
| 2 to 4 | three float32 displacement components, copied bitwise |
| 5 to 7 | three float32 velocity components, copied bitwise |

The record is 32 bytes. It is an opaque communication payload. The floating
values are not numerically converted to integers. Their bit patterns are
copied into the words and restored by the merge kernel.

Float64 uses a separate typed FFI target and fourteen `uint32` words: the same
two key/validity words followed by six two-word float64 bit patterns. The
resulting record is 56 bytes. Keeping both records as `uint32` arrays preserves
the ordinary JAX `ppermute` communication boundary.

Acceleration is absent because the ordinary drift immediately refreshes the
force. When an internal adjoint helper needs acceleration to reconstruct a
route plan, that field is compacted and exchanged separately.

## Classification

For each valid slot, the CUDA classifier applies the same periodic interval
tests as the JAX implementation and emits one class:

```text
0  padding
1  stay
2  send left
3  send right
4  outside the one-hop domain
```

Class 4 and every capacity excess are checked by the surrounding JAX route.
The extension returns uncapped counts even though its output arrays have fixed
capacities.

## Stable packing

Classification is followed by a stable compaction:

1. each CUDA block counts selected rows
2. CUB computes an exclusive scan of the block counts
3. a block-level exclusive scan gives each selected lane its local rank
4. the block offset plus local rank gives its output position.

This scan-based packing follows the parallel prefix-scan and compaction
construction implemented by CUB {cite:p}`merrill2016scan`.

Because input slots are already in canonical key order, preserving input order
also keeps each outgoing record stream sorted by key.

The current route can classify and pack one direction at a time. The
bidirectional target classifies once, scans independent stay, left, and right
counts, and writes all three compact streams while preserving their source
order.

## Stable merge

After `ppermute`, each shard has its sorted stay stream and one or two sorted
incoming streams. The merge must reproduce the canonical order even when
multiple particles have the same key.

The ordinary merge treats the stay stream as virtual. A compact prefix and
block counts locate the $r$th staying particle in the original full-capacity
array. Binary searches compute the output rank of each stay or incoming item.
This avoids first copying the entire stay payload into another full-capacity
buffer.

The bidirectional merge treats stay, left, and right as three sorted streams.
Each output thread finds which stream owns its output diagonal. Lower- and
upper-bound searches encode the tie order

```text
stay < left < right
```

for equal keys. The kernel writes the selected `pmid`, displacement, velocity,
validity, key, and provenance tag and index. The diagonal partitioning is a
GPU merge-path construction extended here to three streams
{cite:p}`green2012merge`.

## Typed FFI targets

The shared library exports these typed handlers:

| Target | Role |
| --- | --- |
| `pmpp_route_pack` | classify and pack one direction |
| `pmpp_route_bidir_pack` | classify and pack stay, left, and right together |
| `pmpp_route_merge` | stable merge with one incoming stream |
| `pmpp_route_merge_aux` | merge plus source provenance |
| `pmpp_route_merge_bidir` | three-stream stable merge plus provenance |
| `pmpp_route_transpose_split` | split merged cotangents by source tag |
| `pmpp_route_transpose_scatter` | scatter returned cotangents to source slots |

These unsuffixed targets are the float32 ABI. Each target also has a float64
counterpart with an `_f64` suffix.

`cuda_routing.py` loads the shared library with `ctypes`, converts exported
symbols to JAX capsules, and registers them for the CUDA platform. Each Python
wrapper declares exact input and output shapes and dtypes with
`jax.ShapeDtypeStruct` before calling [`jax.ffi.ffi_call`][jax-ffi]
{cite:p}`openxlaFfiDocs`.

## Differentiation boundary

The FFI pack and merge targets do not define a standalone JAX derivative.
They execute inside the custom VJP of the full N-body evolution. JAX
[does not automatically differentiate foreign calls][jax-ffi], so the
extension needs an explicit transformation rule such as
[`custom_vjp`][jax-custom-vjp].

During the reverse pass, PM++ reconstructs the canonical forward route and its
provenance. The local transpose is

$$
\bar{\mathbf y}
\xrightarrow{\text{split by source}}
(\bar{\mathbf y}_S,
 \bar{\mathbf y}_L,
 \bar{\mathbf y}_R)
\xrightarrow{\text{inverse }ppermute}
(\bar{\mathbf x}_S,
 \bar{\mathbf x}_L,
 \bar{\mathbf x}_R)
\xrightarrow{\text{scatter to source slots}}
\bar{\mathbf x}.
$$

The split and final scatter can use the CUDA transpose handlers. The inverse
neighbor exchange remains JAX `ppermute`. This construction differentiates the
same canonical route regardless of whether its shard-local forward operations
were evaluated by JAX or CUDA.

## Qualification and fallback

CUDA routing is selected only when all static ABI conditions hold. The active
implementation checks:

- a qualified JAX 0.6 typed-FFI line
- a CUDA JAX backend
- float32 or float64 particle payloads, with dtype-matched classification
- 16- or 32-bit `pmid`, converted to int32 at the FFI boundary
- at least two logical devices
- `mesh_halo` mode
- a mesh raveled-key space that fits in `uint32`
- a loadable library with a compatible record-format manifest.

If any condition fails, configuration resolves `cuda_routing` to false and the
canonical JAX route remains active. Importing PM++ never requires the extension
or loads it unconditionally.

## Build artifact and manifest

The wheel ships the CUDA sources but no compiled library. The installed command
uses the active JAX environment, detects visible GPU architectures, and copies
only the library and manifest into the package or versioned user cache:

```bash
pmpp-build-cuda-routing
```

The command invokes the packaged `cuda/build_cuda_routing.py` in a temporary
directory. That script configures `cuda/CMakeLists.txt` and writes
`libpmpp_cuda_routing.so` plus `pmpp_cuda_routing.manifest.json`. The manifest
records the ABI record version, registered targets, PM++ and JAX versions,
embedded CUDA architectures, source revision when available, and artifact
hash. Developers can still call the lower-level script directly from a source
checkout.

PM++ searches the source build directory and the package-local `pmpp/_cuda`
directory, followed by the versioned user cache. A different artifact can be
selected explicitly:

```bash
export PMPP_CUDA_ROUTING_LIBRARY=/absolute/path/libpmpp_cuda_routing.so
```

`extension_status()` reports the candidate paths, loaded library, manifest,
target registration, backend, and qualification state without raising when
the optional artifact is absent.

## Implementation anchors

- `cuda/route_kernels.cu`: classification, scans, record writing, stable
  merges, transpose kernels, and typed FFI bindings
- `cuda/CMakeLists.txt`: JAX and CUDA headers and shared-library target
- `cuda/build_cuda_routing.py`: reproducible build and ABI manifest
- `build_cuda_routing.py`: installed command, architecture detection, and
  artifact placement
- `_cuda_paths.py`: shared package-local and user-cache paths
- `cuda_routing.py`: discovery, qualification, registration, and Python FFI
  wrappers
- `halo_moving.py`: canonical route, JAX collectives, capacity checks, and
  custom-adjoint integration

[jax-ffi]: https://docs.jax.dev/en/latest/ffi.html
[jax-ppermute]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html
[jax-custom-vjp]: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
