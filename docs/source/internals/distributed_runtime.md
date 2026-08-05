# Distributed runtime

PM++ decomposes the periodic box into slabs along the global x-axis. Particles
are owned by slabs, real meshes are sharded by x, and Fourier transforms
temporarily move the sharding to y. The runtime keeps these representation
changes separate from the physical simulation state.

## Logical device topology

A one-dimensional JAX `Mesh` named `gpus` defines the logical device order.
Topology arrays use positions in that logical mesh, not physical device IDs.
For $P$ devices and a mesh with $N_x$ cells, logical device $r$ owns

$$
x\in[s_r,e_r),
\qquad
s_r=r\frac{N_x}{P},
\qquad
e_r=(r+1)\frac{N_x}{P},
$$

with periodic interpretation at the box boundary. The local owned mesh shape
is

$$
\left(\frac{N_x}{P},N_y,N_z\right).
$$

Left and right ring permutations are built once from the logical mesh. All
neighbor communication uses these permutations through `lax.ppermute`.

## Authoritative particle ownership

The wrapped x-coordinate in mesh-cell units is

$$
\xi_p=\left(m_{p,x}+\frac{d_{p,x}}{\ell_m}\right)\bmod N_x.
$$

A particle is authoritative on device $r$ exactly when

$$
\xi_p\in[s_r,e_r).
$$

The interval test treats $[s,e)$ normally when $s\leq e$ and as
$[s,N_x)\cup[0,e)$ when it wraps. The lower bound is inclusive and the upper
bound is exclusive, so every valid coordinate has one owner.

`mesh_halo` stores only these authoritative particles. Mesh boundary cells are
exchanged for CIC, but particles are not duplicated for the force calculation.
Particle ownership can still change after a drift.

## Static particle buffers

JAX compilation requires fixed shapes. Each device therefore stores
`max_ptcl_per_slice` slots even though its active particle count changes.
`unused_index` marks the inactive suffix.

The routing operation also uses fixed-capacity streams:

- `max_share_ptcl` bounds particles migrating in either direction during one
  drift
- `max_halo_share_ptcl` bounds duplicated particle halos in the legacy
  particle-halo path
- `max_share_gather_ptcl` bounds particle-value exchanges required by that
  path's gather transpose.

The capacities limit representation, not physics. PM++ checks uncapped counts
before a capped stream is accepted. Exceeding a capacity invalidates the step
and is reported as an error.

## Canonical particle order

Every valid authoritative particle has the periodic raveled mesh key

$$
\kappa_p=
\left((m_{p,x}\bmod N_x)N_y+(m_{p,y}\bmod N_y)\right)N_z
+(m_{p,z}\bmod N_z).
$$

Valid slots are stored in nondecreasing $\kappa_p$ order. Invalid slots use a
sentinel key equal to $N_m$, which sorts after every real cell key.

This order serves three purposes:

1. an outgoing mask can be compacted without changing relative order
2. incoming and staying streams can be merged stably
3. route transposes can use saved source ranks rather than a particle hash
   table.

Particles may share a mesh anchor, so keys are not unique. Stable tie order is
part of the representation contract. The JAX route preserves the stay stream
before equal-key incoming streams. The bidirectional native merge uses the
explicit order `stay`, `left`, then `right`.

## Migration after a drift

After updating displacement, PM++ classifies every valid particle as one of:

- **stay**, if its new coordinate is still in the owned slab
- **send left**, if it moved into the immediate left slab
- **send right**, if it moved into the immediate right slab
- **outside the one-hop domain**, which is an error.

For two logical devices, left and right refer to the same neighbor, so the
implementation uses one stream to avoid sending a particle twice.

The route is

$$
\text{canonical input}
\xrightarrow{\text{classify and compact}}
(S,L,R)
\xrightarrow{\text{ring exchange}}
(S,L_\mathrm{in},R_\mathrm{in})
\xrightarrow{\text{stable merge}}
\text{canonical output}.
$$

The compact outgoing streams have shape `max_share_ptcl`. Counts remain
uncapped scalars so capacity checks can detect truncation. JAX `ppermute`
transfers the packed metadata and floating payloads to the neighboring shard.

The stable merge does not need to materialize a second full-capacity stay
payload. It treats outgoing locations as holes in the already sorted input,
maps compact stay ranks back to their source slots, and merges them with the
small incoming streams by key.

## Route transpose

Routing is a permutation, compaction, communication, and merge of floating
particle fields. Its transpose must reverse all four operations.

The forward merge records, or can reconstruct, a source tag and source rank for
each valid output. The transpose then:

1. splits the merged cotangent into stay, incoming-left, and incoming-right
   cotangent streams
2. applies the inverse ring exchanges
3. scatters each returned cotangent into the source slot selected during
   compaction
4. leaves invalid input slots at zero.

Integer `pmid`, masks, and source indices define the map but do not receive
ordinary floating cotangents. Displacement, velocity, and acceleration payloads
do.

## Mesh halos

An owned real mesh slab is extended by $h$ cells on both x edges:

$$
\left(\frac{N_x}{P}+2h,N_y,N_z\right).
$$

The ordinary CIC path uses $h=1$, which is sufficient because CIC reaches only
one neighboring grid point on either side of a particle. A larger static halo
is available for operators whose known support is wider, provided it fits in
one neighbor slab.

There are two complementary edge operations.

### Copy owned edges to halos

For interpolation, each device sends its left owned edge to its left neighbor's
right halo and its right owned edge to its right neighbor's left halo. This
creates a read-only extended mesh containing the values needed by local
particles.

### Reduce halo contributions to owners

For deposition, local particles may write into halo cells. Each device sends
those accumulated halo values to the neighbor that owns the corresponding
cells, where they are added to the owned edge.

If $C$ denotes edge copy and $R$ denotes edge reduction, then under the usual
array inner product

$$
C^\mathsf T=R,
\qquad
R^\mathsf T=C.
$$

The scatter and gather VJPs use this relationship. As a result, halo
communication is included in the derivative of the distributed operator.

## Two-pass distributed FFT

The real-space input has global shape $(N_x,N_y,N_z)$ and sharding

```text
P("gpus", None, None)
```

so each device holds a complete y-z plane for its local x range. This permits a
local rFFT over y and z:

$$
f(x,y,z)
\xrightarrow{\mathrm{rFFT}_{y,z}}
F(x,k_y,k_z).
$$

To transform x, the array is resharded to

```text
P(None, "gpus", None)
```

Now each device has the full x-axis for its local $k_y$ range. A local complex
FFT over x completes the transform:

$$
F(x,k_y,k_z)
\xrightarrow{\mathrm{FFT}_x}
\widehat f(k_x,k_y,k_z).
$$

The resharding between the two layouts lowers to the required global
transpose. PM++ retains the natural y-sharded spectral layout for the Poisson
and gradient operations. It does not transpose back merely to make the Fourier
array resemble the real-space layout.

The inverse transform reverses these stages:

$$
\widehat f(k_x,k_y,k_z)
\xrightarrow{\mathrm{iFFT}_x}
F(x,k_y,k_z)
\xrightarrow{\text{layout transpose}}
F(x,k_y,k_z)
\xrightarrow{\mathrm{irFFT}_{y,z}}
f(x,y,z).
$$

`custom_partitioning` marks each pass as local in its declared layout. Named
input and output shardings make the collective transpose explicit to JAX.

## Real FFT adjoints

An rFFT stores only the nonnegative frequencies of its final transformed axis.
Its adjoint is not an ordinary irFFT call with no adjustment.

For the forward rFFT transpose, PM++ pads the half-spectrum cotangent to the
full real input shape, conjugates it, applies a complex inverse FFT, takes the
real part, and multiplies by the total real-grid size. This matches JAX's
unnormalized forward transform.

For an inverse rFFT with real output shape $\mathbf N$, the transpose first
computes the conjugated rFFT of the real cotangent. It then applies the
Hermitian multiplicity

$$
w(k_z)=
\begin{cases}
1,&k_z=0,\\
1,&k_z=N_z/2\text{ on an even grid},\\
2,&\text{otherwise},
\end{cases}
$$

and divides by $\prod_iN_i$. Interior positive frequencies receive weight two
because their omitted negative-frequency partners carry the same real-field
degree of freedom.

The batched inverse real FFT uses the same rule while leaving the leading
component axis unsharded.

## Reverse reconstruction across ownership changes

The N-body adjoint begins a reverse drift with only the post-drift canonical
state. For a drift factor $D$, it first forms the pre-drift physical
displacement candidate

$$
\mathbf d_\mathrm{in}=\mathbf d_\mathrm{out}-D\mathbf p.
$$

Those candidates are routed according to their reconstructed pre-drift
positions, which recovers the exact authoritative layout that entered the
forward drift. PM++ then rebuilds the forward route from that state and applies
its transpose to the arriving cotangents. A fused implementation performs the
reconstruction and route pullback together, but the mathematical map is the
same sequence.

## Implementation anchors

- `multigpu_configuration.py`: slab geometry, ring permutations, capacities,
  and compiled runtime binding
- `halo_moving.py`: canonical compaction, migration, stable merge,
  reconstruction, and route transpose
- `mesh_halo.py`: mesh edge copy and reduction
- `FFT_distributed.py`: two-pass transforms, partitioning rules, transposed
  spectral layout, and real-FFT custom VJPs
- `utils.py`: ring permutations and periodic raveled keys
