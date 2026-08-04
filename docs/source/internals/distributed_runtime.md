# Distributed runtime

## Purpose

The runtime maps a global periodic particle-mesh calculation onto an ordered
one-dimensional device mesh. Equal x slabs define authoritative particle
ownership and real-space mesh sharding; neighbor exchange supports local CIC,
while a collective transpose supports global FFTs.

The four diagrams below use the shared legend from [How PM++ works](index.md).

## Slab ownership and mesh halos

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Global periodic x axis"]:::state --> B["Equal owned slabs"]:::comm --> C["Authoritative particles"]:::state --> D["Owned density mesh"]:::state --> E["Exchange edge cells"]:::comm --> F["Local CIC with halos"]:::op
```

**Text equivalent:** split the periodic x axis into equal owned slabs; each
holds authoritative particles and density cells; exchange neighboring mesh edge
cells so CIC scatter/gather is local near a boundary.

For $D$ devices and global x extent $N_x$, logical device $d$ owns
$[dN_x/D,(d+1)N_x/D)$. `mesh_halo` stores no duplicated particle halo records;
`store_particle_halos` is false. When $D>1$, the mesh-halo width is at least one
cell (or the configured static width); when $D=1$, particle and mesh halo widths
are both zero.

## Particle crossing and migration

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Drifted positions"]:::state --> B["Classify owner"]:::op --> C["Pack left / stay / right"]:::comm --> D["Ring ppermute"]:::comm --> E["Stable merge + pad"]:::op --> F["Canonical slots and masks"]:::state
```

**Text equivalent:** after drift, classify every valid particle by its new slab;
pack left, staying, and right streams; exchange neighbor streams around the ring;
stably merge them into fixed-capacity storage and rebuild masks.

Movement supports staying on the same slab or crossing to an immediate neighbor
in one drift. A larger exchange buffer cannot make a multi-slab crossing valid;
the schedule must keep per-step movement inside that routing envelope. Reverse
movement uses saved/reconstructed routing logic to return cotangents to the
previous authoritative records.

## Distributed FFT transpose

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Real mesh: x sharded"]:::state --> B["Local y-z rFFT"]:::op --> C["Collective corner turn"]:::comm --> D["Spectral mesh: y sharded"]:::state --> E["Local x FFT"]:::op --> F["Transposed spectrum"]:::state
```

**Text equivalent:** transform local y-z planes in each x slab; a collective
layout change makes the x direction local and y sharded; transform x and keep
that natural transposed spectral layout for Poisson/gradient work.

The forward real shape is $(N_x,N_y,N_z)$. The rFFT logical shape is
$(N_x,N_y,N_z/2+1)$, sharded as `P(None, "gpus", None)` in the transposed
layout. The inverse performs the complementary operations and returns
`P("gpus", None, None)`. Custom VJPs include rFFT Hermitian weights and
normalization; a local FFT on each x slab is not a global transform.

## Static particle slots and buffers

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Logical local particles"]:::state --> B["Per-slice fixed array"]:::state
  B --> C["Unused padded slots"]:::state
  B --> D["Migration buffer"]:::comm
  B --> E["Halo rebuild buffer"]:::comm
  B --> F["Gather-value buffer"]:::comm
```

**Text equivalent:** logical local particles occupy a fixed per-slice array;
unused slots preserve its shape, while separate fixed buffers bound migration,
halo rebuild, and gathered-value communication.

`max_ptcl_per_slice`, `max_share_ptcl`, `max_halo_share_ptcl`, and
`max_share_gather_ptcl` size those arrays. Compaction uses fixed `size=` values;
when a capacity predicate is violated, PM++ raises through a host callback
instead of accepting the truncated operation. This is why capacities remain
correctness constraints even though the failure is now fail-fast.

## Equations, shapes, and units

The storage position is reconstructed from an integer mesh cell and physical
displacement,

$$
\mathbf x_p=\Delta x\,\mathbf i_p+\mathbf d_p \pmod{\mathbf L}.
$$

Real meshes have global shape $(N_x,N_y,N_z)$ and local owned shape
$(N_x/D,N_y,N_z)$. Particle fields use fixed local leading dimension
`max_ptcl_per_slice`; communication arrays replace that dimension with their
named capacity. Slab bounds and halo offsets are stored in mesh-cell units,
while `Particles.disp`, `scatter_offsets`, and `mesh_halo_offsets` are converted
with `conf.cell_size` to the configured physical length unit.

## Implementation anchors

`src/pmpp/multigpu_configuration.py` derives logical device order, slab ranges,
halo offsets, permutations, and effective capacities. `src/pmpp/particles.py`
and `src/pmpp/halo_moving.py` implement canonical routing. `src/pmpp/mesh_halo.py`
implements neighbor edge exchange/reduction. `src/pmpp/FFT_distributed.py`
constructs transposed distributed transforms and their custom VJPs.

## Design trade-offs

- Authoritative-only particles remove duplicate particle work in `mesh_halo`,
  but every ownership-changing drift needs migration.
- One-dimensional slabs keep neighbor topology simple, but FFTs require a
  global corner turn and mesh x/y dimensions must divide the device count
  (particle generation additionally requires particle-grid x divisibility).
- Fixed buffers make JAX shapes stable, but memory is reserved for capacity and
  bad estimates cannot grow dynamically.
- Keeping the transposed spectrum avoids an extra layout bounce around each
  spectral operator, but extensions must respect which axis is sharded.

## Validation

Check ownership uniqueness, slot/mask packing, periodic neighbor permutations,
mass before/after migration, scatter/gather boundary parity, distributed FFT
round trips, and custom-VJP gradients. A production run must complete with zero
capacity errors and should inspect y/z projections where the global x slab
boundaries remain visible.
