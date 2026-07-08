# Domain decomposition

PM++ multi-GPU execution decomposes the mesh along the x-axis into slabs. Each GPU owns an x-slab and the particles whose authoritative positions belong to that slab. During drift steps, particles can cross slab boundaries, so PM++ detects crossings, packs particles for exchange, communicates with neighboring slabs, and restores canonical packed order.

Neighbor communication is ring-like through left/right permutations. The current preferred path is `mesh_halo`: particles remain authoritatively owned by slabs, while mesh edge cells are exchanged so local gather operations can see neighboring field values. The older `particle_halo` path remains useful for comparison and legacy experiments.

```{image} ../_static/domain_decomposition.svg
:alt: x-slab domain decomposition with halo cells
```

A multi-GPU run may need larger static capacities when particles cluster near slab boundaries or when halo gather traffic increases.
