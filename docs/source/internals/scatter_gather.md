# Scatter and gather

Scatter deposits particle mass to the mesh, usually with CIC assignment. Gather interpolates mesh forces back to particle positions. PM++ has local paths for serial runs and distributed paths for multi-GPU decompositions.

`mesh_halo` gather extends owned mesh slabs with neighboring halo cells so interpolation near slab boundaries can remain local after exchange. The legacy particle-halo path instead communicates particle records for comparison and older workflows.

Adjoint behavior matters: scatter and gather transposes must send cotangents back to the correct particles or mesh cells, including across halo boundaries. Internals pages may name private transpose helpers, but users should rely on public scatter/gather functions unless developing PM++ itself.
