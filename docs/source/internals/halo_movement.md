# Halo movement

Particle movement across slab boundaries is handled explicitly. The runtime detects particles that crossed ownership boundaries, packs outgoing records, exchanges with left/right neighbors, merges incoming particles, and restores canonical order with sentinel padding.

Important invariants:

- valid particles are packed before padding;
- invalid entries use sentinel keys;
- ownership after drift is canonical;
- buffer capacities are large enough for worst-case exchange;
- the transpose of movement sends cotangents back through the same logical ownership transfer.

Violating these invariants can produce wrong forces or wrong gradients even if array shapes still compile.
