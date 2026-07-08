# Particles and ownership

Particle containers are JAX PyTrees that carry identifiers and dynamical fields. Common fields include particle ids (`pmid`), displacements (`disp`), velocities (`vel`), accelerations (`acc`), padding/sentinel entries, and masks such as `halo_mask` when a halo path is active.

In multi-GPU execution, ownership is authoritative: each valid particle belongs to exactly one x-slab after migration has completed. Packing invariants keep valid particles before padding entries, and invalid entries use sentinel keys so sort and merge operations are deterministic under JIT.

The most important invariant for maintainers is that every drift that can move particles across a slab boundary must be followed by the correct migration and canonicalization step before later force or gather operations assume local ownership.
