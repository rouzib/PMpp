# Debug capacity overflow

A capacity overflow means PM++ attempted to place more particles or communication records into a fixed-shape buffer than the buffer can hold. This is a correctness failure, not a warning to ignore.

## Identify the buffer

Use the name in the error or diagnostic output:

- `max_ptcl_per_slice`: owned particle storage for an x-slab.
- `max_share_ptcl`: particles migrating across ownership boundaries.
- `max_halo_share_ptcl`: halo exchange records.
- `max_share_gather_ptcl`: gather communication records.

## Fix

Increase the named capacity and rerun with the same seed and simulation settings. Because capacities are static arguments for compiled shapes, changing them may trigger recompilation.

## Avoid repeated failures

- Start multi-GPU tutorials with low resolution.
- Keep `mesh_halo` as the default serious multi-GPU path.
- Check that the x-resolution is compatible with the device count.
- Record successful capacities next to validation results.
