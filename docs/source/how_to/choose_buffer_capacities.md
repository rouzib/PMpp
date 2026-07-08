# Choose buffer capacities

Static-capacity buffers are part of the simulation contract for JIT-compiled multi-GPU PM++ runs. They must be large enough for the worst particle distribution reached by the compiled run, not only for the initial uniform grid.

## Start from the particle count

For an x-slab decomposition with `n_dev` devices and `res**3` particles, a uniform first estimate is:

```python
base = res**3 / n_dev
max_ptcl_per_slice = int(base * 1.5)
```

Use larger safety factors for clustered late-time runs, small device counts, or stress tests with large displacements. A first diagnostic run can deliberately overallocate to learn the peak usage, then production runs can tighten values to save memory.

## Communication capacities

- Increase `max_share_ptcl` when ownership migration overflows during drift.
- Increase `max_halo_share_ptcl` when a particle-halo comparison path overflows.
- Increase `max_share_gather_ptcl` when gather communication overflows.

## Practical workflow

1. Run a small mesh with generous capacities.
2. Increase resolution while keeping a conservative safety factor.
3. Record the capacities, device count, mesh shape, random seed, and final scale factor.
4. Treat any overflow as a failed run and rerun from the beginning with larger capacities.
