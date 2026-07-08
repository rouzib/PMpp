# Static-capacity buffers

JAX compilation works best with fixed shapes. PM++ therefore overallocates arrays for particle slots and communication buffers. A capacity overflow is a correctness failure: the simulation attempted to store or communicate more entries than the compiled buffer can hold.

Important user-facing capacities include:

- `max_ptcl_per_slice`: maximum particle slots allocated for an owning slab.
- `max_share_ptcl`: maximum particles exchanged during ownership migration.
- `max_halo_share_ptcl`: maximum particles or halo records exchanged for halo-style communication.
- `max_share_gather_ptcl`: maximum records used by multi-GPU gather communication.

For small examples, start with generous values, verify memory use, then reduce only after profiling. If an overflow occurs, increase the named capacity and rerun. Oversized buffers consume memory, so production runs should record the capacities used for each resolution and device count.
