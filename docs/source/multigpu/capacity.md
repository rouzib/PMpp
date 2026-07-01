# Static capacity parameters

JAX compilation favors static shapes. PM++ therefore uses capacity parameters such as maximum particles per slice and maximum shared or halo particles to allocate fixed-size buffers for distributed particle exchange.

These are not tuning hints. If a run exceeds capacity, data needed for a correct simulation would be lost or misrepresented, so overflow is a correctness failure. Increase capacities and rerun when overflow diagnostics appear.
