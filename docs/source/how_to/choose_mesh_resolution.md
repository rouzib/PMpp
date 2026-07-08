# Choose mesh resolution

Mesh resolution controls force resolution, memory footprint, FFT cost, and static buffer pressure. Start from the smallest resolution that tests the scientific question.

## Rules of thumb

- Use `32^3` or smaller for documentation, debugging, and gradient smoke tests.
- Use moderate meshes for validation against PMWD before scaling.
- Ensure the x-resolution is compatible with the number of devices in slab-decomposed multi-GPU runs.
- Increase capacities when higher resolution or stronger clustering increases boundary traffic.

Never use documentation example sizes as evidence of production accuracy; they are chosen for fast compilation and clear diagnostics.
