# Initial conditions

## What you will do

Generate deterministic white noise, convert it to linear modes, and initialize particles with LPT.

## Requirements

CPU JAX is sufficient for small examples.

## Complete code

```python
from pmpp.modes import white_noise, white_noise_nested, linear_modes
from pmpp.lpt import lpt
```

## Step-by-step explanation

`white_noise` creates the random field for one resolution. `linear_modes` applies transfer and growth information from the cosmology. `lpt` converts modes into particle displacements and velocities. `white_noise_nested` is useful when comparing resolutions while preserving large-scale structure.

## Expected output

For a fixed seed and configuration, repeated runs should produce identical arrays.

## Common failures

Changing mesh shape changes mode labels and array layout. Nested modes are deterministic by mode label, not by array index alone.

## Next steps

Continue with [nested white noise](nested_white_noise.md).
