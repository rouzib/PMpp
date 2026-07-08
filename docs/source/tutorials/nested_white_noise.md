# Nested white noise

## What you will do

Compare white-noise fields across resolutions while preserving overlapping large-scale modes.

## Requirements

CPU JAX is sufficient for a small demonstration.

## Complete code

```python
from pmpp.modes import white_noise_nested
```

## Step-by-step explanation

Nested white noise is useful for resolution studies and variable particle counts at fixed large-scale structure. The important invariant is deterministic mode labeling: consistency is defined by overlapping physical Fourier modes, not by raw array indices alone.

## Expected output

Overlapping low-k modes should agree within numerical precision when configurations are compatible.

## Common failures

Comparing raw arrays across different spectral layouts can give misleading results.

## Next steps

Use nested initial conditions in validation and convergence studies.
