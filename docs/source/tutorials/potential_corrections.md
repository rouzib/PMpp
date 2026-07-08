# Potential corrections

## What you will do

Inspect PM++ potential-correction families and learn where to attach them in force workflows.

## Requirements

Install the optional dependencies required by the chosen correction model. Some correction families are experimental.

## Complete code

```python
from pmpp import corrections
```

## Step-by-step explanation

Correction implementations live under `pmpp.corrections`. Current families include radial/window-style utilities, softening-related corrections, particle-grid-deconvolution helpers, mesh CNN components, and combined correction wrappers when their optional dependencies are available.

## Expected output

Correction objects should be initialized explicitly and validated against a known reference before being used in scientific analysis.

## Common failures

Optional neural-network dependencies may not be installed. Do not assume correction models improve physical accuracy without validation.

## Next steps

See [potential-correction internals](../internals/potential_corrections.md).
