# Observers and density projections

## What you will do

Use forward-only observers to save diagnostics such as density projections during an N-body run.

## Requirements

A working serial PM++ environment.

## Complete code

```python
from pmpp.nbody import nbody_collect, nbody_observe
from pmpp.nbody_observers import density_projection_observer

observer = density_projection_observer(axis=0, normalize=True)
```

## Step-by-step explanation

`nbody_observe` and `nbody_collect` are diagnostic interfaces. Keep observers separate from adjoint workflows unless a specific differentiable observer path is documented.

## Expected output

A projection observer should return finite two-dimensional maps with the selected projection axis removed.

## Common failures

Observer output shapes depend on the selected axis and mesh shape.

## Next steps

Use projections as lightweight checks before full power-spectrum validation.
