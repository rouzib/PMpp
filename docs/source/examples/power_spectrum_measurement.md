# Power spectrum measurement

A power-spectrum workflow should separate simulation, density deposition, and measurement diagnostics.

```python
# After a serial or multi-GPU run:
from pmpp.scatter import scatter
from pmpp import power_spectrum

density = scatter(ptcl_final, conf)
# Use the public helper appropriate for the current power_spectrum module API.
print("density mean", float(density.mean()))
print("density finite", bool((density == density).all()))
```

Record the mesh shape, box size, assignment scheme, interlacing/filter settings, and binning choices next to any plotted `P(k)`. A compact diagnostic should include the first few finite bins and a note about zero-mode handling.
