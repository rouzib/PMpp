# Cosmology and units

`Configuration` stores the mesh spacing, particle grid shape, runtime dtype, and derived arrays used by transfer, growth, LPT, and N-body routines. Cosmology objects such as `SimpleLCDM` provide parameters consumed by `boltzmann`, growth utilities, and the integrator.

Use one consistent convention for box size, mesh spacing, scale factor `a`, and redshift `z` within a workflow. Small examples in these docs choose compact boxes and low resolution for fast compilation rather than production accuracy.
