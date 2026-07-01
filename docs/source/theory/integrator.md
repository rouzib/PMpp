# N-body integrator

The N-body path advances particles with drift, force, and kick operations. At each macro-step PM++ evaluates mesh gravity, updates particle accelerations, and advances positions and velocities between scale-factor times.

The solver is organized around `pmpp.nbody`, with lower-level operations in `pmpp.steps`. Potential corrections can be supplied to force evaluation when the correction objects are explicitly constructed by the caller.
