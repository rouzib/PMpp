# Particle-Mesh method

PM++ represents matter with particles and computes long-range gravity on a mesh. The Particle-Mesh cycle is:

1. Store particle state on a Lagrangian grid with displacements, velocities, and accelerations.
2. Scatter particle mass to mesh cells with cloud-in-cell (CIC) assignment.
3. Solve Poisson's equation in Fourier space.
4. Take spectral derivatives to form forces.
5. Gather forces from mesh cells back to particles.
6. Advance the system with drift and kick updates.

For density contrast $\delta = \rho/\bar\rho - 1$, the Fourier-space solve uses a kernel proportional to

$$
\phi(\mathbf{k}) \propto -\frac{\delta(\mathbf{k})}{|\mathbf{k}|^2},
$$

with implementation details for zero modes, filters, and corrections handled by the gravity code. This avoids direct pair-force scaling: the dominant mesh solve is $O(N\log N)$ because of FFTs rather than $O(N^2)$ pair interactions.

```{image} ../_static/pm_pipeline.svg
:alt: PM++ pipeline from white noise to density products
```

Implementation anchors: {mod}`pmpp.scatter`, {mod}`pmpp.gather`, {mod}`pmpp.gravity`, and {mod}`pmpp.steps`.
