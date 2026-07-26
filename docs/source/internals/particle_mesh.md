# Particle-mesh force

## Purpose

The PM force maps particle positions to acceleration with two uses of the same
CIC stencil around a spectral Poisson solve. Time dependence is separated into
the kick/drift factors, so the gravity kernel computes the scaled spatial force.

## Force solve

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Particle positions"]:::state --> B["CIC scatter"]:::op --> C["Overdensity source"]:::state --> D["Distributed rFFT"]:::comm --> E["Poisson + spectral gradient"]:::op --> F["Inverse FFT force mesh"]:::comm --> G["CIC gather acceleration"]:::state
```

**Text equivalent:** particles deposit normalized mass with CIC; subtracting
one gives overdensity; a distributed rFFT, Poisson kernel, spectral derivative,
and inverse FFT create force meshes; CIC interpolates them to particles.

## Equations

For grid cell $g$, particle $p$, and CIC weight $W_{gp}$,

$$
\rho_g=\frac{N_\mathrm{mesh\ cells}}{N_\mathrm{particles}}
\sum_p W_{gp}(\mathbf x_p),\qquad
\delta_g=\rho_g-1.
$$

The uncorrected solver forms $S=\tfrac32\Omega_m\delta$ and solves

$$
\hat\phi(\mathbf k)=-\frac{\hat S(\mathbf k)}{k^2},
\qquad \hat\phi(\mathbf 0)=0,
$$

then applies the negative spectral gradient,

$$
\hat a_i(\mathbf k)=-i k_i\hat\phi(\mathbf k),\qquad
a_{p,i}=\sum_g W_{gp}(\mathbf x_p)a_{g,i}.
$$

Nyquist derivatives are zeroed to preserve the Hermitian structure required by
the inverse rFFT. Optional correction objects may replace the continuum
$k^2$ symbol or alter the potential; they are part of the scientific model.

## Shapes and units

- normalized density/source: global real shape `mesh_shape`, mean density one;
- rFFT potential: global $(N_x,N_y,N_z/2+1)$ in the natural transposed spectral
  sharding on multiple devices;
- force meshes: three real fields of global `mesh_shape`;
- particle acceleration: padded logical shape $(N_\mathrm{slots},3)$ in PM++'s
  scaled acceleration convention.

## Implementation anchors

`src/pmpp/scatter.py` and `src/pmpp/gather.py` implement CIC and their custom
VJPs. `src/pmpp/gravity.py` normalizes density, applies the particle-Nyquist
filter when particle and mesh grids differ, solves Poisson, differentiates, and
gathers. Distributed transforms come from `src/pmpp/FFT_distributed.py`.

The gather transpose is scatter-like on mesh cotangents; the scatter transpose
also differentiates the CIC weights with respect to particle displacement. In
`mesh_halo` mode, those VJPs include the transpose of edge exchange/reduction.

## Design trade-offs

- CIC is compact and differentiable but suppresses mesh-scale power; spectrum
  estimators must state/deconvolve the assignment window deliberately.
- Spectral Poisson/gradient operators are accurate and simple on a periodic
  mesh but require global transposes in distributed execution.
- A finer force mesh can resolve more modes but increases FFT memory/traffic;
  modes beyond the particle Nyquist limit are filtered.
- Correction models can improve a calibrated target while adding FFTs, memory,
  parameters, and generalization risk.

## Validation

Check mean density and total normalized mass, the zero mode, serial/distributed
FFT parity on small meshes, translation/periodicity, and focused scatter,
gather, and gravity gradients. A force test is not complete until the selected
runtime mode and halo boundaries are included.
