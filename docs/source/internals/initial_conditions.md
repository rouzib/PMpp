# Initial modes and LPT

## Purpose

Initial conditions separate stochastic phases/amplitudes from cosmology and
from the deterministic LPT map. This makes resolution studies and gradients
with respect to either modes or cosmological parameters explicit.

## Initial-mode and nested-noise flow

```{mermaid}
flowchart LR
  classDef state fill:#E8F5E9,stroke:#2E7D32,color:#111;
  classDef op fill:#FFF3CD,stroke:#B7791F,color:#111;
  classDef comm fill:#E8EAFB,stroke:#3949AB,color:#111;
  classDef check fill:#FCE8E6,stroke:#C62828,color:#111;
  A["Seed + signed k labels"]:::comm --> B["Mode-local hash"]:::op --> C["Hermitian white modes"]:::state --> E["Linear density modes"]:::state --> F["1LPT / 2LPT solve"]:::op --> G["Owned particle state"]:::state
  D["Transfer + growth"]:::op --> E
```

**Text equivalent:** signed Fourier labels and a seed feed a deterministic hash;
the coefficients are made Hermitian; the linear spectrum from transfer/growth
scales them; 1LPT or 2LPT produces an ownership-ready particle state.

For ordinary `white_noise`, the first two boxes are replaced by one sequential
real-space Gaussian draw and rFFT. `white_noise_nested` uses the diagrammed
mode-local path.

## Equations

The linear field is

$$
\delta_\mathrm{lin}(\mathbf k)
= \sqrt{V P_\mathrm{lin}(k;\theta)}\,\omega(\mathbf k),
$$

with the rFFT Hermitian condition

$$
\omega(-\mathbf k)=\omega(\mathbf k)^*.
$$

For a fixed box and seed, nested modes satisfy

$$
\omega_{N_c}(\mathbf n)=\omega_{N_f}(\mathbf n)
$$

for signed integer labels $\mathbf n$ shared by coarse and fine grids, excluding
the coarse Nyquist planes. This is exact coefficient identity, not equality of
the different-resolution real-space fields.

Particles begin at Lagrangian grid coordinates $\mathbf q$ and

$$
\mathbf x(a)=\mathbf q+D_1(a)\,\mathbf s^{(1)}
+D_2(a)\,\mathbf s^{(2)},
$$

where $\mathbf s^{(1)}=-\nabla\phi^{(1)}$ and
$\nabla^2\phi^{(1)}=\delta_\mathrm{lin}$. The second-order source is the
quadratic invariant of the first-order strain tensor, implemented by `_L` and
`_strain`. Canonical velocity uses the corresponding growth derivatives.

## Shapes and units

- real noise: `ptcl_grid_shape`, dimensionless with unit variance;
- local Fourier noise: $(N_x,N_y,N_z/2+1)$; distributed Fourier noise uses the
  same global shape with y sharding after the transpose;
- Fourier `linear_modes`: linear overdensity modes carrying the configured
  volume normalization; `real=True` returns dimensionless overdensity;
- `Particles.disp`: $(C_\mathrm{slice}D,3)$ physical displacement in the length
  unit, including padded slots in distributed storage.

## Implementation anchors

`src/pmpp/modes.py` contains ordinary noise, signed-label hashing, Hermitian
canonicalization, nested noise, and linear scaling. `src/pmpp/boltzmann.py`
provides transfer/growth tables and the linear power. `src/pmpp/lpt.py` builds
the first- and second-order potentials and routes the output through the active
ownership logic.

## Design trade-offs

- Mode-local hashing costs more integer work than a sequential draw but makes
  shared physical modes independent of array length.
- Unit-modulus noise removes amplitude variance and changes the ensemble; it is
  useful only when that choice is scientifically intended.
- Cached 2LPT strains reduce inverse FFT count but increase live memory.
- LPT orders 0, 1, and 2 operate. Order 3 is explicitly unfinished and raises
  `NotImplementedError` despite passing configuration validation.

## Validation

Test ordinary determinism by repeating the same seed/configuration. Test nested
noise by signed-index mapping and exact equality after removing coarse Nyquist
planes, matching `tests/test_nested_white_noise.py`. Validate LPT with finite
fields, ownership/capacity invariants, and focused order-1/order-2 gradient tests.
