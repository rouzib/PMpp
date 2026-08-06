# Particle-mesh force

The force operator maps particle displacement to particle acceleration through
a mesh. It uses cloud-in-cell assignment twice: scatter deposits normalized
particle mass, while gather interpolates the force field back to the same
particle positions {cite:p}`hockney1988particles,li2024adjoint`.

```{mermaid}
flowchart LR
  A["Authoritative particles"] --> B["CIC scatter"]
  B --> C["Density contrast"]
  C --> D["Real FFT"]
  D --> E["Poisson solve"]
  E --> F["Three spectral derivatives"]
  F --> G["Inverse real FFT"]
  G --> H["CIC gather"]
  H --> I["Particle acceleration"]
```

## Cloud-in-cell stencil

Let $\ell_m$ be the mesh-cell size. For a particle at $\mathbf x_p$ and a mesh
vertex at $\mathbf x_g$, define the periodic, cell-normalized separation

$$
u_{gp,i}=\frac{x_{p,i}-x_{g,i}}{\ell_m}.
$$

The three-dimensional CIC weight is

$$
W_{gp}=\prod_{i=1}^{3}\max(1-|u_{gp,i}|,0).
$$

Only the two adjacent vertices on each axis have nonzero weight, so a particle
touches $2^3=8$ mesh vertices {cite:p}`hockney1988particles`. Within a stencil
cell,

$$
\frac{\partial W_{gp}}{\partial x_{p,i}}
=-\frac{\operatorname{sign}(u_{gp,i})}{\ell_m}
\prod_{j\ne i}(1-|u_{gp,j}|).
$$

At an exact vertex the implemented sign is zero. The piecewise-linear map is
continuous, while its derivative is defined by this selected subgradient at
the knot.

`enmesh` computes the eight indices, the weights, and the weight derivatives.
Offsets allow the same routine to address a local slab whose physical origin
is not zero. All index construction is periodic before local bounds are
applied.

## Scatter and density normalization

For a particle value $v_p$ and an optional input mesh $M_g^\mathrm{in}$,
scatter computes

$$
M_g=M_g^\mathrm{in}+\sum_p v_pW_{gp}.
$$

The gravity path uses the scalar value

$$
v_p=\frac{N_m}{N_p},
$$

where $N_m$ is the number of mesh cells and $N_p$ is the number of physical
particles. Since the CIC weights for each valid particle sum to one, the
result has mean one {cite:p}`hockney1988particles,li2024adjoint`:

$$
\rho_g=\frac{N_m}{N_p}\sum_pW_{gp},
\qquad
\frac1{N_m}\sum_g\rho_g=1.
$$

The mesh density contrast is therefore

$$
\delta_g=\rho_g-1.
$$

Padding and inactive particle slots have zero value and make no contribution.

## Periodic Poisson solve

PM++ stores the scaled potential used by the cosmological PM formulation
{cite:p}`li2024adjoint`:

$$
\varphi=a\phi,
$$

which satisfies

$$
\nabla^2\varphi
=\frac32\Omega_mH_0^2\delta.
$$

With the default internal time unit, the numerical $H_0^2$ factor is one. The
gravity code therefore forms

$$
S_g=\frac32\Omega_m\delta_g.
$$

For every nonzero Fourier mode,

$$
\widehat\varphi(\mathbf k)
=-\frac{\widehat S(\mathbf k)}{k^2},
\qquad
k^2=k_x^2+k_y^2+k_z^2.
$$

The zero mode is explicitly zero. This fixes the arbitrary additive constant
in the potential and avoids division by zero. The Poisson operator is real,
diagonal, and self-adjoint under the discrete inner product. Its
[custom VJP][jax-custom-vjp] is therefore the same Poisson solve applied to the
potential cotangent {cite:p}`li2024adjoint`.

## Spectral force

The stored acceleration is

$$
\mathbf a=-\nabla\varphi.
$$

Each Fourier component is

$$
\widehat a_i(\mathbf k)
=-ik_i\widehat\varphi(\mathbf k).
$$

On an even grid, a Nyquist mode is its own negative. A first derivative at
that frequency cannot simultaneously retain the ordinary $ik$ value and the
Hermitian symmetry needed for a real inverse transform. PM++ sets the
corresponding derivative component to zero, following the usual Fourier
spectral convention for odd derivatives {cite:p}`trefethen2000spectral`.

When the force mesh is finer than the particle lattice, the density spectrum
also contains modes above the particle Nyquist limit. PM++ applies the
separable mask

$$
|k_i|\leq\frac{\pi}{\ell_p}
\quad\text{for every axis }i
$$

before the Poisson solve. The separable representation avoids constructing a
second dense three-dimensional mask and is compatible with the sharded
spectral layout. PM++ treats modes beyond the particle-lattice resolution as
unresolved and suppresses them with this explicit model cutoff. This is
distinct from the general mass-assignment and FFT aliasing problem discussed
by {cite:t}`sefusatti2016aliasing`.

The three force components are transformed back to real space. A batched
inverse-transform path keeps the component axis unsharded, then a stacked CIC
gather interpolates all three components in one particle pass when the active
distributed layout supports it.

## Gather

For any mesh field $F_g$, gather computes

$$
f_p=\sum_gW_{gp}F_g.
$$

Using the same stencil for deposition and interpolation gives a matched PM
pair {cite:p}`hockney1988particles,li2024adjoint`. It also makes the discrete
transposes particularly direct.

For gather output cotangent $\bar f_p$,

$$
\bar F_g\mathrel{+}=W_{gp}\bar f_p,
$$

$$
\bar x_{p,i}\mathrel{+}=
\bar f_p\sum_gF_g
\frac{\partial W_{gp}}{\partial x_{p,i}}.
$$

The first equation is a scatter. The second contracts the mesh values with the
derivative of the interpolation stencil.

For scatter output cotangent $\bar M_g$,

$$
\bar v_p=\sum_gW_{gp}\bar M_g,
$$

$$
\bar x_{p,i}\mathrel{+}=
v_p\sum_g\bar M_g
\frac{\partial W_{gp}}{\partial x_{p,i}}.
$$

The displacement derivative returned by the implementation divides the
normalized coordinate derivative by the actual mesh-cell size, as required by
the chain rule.

## Reference JAX kernels

The reference CIC implementation materializes the eight neighbors for a chunk
of particles, then uses indexed JAX gather or scatter operations.
[`lax.scan`][jax-scan] can process multiple fixed-size chunks, which bounds the
size of intermediate arrays without changing the operator.

[Custom VJPs][jax-custom-vjp] implement the equations above directly. This
avoids asking JAX to differentiate through index construction and makes the
intended derivative at CIC knots explicit {cite:p}`li2024adjoint`.

## Pallas CIC kernels

When the selected dtype and JAX backend satisfy the Pallas kernel contract,
PM++ can evaluate the same CIC maps with tiled Pallas kernels. A program tile
handles up to 128 particle lanes. The final partial tile is padded, and a
validity mask prevents padded lanes from loading a mesh value, issuing an
atomic update, or writing a gradient. [Pallas][jax-pallas] provides the
JAX-traceable custom-kernel model used here.

The coordinate helper statically unrolls the eight neighbors. Gather performs
eight masked loads per particle and accumulates the weighted result. Scatter
uses atomic additions because multiple particles can contribute to one mesh
cell.

The backward kernels are also explicit:

- gather backward atomically scatters value cotangents to the mesh and writes
  one displacement cotangent per particle
- scatter backward gathers the mesh cotangent to particle values and writes
  the displacement cotangent
- a scalar scatter value uses a tile reduction followed by a scalar atomic
  addition for its cotangent.

The Pallas and reference paths share the same positions, weights, masking,
periodicity, output shapes, and VJP contract. Kernel selection does not change
the PM equations.

## Mesh-halo form of CIC

In `mesh_halo` mode, every device stores authoritative particles for one
x-slab and an owned mesh slab. CIC support can cross a slab boundary, so local
scatter and gather use a mesh with one edge cell from each neighbor. This
follows the neighboring ghost-data pattern of spatial domain decomposition
{cite:p}`plimpton1995domain`. PM++ expresses the per-device work with
[`shard_map`][jax-shard-map].

For scatter:

1. allocate a zero local mesh with left and right halo cells
2. deposit all authoritative local particles into it
3. send the halo-cell contributions to the device that owns those cells
4. add the received values to the owned edge cells.

For gather:

1. copy owned edge cells to the neighboring mesh halos
2. gather locally from the extended mesh.

The copy used by gather and the reduction used by scatter are transposes. Their
custom VJPs apply the opposite operation, so the distributed CIC derivative is
the transpose of the complete distributed forward map.

## Implementation anchors

- `enmesh.py`: periodic neighbor indices, CIC weights, and weight derivatives
- `scatter.py`: reference and distributed scatter plus custom VJPs
- `gather.py`: reference, stacked, and distributed gather plus custom VJPs
- `pallas_cic.py`: tiled forward and backward CIC kernels
- `gravity.py`: density normalization, Nyquist filtering, Poisson solve,
  spectral differentiation, inverse transforms, and force gather
- `mesh_halo.py`: edge copy and edge reduction

[jax-custom-vjp]: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
[jax-scan]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
[jax-pallas]: https://docs.jax.dev/en/latest/pallas/
[jax-shard-map]: https://docs.jax.dev/en/latest/notebooks/shard_map.html
