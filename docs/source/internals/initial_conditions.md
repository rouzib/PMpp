# Initial modes and LPT

PM++ constructs initial particle positions and canonical velocities from three
inputs: cosmological parameters $\boldsymbol\theta$, a white-noise realization
$\boldsymbol\omega$, and a periodic particle lattice. The construction keeps
the random realization separate from the cosmology, which allows either input
to be differentiated or replaced independently.

## Background expansion

The dimensionless expansion rate is

$$
E^2(a)=\Omega_m a^{-3}+\Omega_k a^{-2}
+\Omega_\mathrm{de}a^{-3(1+w_0+w_a)}e^{-3w_a(1-a)}.
$$

PM++ tabulates the first- and second-order growth functions on
`conf.growth_a`. To make the early-time solutions slowly varying, the ODE is
written for

$$
G_m(a)=\frac{D_m(a)}{a^m},
$$

where primes below mean $d/d\ln a$. Defining
$h'=d\ln H/d\ln a$, the implemented equations are

$$
G_1''=-\left(3+h'-\frac32\Omega_m(a)\right)G_1
-\left(4+h'\right)G_1',
$$

$$
G_2''=\frac32\Omega_m(a)G_1^2
-\left(8+2h'-\frac32\Omega_m(a)\right)G_2
-\left(6+h'\right)G_2'.
$$

The matter-dominated initial conditions are

$$
G_1=1,\quad G_1'=0,
\qquad
G_2=\frac37,\quad G_2'=0.
$$

After integration, PM++ reconstructs the derivatives needed by LPT and the
time integrator:

$$
D_m=a^mG_m,
$$

$$
D_m'=a^m(mG_m+G_m'),
\qquad
D_m''=a^m(m^2G_m+2mG_m'+G_m'').
$$

Growth values between table entries use a custom differentiable linear
interpolator. At a knot its JVP averages the slopes on the two adjacent
intervals. Outside the tabulated interval the extrapolation slope is zero.

## Transfer function and linear power

`boltzmann` builds an Eisenstein-Hu transfer-function table, with either its
baryonic-wiggle or no-wiggle analytic fit. Despite its historical function
name, this path does not numerically solve the Einstein-Boltzmann hierarchy.
It evaluates the fit, integrates the growth ODE, and builds the linear-variance
table used by `sigma8` and related helpers.

For pivot scale $k_\mathrm{pivot}$, scalar amplitude $A_s$, and tilt $n_s$,
the implemented linear matter spectrum satisfies

$$
\frac{k^3P_\mathrm{lin}(k,a)}{2\pi^2}
=\frac4{25}A_s
\left(\frac{k}{k_\mathrm{pivot}}\right)^{n_s-1}
T^2(k)
\left(\frac{ck}{H_0}\right)^4
\left(\frac{D_1(a)}{\Omega_m}\right)^2.
$$

When no scale factor is passed, the $D_1^2(a)$ factor is omitted. A safe power
and a safe square-root VJP set the derivative to zero at exact zero modes. This
avoids undefined intermediate derivatives without changing nonzero modes.

## Ordinary white noise

`white_noise` starts from independent standard-normal samples on the real
particle lattice. It applies an orthonormal rFFT, so its stored Fourier
coefficients have the normalization expected by `linear_modes`:

$$
\omega_\mathbf{k}=\frac{1}{\sqrt{N_p}}
\sum_\mathbf{q}\omega_\mathbf{q}
e^{-i\mathbf{k}\cdot\mathbf{q}}.
$$

Because the source field is real, the full spectrum obeys

$$
\omega(-\mathbf k)=\omega(\mathbf k)^*.
$$

The optional unit-amplitude mode divides every stored coefficient by its
magnitude. It preserves phases but changes the Gaussian ensemble, so it is a
modeling choice rather than another representation of the same realization.

## Resolution-nested white noise

`white_noise_nested` assigns a random value to each signed integer Fourier
label $(n_x,n_y,n_z)$ instead of consuming a sequential random stream. The
seed, the three labels, and an independent stream salt are mixed into unsigned
32-bit hashes. Two hashes are converted to Gaussian variates with a Box-Muller
map.

The rFFT stores only nonnegative $n_z$. On the self-conjugate planes $n_z=0$
and, for even grids, $n_z=N_z/2$, PM++ chooses one canonical representative of
each $(n_x,n_y)\leftrightarrow(-n_x,-n_y)$ pair. The partner is its complex
conjugate, and points that are their own partner have exactly zero imaginary
part.

Consequently, for two resolutions with the same periodic box and seed,

$$
\omega_{N_c}(\mathbf n)=\omega_{N_f}(\mathbf n)
$$

for every shared non-Nyquist integer mode. The corresponding real-space arrays
are not identical because the fine grid also contains additional modes.

## Scaling white noise to density modes

For a periodic volume $V_\mathrm{box}$, PM++ constructs

$$
\widehat\delta_\mathrm{lin}(\mathbf k,a)
=\sqrt{V_\mathrm{box}P_\mathrm{lin}(k,a)}\,omega(\mathbf k).
$$

This is the discrete form of

$$
\left\langle
\delta(\mathbf k)\delta(\mathbf k')
\right\rangle
=V_\mathrm{box}P_\mathrm{lin}(k)
\delta^K_{\mathbf k,-\mathbf k'}.
$$

`linear_modes` accepts either the real white field or its already transformed
coefficients. In a distributed run it constrains the result to the transposed
spectral layout used by the distributed FFT.

## First-order LPT

Particles begin on Lagrangian lattice positions $\mathbf q$. The first-order
potential and displacement field are

$$
\nabla_\mathbf q^2\phi^{(1)}=\delta_\mathrm{lin},
\qquad
\mathbf s^{(1)}=-\nabla_\mathbf q\phi^{(1)}.
$$

In Fourier space, PM++ uses

$$
\widehat\phi^{(1)}(\mathbf k)
=-\frac{\widehat\delta_\mathrm{lin}(\mathbf k)}{k^2},
\qquad
\widehat{\mathbf s}^{(1)}=-i\mathbf k\widehat\phi^{(1)},
$$

with the zero mode set to zero. `linear_modes` carries the periodic-volume
normalization, so `lpt` first divides by the particle-cell volume before the
discrete Poisson solve.

## Second-order LPT

The first-order strain tensor is

$$
\phi^{(1)}_{,ij}
=\frac{\partial^2\phi^{(1)}}{\partial q_i\partial q_j},
\qquad
\widehat{\phi^{(1)}_{,ij}}
=-k_i k_j\widehat\phi^{(1)}.
$$

The second-order source is the quadratic invariant

$$
L^{(2)}(\mathbf q)=
\sum_{i<j}
\left(
\phi^{(1)}_{,ii}\phi^{(1)}_{,jj}
-\phi^{(1)}_{,ij}\phi^{(1)}_{,ji}
\right).
$$

PM++ transforms this real source back to Fourier space and solves

$$
\nabla_\mathbf q^2\phi^{(2)}=L^{(2)},
\qquad
\mathbf s^{(2)}=-\nabla_\mathbf q\phi^{(2)}.
$$

Off-diagonal strain terms zero the particle-grid Nyquist component before
multiplication. This preserves a real, unambiguous spectral derivative on the
self-conjugate plane. Diagonal second derivatives retain their Nyquist terms.

The strain diagonals may be cached and reused while forming $L^{(2)}$. Turning
off this cache recomputes some inverse FFTs while keeping fewer strain arrays
live. Both paths evaluate the same source.

## Particle state at the starting scale factor

At $a=a_\mathrm{start}$, PM++ sums the requested LPT orders:

$$
\mathbf x(\mathbf q,a)
=\mathbf q+\sum_{m=1}^{2}D_m(a)\mathbf s^{(m)}(\mathbf q),
$$

$$
\mathbf p(\mathbf q,a)
=a^2H(a)\sum_{m=1}^{2}D_m'(a)\mathbf s^{(m)}(\mathbf q).
$$

Here $\mathbf p=a^2\dot{\mathbf x}$ is the canonical momentum per unit mass,
stored as `vel`. In the configured units the coefficient used by the code is
$a^2\sqrt{E^2(a)}D_m'(a)$.

The displacement fields live on the Lagrangian particle grid. PM++ maps every
stored particle slot back to its particle-grid coordinate and gathers the LPT
field there. This remains correct when the force mesh and particle grid have
different resolutions.

Finally, the LPT output passes through the same ownership map as an N-body
drift. The custom VJP attached to this step applies the transpose of that route
when gradients flow back to the LPT fields.

## Supported scope

The active implementation supports LPT orders 0, 1, and 2. Order 0 returns the
unperturbed particle lattice. First- and second-order initial conditions use
the equations above. Higher-order source terms are not part of the current
implementation.

## Implementation anchors

- `cosmo.py`: $E^2(a)$, $d\ln H/d\ln a$, and differentiable parameters
- `growth.py`: growth-table interpolation and derivative conversion
- `boltzmann.py`: transfer fit, growth integration, power, and variance tables
- `modes.py`: ordinary and nested noise and linear-mode scaling
- `lpt.py`: strain tensors, the second-order source, LPT state construction,
  and the ownership-route VJP
