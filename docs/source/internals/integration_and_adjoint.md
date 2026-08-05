# Integration and discrete adjoint

PM++ advances particles with a symplectic drift-force-kick composition. The
time factors are matched to linear growth rather than assuming that momentum
and force remain constant in ordinary cosmic time. The custom N-body VJP then
reverses the exact discrete maps while reconstructing particle states.

## Canonical equations

With comoving position $\mathbf x$ and canonical momentum per unit mass

$$
\mathbf p=a^2\dot{\mathbf x},
$$

the equations of motion are

$$
\dot{\mathbf x}=\frac{\mathbf p}{a^2},
\qquad
\dot{\mathbf p}=-\frac{\nabla\varphi}{a},
$$

where $\varphi=a\phi$ and the stored acceleration is

$$
\mathbf a=-\nabla\varphi.
$$

In the Zel'dovich approximation,

$$
\mathbf x=\mathbf q+D_1(a)\mathbf s^{(1)}.
$$

This gives the reference time dependence of canonical velocity,

$$
G_D(a)=a^2H(a)D_1'(a).
$$

After removing the configured $H_0$ unit, the implementation evaluates

$$
G_D(a)=a^2E(a)D_1'(a).
$$

The corresponding force time dependence is

$$
G_K(a)=a\frac{dG_D}{dt}\frac{1}{H_0^2}
=a^3E^2(a)
\left[D_1''(a)+
\left(2+\frac{d\ln H}{d\ln a}\right)D_1'(a)\right].
$$

Primes denote $d/d\ln a$.

## Growth-matched drift and kick factors

For a drift from $a_0$ to $a_1$ whose momentum is represented at $a_v$,

$$
\mathcal D(a_v\mid a_0,a_1)
=\frac{D_1(a_1)-D_1(a_0)}{G_D(a_v)}.
$$

For a kick from $a_0$ to $a_1$ whose acceleration is represented at $a_f$,

$$
\mathcal K(a_f\mid a_0,a_1)
=\frac{G_D(a_1)-G_D(a_0)}{G_K(a_f)}.
$$

The discrete maps are

$$
\mathbf d_1=\mathbf d_0+
\mathcal D\mathbf p,
$$

$$
\mathbf p_1=\mathbf p_0+
\mathcal K\mathbf a.
$$

These factors exactly reproduce the linear Zel'dovich growth history when the
state follows its assumed time dependence. They also work for a reversed
interval because exchanging $a_0$ and $a_1$ changes the numerator's sign.

## Configurable symplectic composition

`symp_splits` is a sequence of pairs $(d_j,k_j)$. PM++ requires

$$
\sum_jd_j=1,
\qquad
\sum_jk_j=1.
$$

Cumulative drift and kick fractions place substep endpoints by linear
interpolation in scale factor:

$$
a_{D,j}=a_\mathrm{prev}
+\left(\sum_{r\leq j}d_r\right)
(a_\mathrm{next}-a_\mathrm{prev}),
$$

$$
a_{K,j}=a_\mathrm{prev}
+\left(\sum_{r\leq j}k_r\right)
(a_\mathrm{next}-a_\mathrm{prev}).
$$

For every nonzero drift, the implementation:

1. drifts from the previous displacement time to $a_{D,j}$ using the current
   velocity time
2. routes particles to the owner of the new position
3. evaluates and stores the force at $a_{D,j}$.

For every nonzero kick, it advances velocity to $a_{K,j}$ using the most
recent stored force.

The default split is

```python
((0, 0.5), (1, 0.5))
```

which produces the kick-drift-force-kick sequence

$$
\mathbf p_{i-1/2}
=\mathbf p_{i-1}
+\mathcal K(a_{i-1}\mid a_{i-1},a_{i-1/2})\mathbf a_{i-1},
$$

$$
\mathbf d_i
=\mathbf d_{i-1}
+\mathcal D(a_{i-1/2}\mid a_{i-1},a_i)\mathbf p_{i-1/2},
$$

$$
\mathbf a_i=F(\mathbf d_i,\boldsymbol\theta),
$$

$$
\mathbf p_i
=\mathbf p_{i-1/2}
+\mathcal K(a_i\mid a_{i-1/2},a_i)\mathbf a_i.
$$

Before the first macro-step, `nbody_init` evaluates
$\mathbf a_0=F(\mathbf d_0,\boldsymbol\theta)$.

## Forward scan

The N-body scale-factor array is either supplied explicitly or constructed
from `a_start`, `a_stop`, and `a_nbody_maxstep`. Consecutive pairs define the
macro-steps. `lax.scan` applies `integrate` to every pair, which keeps the loop
inside the compiled JAX program.

The final state contains particle anchors, displacement, velocity,
acceleration, padding masks, and optional attributes. In a multi-device run,
the state after every drift is already in canonical authoritative order.

Forward-only collector and observer entry points expose intermediate results
without changing the core N-body state transition. They are separate from the
compact custom-adjoint contract.

## Discrete adjoint notation

For a scalar objective $J$, write a bar for a cotangent. For example,

$$
\bar{\mathbf d}=\frac{\partial J}{\partial\mathbf d}.
$$

The adjoint of a composed discrete program applies the transpose Jacobians in
the reverse order. PM++ differentiates the implemented time step, so its
reverse pass follows the same substep schedule in reverse.

## Kick adjoint

For

$$
\mathbf p_1=\mathbf p_0+\mathcal K\mathbf a,
$$

the state is reconstructed by

$$
\mathbf p_0=\mathbf p_1-\mathcal K\mathbf a.
$$

The cotangent update is

$$
\bar{\mathbf p}_0\mathrel{+}=\bar{\mathbf p}_1,
\qquad
\bar{\mathbf a}\mathrel{+}=\mathcal K\bar{\mathbf p}_1.
$$

If the cosmological parameters are differentiated, the factor contributes

$$
\bar{\boldsymbol\theta}\mathrel{+}=
\left\langle\bar{\mathbf p}_1,\mathbf a\right\rangle
\frac{\partial\mathcal K}{\partial\boldsymbol\theta}.
$$

PM++ obtains the parameter derivative of the scalar factor with JAX and
projects it onto the independent cosmological fields.

## Force adjoint

The force stage overwrites acceleration:

$$
(\mathbf d,\mathbf p,\mathbf a_\mathrm{in})
\mapsto
(\mathbf d,\mathbf p,F(\mathbf d,\boldsymbol\theta)).
$$

Its transpose applies the gravity VJP to $\bar{\mathbf a}_\mathrm{out}$:

$$
\bar{\mathbf d}_\mathrm{in}
=\bar{\mathbf d}_\mathrm{out}
+\left(\frac{\partial F}{\partial\mathbf d}\right)^\mathsf T
\bar{\mathbf a}_\mathrm{out},
$$

$$
\bar{\boldsymbol\theta}\mathrel{+}=
\left(\frac{\partial F}{\partial\boldsymbol\theta}\right)^\mathsf T
\bar{\mathbf a}_\mathrm{out}.
$$

Velocity cotangents pass through unchanged. Since the input acceleration was
overwritten, its cotangent is zero at this boundary.

An earlier kick may still require the acceleration that existed before this
force stage. After reversing the force and drift, PM++ recomputes that earlier
force from the reconstructed particle position. This restores the primal state
needed by the next reverse substep without storing it from the forward run.

## Drift and route adjoint

Ignoring ownership for a moment, a drift is

$$
\mathbf d_1=\mathbf d_0+\mathcal D\mathbf p.
$$

It is inverted by

$$
\mathbf d_0=\mathbf d_1-\mathcal D\mathbf p.
$$

Its local transpose is

$$
\bar{\mathbf d}_0\mathrel{+}=\bar{\mathbf d}_1,
\qquad
\bar{\mathbf p}\mathrel{+}=\mathcal D\bar{\mathbf d}_1,
$$

with the parameter contribution

$$
\bar{\boldsymbol\theta}\mathrel{+}=
\left\langle\bar{\mathbf d}_1,\mathbf p\right\rangle
\frac{\partial\mathcal D}{\partial\boldsymbol\theta}.
$$

In distributed execution, the physical drift is followed by the canonical
ownership route $R$. The full map is

$$
\mathbf z_1=R(\mathbf z_0+\mathcal D\mathbf v).
$$

The reverse pass reconstructs the pre-drift canonical layout, rebuilds the
route plan, applies $R^\mathsf T$ to displacement, velocity, and acceleration
cotangents, and only then applies the local drift transpose. This is necessary
because a slot index on the post-drift owner is generally not the same slot or
device as the corresponding pre-drift particle.

## Reverse macro-step and full custom VJP

`integrate_adj` records the same static substep schedule as `integrate` and
traverses it backward. A kick is reversed directly. A drift stage reverses, in
order:

1. the force that followed the drift
2. the particle route and drift
3. the restoration of the force that entered that drift stage.

`nbody_adj` scans all macro-steps in reverse scale-factor order, then applies
the adjoint of the initial force evaluation at $a_0$.

The `nbody` custom VJP saves the final state rather than the full forward
trajectory. Reverse reconstruction and force reevaluation make the amount of
stored particle state independent of the number of time steps. More steps
increase reverse computation, but do not require a tape containing every
particle state.

## Implementation anchors

- `steps.py`: growth-matched factors, drift, force, kick, reconstruction, and
  substep adjoints
- `nbody.py`: forward scan, observer and collector entry points, reverse scan,
  and the `nbody` custom VJP
- `halo_moving.py`: reconstruction and transpose of the ownership route
- `gravity.py`, `scatter.py`, `gather.py`, `FFT_distributed.py`: force VJP
  primitives used inside `force_adj`
