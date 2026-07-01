# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JAX-based Dormand-Prince ODE integration with adaptive stepsize.

Integrate systems of ordinary differential equations (ODEs) using the JAX
autograd/diff library and the Dormand-Prince method for adaptive integration
stepsize calculation. Provides improved integration accuracy over fixed
stepsize integration methods.

For details of the mixed 4th/5th order Runge-Kutta integration method, see
https://doi.org/10.1090/S0025-5718-1986-0815836-3

Adjoint algorithm based on Appendix C of https://arxiv.org/pdf/1806.07366.pdf
"""


from functools import partial
import operator as op

import jax
import jax.numpy as jnp
from jax._src import core
from jax import custom_derivatives
from jax import lax
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.util import safe_map, safe_zip
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_leaves, tree_map
from jax._src import linear_util as lu
from jax._src.linear_util import _missing_debug_info

map = safe_map
zip = safe_zip


def ravel_first_arg(f, unravel):
    """Wrap an ODE function so its first pytree argument is a flat vector.

    Parameters
    ----------
    f : callable
        Function accepting a pytree state as its first argument.
    unravel : callable
        Callable converting the flattened state vector back to the original
        pytree structure.

    Returns
    -------
    callable
        Wrapper around ``f`` that accepts a flattened first argument and
        returns a flattened output pytree.
    """
    return ravel_first_arg_(lu.wrap_init(f, debug_info=_missing_debug_info("lu")), unravel).call_wrapped


@lu.transformation
def ravel_first_arg_(unravel, y_flat, *args):
    """Linear-util transformation backing ``ravel_first_arg``.

    Parameters
    ----------
    unravel : callable
        Callable converting a flattened state vector to the original pytree.
    y_flat : jax.Array
        Flattened state vector.
    *args
        Additional positional arguments passed through to the wrapped function.

    Yields
    ------
    tuple
        Input tuple for the wrapped function followed by the flattened result.
    """
    y = unravel(y_flat)
    ans = yield (y,) + args, {}
    ans_flat, _ = ravel_pytree(ans)
    yield ans_flat


def interp_fit_dopri(y0, y1, k, dt):
    """Fit Dormand-Prince dense-output interpolation coefficients.

    Parameters
    ----------
    y0, y1 : jax.Array
        State at the beginning and end of the accepted Runge-Kutta step.
    k : jax.Array
        Stage derivatives produced by the Dormand-Prince tableau.
    dt : jax.Array or float
        Step size.

    Returns
    -------
    jax.Array
        Quartic dense-output coefficients ordered for Horner evaluation.
    """
    # Fit a polynomial to the results of a Runge-Kutta step.
    dps_c_mid = jnp.array([
        6025192743 / 30085553152 / 2, 0, 51252292925 / 65400821598 / 2,
        -2691868925 / 45128329728 / 2, 187940372067 / 1594534317056 / 2,
        -1776094331 / 19743644256 / 2, 11237099 / 235043384 / 2], dtype=y0.dtype)
    y_mid = y0 + dt.astype(y0.dtype) * jnp.dot(dps_c_mid, k)
    return jnp.asarray(fit_4th_order_polynomial(y0, y1, y_mid, k[0], k[-1], dt))


def fit_4th_order_polynomial(y0, y1, y_mid, dy0, dy1, dt):
    """Fit a quartic polynomial from endpoint and midpoint RK data.

    Parameters
    ----------
    y0, y1, y_mid : jax.Array
        State at the step start, end, and midpoint.
    dy0, dy1 : jax.Array
        Derivatives at the step start and end.
    dt : jax.Array or float
        Step size.

    Returns
    -------
    tuple[jax.Array, ...]
        Quartic coefficients ``(a, b, c, d, e)`` such that the dense solution
        can be evaluated on the normalized interval.
    """
    dt = dt.astype(y0.dtype)
    a = -2.*dt*dy0 + 2.*dt*dy1 - 8.*y0 - 8.*y1 + 16.*y_mid
    b = 5.*dt*dy0 - 3.*dt*dy1 + 18.*y0 + 14.*y1 - 32.*y_mid
    c = -4.*dt*dy0 + dt*dy1 - 11.*y0 - 5.*y1 + 16.*y_mid
    d = dt * dy0
    e = y0
    return a, b, c, d, e


def initial_step_size(fun, t0, y0, order, rtol, atol, f0):
    """Choose an initial adaptive RK step size from Hairer et al.'s heuristic.

    Parameters
    ----------
    fun : callable
        RHS function evaluated as ``fun(y, t)``.
    t0 : jax.Array or float
        Initial integration time.
    y0 : jax.Array
        Initial state.
    order : int
        Convergence order of the embedded error estimator.
    rtol, atol : float
        Relative and absolute solver tolerances.
    f0 : jax.Array
        Derivative at ``(y0, t0)``.

    Returns
    -------
    jax.Array
        Suggested first step size.
    """
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    y0, f0 = promote_dtypes_inexact(y0, f0)
    dtype = y0.dtype

    scale = atol + jnp.abs(y0) * rtol
    d0 = jnp.linalg.norm(y0 / scale.astype(dtype))
    d1 = jnp.linalg.norm(f0 / scale.astype(dtype))

    h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
    y1 = y0 + h0.astype(dtype) * f0
    f1 = fun(y1, t0 + h0)
    d2 = jnp.linalg.norm((f1 - f0) / scale.astype(dtype)) / h0

    h1 = jnp.where((d1 <= 1e-15) & (d2 <= 1e-15),
                   jnp.maximum(1e-6, h0 * 1e-3),
                   (0.01 / jnp.maximum(d1, d2)) ** (1. / (order + 1.)))

    return jnp.minimum(100. * h0, h1)


def runge_kutta_step(func, y0, f0, t0, dt):
    """Take one Dormand-Prince 5(4) Runge-Kutta step.

    Parameters
    ----------
    func : callable
        RHS function evaluated as ``func(y, t)``.
    y0 : jax.Array
        State at the start of the step.
    f0 : jax.Array
        Derivative at ``(y0, t0)``.
    t0 : jax.Array or float
        Step start time.
    dt : jax.Array or float
        Proposed step size.

    Returns
    -------
    tuple
        ``(y1, f1, error_estimate, k)`` containing the step endpoint,
        derivative at the endpoint, embedded local error estimate, and all
        tableau stage derivatives.
    """
    # Dopri5 Butcher tableaux
    alpha = jnp.array([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0], dtype=dt.dtype)
    beta = jnp.array(
        [[1 / 5, 0, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
         [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
         [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
         [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
         [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]],
        dtype=f0.dtype)
    c_sol = jnp.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        dtype=f0.dtype)
    c_error = jnp.array([
        35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085, 125 / 192 -
        451 / 720, -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1. / 60.
    ], dtype=f0.dtype)

    def body_fun(i, k):
        """Advance one body iteration in the enclosing loop.

        Parameters
        ----------
        i
            Loop index.
        k
            Wavenumber samples at which to evaluate the CDM transfer fit.
        """
        ti = t0 + dt * alpha[i-1]
        yi = y0 + dt.astype(f0.dtype) * jnp.dot(beta[i-1, :], k)
        ft = func(yi, ti)
        return k.at[i, :].set(ft)

    k = jnp.zeros((7, f0.shape[0]), f0.dtype).at[0, :].set(f0)
    k = lax.fori_loop(1, 7, body_fun, k)

    y1 = dt.astype(f0.dtype) * jnp.dot(c_sol, k) + y0
    y1_error = dt.astype(f0.dtype) * jnp.dot(c_error, k)
    f1 = k[-1]
    return y1, f1, y1_error, k


def abs2(x):
    """Squared magnitude that handles real and complex states."""
    if jnp.iscomplexobj(x):
        return x.real ** 2 + x.imag ** 2
    else:
        return x ** 2


def mean_error_ratio(error_estimate, rtol, atol, y0, y1):
    """Return the RMS local-error ratio against adaptive tolerances.

    Parameters
    ----------
    error_estimate
        Estimated local truncation error from the Runge-Kutta step.
    rtol
        Relative error tolerance.
    atol
        Absolute error tolerance.
    y0
        State at the beginning of the step.
    y1
        Candidate state at the end of the step."""
    err_tol = atol + rtol * jnp.maximum(jnp.abs(y0), jnp.abs(y1))
    err_ratio = error_estimate / err_tol.astype(error_estimate.dtype)
    return jnp.sqrt(jnp.mean(abs2(err_ratio)))


def optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0,
                      dfactor=0.2, order=5.0):
    """Compute the next adaptive Runge-Kutta step size.

    Parameters
    ----------
    last_step : float
        Previous accepted step size.
    mean_error_ratio : float
        Scaled local error ratio; values below one are acceptable.
    safety : float, optional
        Safety factor applied to the asymptotic update.
    ifactor : float, optional
        Maximum allowed step-growth factor.
    dfactor : float, optional
        Maximum allowed step-shrink factor.
    order : float, optional
        Effective order of the embedded error estimate.

    Returns
    -------
    float
        Suggested next step size.
    """
    dfactor = jnp.where(mean_error_ratio < 1, 1.0, dfactor)

    factor = jnp.minimum(ifactor,
                         jnp.maximum(mean_error_ratio**(-1.0 / order) * safety, dfactor))
    return jnp.where(mean_error_ratio == 0, last_step * ifactor, last_step * factor)


def odeint(func, y0, t, *args, rtol=1.4e-8, atol=1.4e-8, mxstep=jnp.inf, hmax=jnp.inf,
           dt0=None):
    """Adaptive stepsize (Dormand-Prince) Runge-Kutta odeint implementation.

    Parameters
    ----------
    func : callable
        Right-hand side with signature ``func(y, t, *args)``.
    y0 : pytree
        Initial state.
    t : array-like
        Strictly increasing times at which the solution is requested.
    *args
        Extra arguments forwarded to ``func``.
    rtol, atol : float, optional
        Relative and absolute local error tolerances.
    mxstep : int or float, optional
        Maximum number of internal adaptive steps per target time.
    hmax : float, optional
        Maximum allowed internal step size.
    dt0 : float, None, or 2-tuple of float or None, optional
        Optional initial step size for forward and reverse integration.

    Returns
    -------
    pytree
        Solution values at each time in ``t`` with the same structure as
        ``y0`` and a new leading time axis.
    """
    for arg in tree_leaves(args):
        if not isinstance(arg, core.Tracer) and not core.valid_jaxtype(arg):
            raise TypeError(
                f"The contents of odeint *args must be arrays or scalars, but got {arg}.")
    if not jnp.issubdtype(t.dtype, jnp.floating):
        raise TypeError(f"t must be an array of floats, but got {t}.")

    converted, consts = custom_derivatives.closure_convert(
        func, y0, t[0], *args)
    return _odeint_wrapper(converted, rtol, atol, mxstep, hmax, dt0, y0, t, *args, *consts)


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _odeint_wrapper(func, rtol, atol, mxstep, hmax, dt0, y0, ts, *args):
    """Flatten pytree state, call the vector ODE integrator, and unravel output."""
    y0, unravel = ravel_pytree(y0)
    func = ravel_first_arg(func, unravel)
    out = _odeint(func, rtol, atol, mxstep, hmax, dt0, y0, ts, *args)
    return jax.vmap(unravel)(out)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def _odeint(func, rtol, atol, mxstep, hmax, dt0, y0, ts, *args):
    """Flat-state adaptive ODE integrator with a custom adjoint."""
    def func_(y, t):
        """Evaluate the wrapped ODE function.

        Parameters
        ----------
        y
            Flattened ODE state.
        t
            Scalar integration time.
        """
        return func(y, t, *args)

    def scan_fun(carry, target_t):

        """Advance one scan iteration in the enclosing integration routine.

        Parameters
        ----------
        carry
            Loop-carried state for a JAX scan or while-loop body.
        target_t
            Requested output time for the adaptive ODE scan.
        """
        def cond_fun(state):
            """Test whether the adaptive ODE loop should continue stepping.

            Parameters
            ----------
            state
                Loop or custom-adjoint state tuple.
            """
            i, _, _, t, dt, _, _ = state
            return (t < target_t) & (i < mxstep) & (dt > 0)

        def body_fun(state):
            """Advance one body iteration in the enclosing loop.

            Parameters
            ----------
            state
                Loop or custom-adjoint state tuple.
            """
            i, y, f, t, dt, last_t, interp_coeff = state
            next_y, next_f, next_y_error, k = runge_kutta_step(
                func_, y, f, t, dt)
            next_t = t + dt
            error_ratio = mean_error_ratio(next_y_error, rtol, atol, y, next_y)
            new_interp_coeff = interp_fit_dopri(y, next_y, k, dt)
            dt = jnp.clip(optimal_step_size(
                dt, error_ratio), a_min=0., a_max=hmax)

            new = [i + 1, next_y, next_f, next_t, dt,      t, new_interp_coeff]
            old = [i + 1,      y,      f,      t, dt, last_t,     interp_coeff]
            return map(partial(jnp.where, error_ratio <= 1.), new, old)

        _, *carry = lax.while_loop(cond_fun, body_fun, [0] + carry)
        _, _, t, _, last_t, interp_coeff = carry
        relative_output_time = (target_t - last_t) / (t - last_t)
        y_target = jnp.polyval(
            interp_coeff, relative_output_time.astype(interp_coeff.dtype))
        return carry, y_target

    f0 = func_(y0, ts[0])
    dt = dt0[0] if isinstance(dt0, tuple) else dt0
    if dt is None:
        dt = initial_step_size(func_, ts[0], y0, 4, rtol, atol, f0)
    dt = jnp.clip(dt, a_min=0., a_max=hmax)
    interp_coeff = jnp.array([y0] * 5)
    init_carry = [y0, f0, ts[0], dt, ts[0], interp_coeff]
    _, ys = lax.scan(scan_fun, init_carry, ts[1:])
    return jnp.concatenate((y0[None], ys))


def _odeint_fwd(func, rtol, atol, mxstep, hmax, dt0, y0, ts, *args):
    """Forward rule for the adaptive ODE custom VJP."""
    ys = _odeint(func, rtol, atol, mxstep, hmax, dt0, y0, ts, *args)
    return ys, (ys, ts, args)


def _odeint_rev(func, rtol, atol, mxstep, hmax, dt0, res, g):
    """Reverse-time adjoint rule for the adaptive ODE solver."""
    dt = dt0[1] if isinstance(dt0, tuple) else dt0
    ys, ts, args = res

    def aug_dynamics(augmented_state, t, *args):
        """Original system augmented with vjp_y, vjp_t and vjp_args.

        Parameters
        ----------
        augmented_state
            Adjoint state containing time, primal state, adjoint, and parameter cotangents."""
        y, y_bar, *_ = augmented_state
        # `t` here is negatice time, so we need to negate again to get back to
        # normal time. See the `odeint` invocation in `scan_fun` below.
        y_dot, vjpfun = jax.vjp(func, y, -t, *args)
        return (-y_dot, *vjpfun(y_bar))

    y_bar = g[-1]
    ts_bar = []
    t0_bar = 0.

    def scan_fun(carry, i):
        """Advance one scan iteration in the enclosing integration routine.

        Parameters
        ----------
        carry
            Loop-carried state for a JAX scan or while-loop body.
        i
            Loop index.
        """
        y_bar, t0_bar, args_bar = carry
        # Compute effect of moving measurement time
        # `t_bar` should not be complex as it represents time
        t_bar = jnp.dot(func(ys[i], ts[i], *args), g[i]).real
        t0_bar = t0_bar - t_bar
        # Run augmented system backwards to previous observation
        _, y_bar, t0_bar, args_bar = odeint(
            aug_dynamics, (ys[i], y_bar, t0_bar, args_bar),
            jnp.array([-ts[i], -ts[i - 1]]),
            *args, rtol=rtol, atol=atol, mxstep=mxstep, hmax=hmax, dt0=dt)
        y_bar, t0_bar, args_bar = tree_map(
            op.itemgetter(1), (y_bar, t0_bar, args_bar))
        # Add gradient from current output
        y_bar = y_bar + g[i - 1]
        return (y_bar, t0_bar, args_bar), t_bar

    init_carry = (g[-1], 0., tree_map(jnp.zeros_like, args))
    (y_bar, t0_bar, args_bar), rev_ts_bar = lax.scan(
        scan_fun, init_carry, jnp.arange(len(ts) - 1, 0, -1))
    ts_bar = jnp.concatenate([jnp.array([t0_bar]), rev_ts_bar[::-1]])
    return (y_bar, ts_bar, *args_bar)


_odeint.defvjp(_odeint_fwd, _odeint_rev)
