import jax
import jax.numpy as jnp


def _linear_interp_primal(x, xp, fp):
    x = jnp.asarray(x)
    idx = jnp.clip(jnp.searchsorted(xp, x, side='right') - 1, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + (x - x0) * slope
    y = jnp.where(x <= xp[0], fp[0], y)
    y = jnp.where(x >= xp[-1], fp[-1], y)
    return y


def _linear_interp_slope(x, xp, fp):
    x = jnp.asarray(x)
    left_pos = jnp.searchsorted(xp, x, side='left')
    right_pos = jnp.searchsorted(xp, x, side='right')
    left_idx = jnp.clip(left_pos - 1, 0, xp.shape[0] - 2)
    right_idx = jnp.clip(right_pos - 1, 0, xp.shape[0] - 2)

    left_slope = (fp[left_idx + 1] - fp[left_idx]) / (xp[left_idx + 1] - xp[left_idx])
    right_slope = (fp[right_idx + 1] - fp[right_idx]) / (xp[right_idx + 1] - xp[right_idx])

    exact_knot = right_pos > left_pos
    interior_knot = exact_knot & (left_pos > 0) & (left_pos < xp.shape[0] - 1)
    left_edge_knot = exact_knot & (left_pos == 0)
    right_edge_knot = exact_knot & (left_pos == xp.shape[0] - 1)

    slope = right_slope
    slope = jnp.where(interior_knot, 0.5 * (left_slope + right_slope), slope)
    slope = jnp.where(left_edge_knot, 0.5 * right_slope, slope)
    slope = jnp.where(right_edge_knot, 0.5 * left_slope, slope)
    slope = jnp.where((x < xp[0]) | (x > xp[-1]), 0, slope)
    return slope


@jax.custom_jvp
def _linear_interp(x, xp, fp):
    return _linear_interp_primal(x, xp, fp)


@_linear_interp.defjvp
def _linear_interp_jvp(primals, tangents):
    x, xp, fp = primals
    x_dot, xp_dot, fp_dot = tangents
    y = _linear_interp_primal(x, xp, fp)
    y_dot = jnp.zeros_like(y)

    if not isinstance(x_dot, jax.custom_derivatives.SymbolicZero):
        y_dot = y_dot + _linear_interp_slope(x, xp, fp) * x_dot
    if not isinstance(xp_dot, jax.custom_derivatives.SymbolicZero):
        y_dot = y_dot + jax.jvp(lambda xp_arg: _linear_interp_primal(x, xp_arg, fp), (xp,), (xp_dot,))[1]
    if not isinstance(fp_dot, jax.custom_derivatives.SymbolicZero):
        y_dot = y_dot + jax.jvp(lambda fp_arg: _linear_interp_primal(x, xp, fp_arg), (fp,), (fp_dot,))[1]

    return y, y_dot


def growth(a, cosmo, conf, order=1, deriv=0):
    """Evaluate interpolation of (LPT) growth function or derivative, the n-th
    derivatives of the m-th order growth function :math:`\mathrm{d}^n D_m /
    \mathrm{d}\ln^n a`, at given scale factors. Growth functions are normalized at the
    matter dominated era instead of today.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology
    conf : Configuration
    order : int in {1, 2}, optional
        Order of growth function.
    deriv : int in {0, 1, 2}, optional
        Order of growth function derivatives.

    Returns
    -------
    D : jax.Array of (a * 1.).dtype
        Growth functions or derivatives.

    Raises
    ------
    ValueError
        If ``cosmo.growth`` table is empty.

    """
    if cosmo.growth is None:
        raise ValueError('Growth table is empty. Call growth_integ or boltzmann first.')

    a = jnp.asarray(a)
    float_dtype = jnp.promote_types(a.dtype, float)

    D = a ** order * _linear_interp(a, conf.growth_a, cosmo.growth[order - 1][deriv])

    return D.astype(float_dtype)
