from dataclasses import field
from functools import partial
from operator import add, sub
from typing import ClassVar, Optional

import jax
import jax.numpy as jnp
from jax import value_and_grad, Array
from jax.tree_util import tree_map
from jax.typing import ArrayLike

from .configuration import Configuration
from .utils import pytree_dataclass


def E2(a, cosmo):
    r"""Squared Hubble parameter time scaling factors, :math:`E^2`, at given scale
    factors.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    E2 : jax.Array of cosmo.conf.cosmo_dtype
        Squared Hubble parameter time scaling factors.

    Notes
    -----
    The squared Hubble parameter

    .. math::

        H^2(a) = H_0^2 E^2(a),

    has time scaling

    .. math::

        E^2(a) = \Omega_\mathrm{m} a^{-3} + \Omega_\mathrm{k} a^{-2}
                 + \Omega_\mathrm{de} a^{-3 (1 + w_0 + w_a)} e^{-3 w_a (1 - a)}.

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    de_a = a ** (-3 * (1 + cosmo.w_0 + cosmo.w_a)) * jnp.exp(-3 * cosmo.w_a * (1 - a))
    return cosmo.Omega_m * a ** -3 + cosmo.Omega_k * a ** -2 + cosmo.Omega_de * de_a


@partial(jnp.vectorize, excluded=(1,))
def H_deriv(a, cosmo):
    r"""Hubble parameter derivatives, :math:`\mathrm{d}\ln H / \mathrm{d}\ln a`, at
    given scale factors.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    dlnH_dlna : jax.Array of cosmo.conf.cosmo_dtype
        Hubble parameter derivatives.

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    E2_value, E2_grad = value_and_grad(E2)(a, cosmo)
    return 0.5 * a * E2_grad / E2_value


def Omega_m_a(a, cosmo):
    r"""Matter density parameters, :math:`\Omega_\mathrm{m}(a)`, at given scale factors.

    Parameters
    ----------
    a : ArrayLike
        Scale factors.
    cosmo : Cosmology

    Returns
    -------
    Omega : jax.Array of cosmo.conf.cosmo_dtype
        Matter density parameters.

    Notes
    -----

    .. math::

        \Omega_\mathrm{m}(a) = \frac{\Omega_\mathrm{m} a^{-3}}{E^2(a)}

    """
    a = jnp.asarray(a, dtype=cosmo.conf.cosmo_dtype)

    return cosmo.Omega_m / (a ** 3 * E2(a, cosmo))


@partial(pytree_dataclass, aux_fields="conf", frozen=True)
class Cosmology:
    """Cosmological and configuration parameters, "immutable" as a frozen dataclass.

    Cosmological parameters with trailing underscores ("foo_") can be set to None, in
    which case they take some fixed values (set by class variable "foo_fixed") and will
    not receive gradients. They should be accessed through corresponding properties
    named without the trailing underscores ("foo").

    Linear operators (addition, subtraction, and scalar multiplication) are defined for
    Cosmology tangent and cotangent vectors.

    Float parameters are converted to JAX arrays of conf.cosmo_dtype at instantiation,
    to avoid possible JAX weak type problems.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    A_s_1e9 : float ArrayLike
        Primordial scalar power spectrum amplitude, multiplied by 1e9.
    n_s : float ArrayLike
        Primordial scalar power spectrum spectral index.
    Omega_m : float ArrayLike
        Total matter density parameter today.
    Omega_b : float ArrayLike
        Baryonic matter density parameter today.
    Omega_k_ : None or float ArrayLike, optional
        Spatial curvature density parameter today. Default is None.
    w_0_ : None or float ArrayLike, optional
        Dark energy equation of state constant parameter. Default is None.
    w_a_ : None or float ArrayLike, optional
        Dark energy equation of state linear parameter. Default is None.
    h : float ArrayLike
        Hubble constant in unit of 100 [km/s/Mpc].

    """

    conf: Configuration = field(repr=False)

    A_s_1e9: ArrayLike
    n_s: ArrayLike
    Omega_m: ArrayLike
    Omega_b: ArrayLike
    h: ArrayLike

    Omega_k_: Optional[ArrayLike] = None
    Omega_k_fixed: ClassVar[float] = 0
    w_0_: Optional[ArrayLike] = None
    w_0_fixed: ClassVar[float] = -1
    w_a_: Optional[ArrayLike] = None
    w_a_fixed: ClassVar[float] = 0

    transfer: Optional[Array] = field(default=None, compare=False)

    growth: Optional[Array] = field(default=None, compare=False)

    varlin: Optional[Array] = field(default=None, compare=False)

    def __post_init__(self):
        if self._is_transforming():
            return

        dtype = self.conf.cosmo_dtype
        for name, value in self.named_children():
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, name, value)

    def __add__(self, other):
        return tree_map(add, self, other)

    def __sub__(self, other):
        return tree_map(sub, self, other)

    def __mul__(self, other):
        return tree_map(lambda x: x * other, self)

    def __rmul__(self, other):
        return self.__mul__(other)

    @classmethod
    def from_sigma8(cls, conf, sigma8, *args, **kwargs):
        """Construct cosmology with sigma8 instead of A_s."""
        from .boltzmann import boltzmann

        cosmo = cls(conf, 1, *args, **kwargs)
        cosmo = boltzmann(cosmo, conf)

        A_s_1e9 = (sigma8 / cosmo.sigma8) ** 2

        return cls(conf, A_s_1e9, *args, **kwargs)

    def astype(self, dtype):
        """Cast parameters to dtype by changing conf.cosmo_dtype."""
        conf = self.conf.replace(cosmo_dtype=dtype)
        return self.replace(conf=conf)  # calls __post_init__

    @property
    def k_pivot(self):
        """Primordial scalar power spectrum pivot scale in [1/L].

        Pivot scale is defined h-less unit, so needs h to convert its unit to [1/L].

        """
        return self.conf.k_pivot_Mpc / (self.h * self.conf.Mpc_SI) * self.conf.L

    @property
    def A_s(self):
        """Primordial scalar power spectrum amplitude."""
        return self.A_s_1e9 * 1e-9

    @property
    def Omega_c(self):
        """Cold dark matter density parameter today."""
        return self.Omega_m - self.Omega_b

    @property
    def Omega_k(self):
        """Spatial curvature density parameter today."""
        return self.Omega_k_fixed if self.Omega_k_ is None else self.Omega_k_

    @property
    def Omega_de(self):
        """Dark energy density parameter today."""
        return 1 - (self.Omega_m + self.Omega_k)

    @property
    def w_0(self):
        """Dark energy equation of state constant parameter."""
        return self.w_0_fixed if self.w_0_ is None else self.w_0_

    @property
    def w_a(self):
        """Dark energy equation of state linear parameter."""
        return self.w_a_fixed if self.w_a_ is None else self.w_a_

    @property
    def sigma8(self):
        """Linear matter rms overdensity within a tophat sphere of 8 Mpc/h radius at a=1."""
        from .boltzmann import varlin
        R = 8 * self.conf.Mpc_SI / self.conf.L
        return jnp.sqrt(varlin(R, 1, self, self.conf))

    @property
    def ptcl_mass(self):
        """Particle mass in [M]."""
        return self.conf.rho_crit * self.Omega_m * self.conf.ptcl_cell_vol


SimpleLCDM = partial(
    Cosmology,
    A_s_1e9=2.0,
    n_s=0.96,
    Omega_m=0.3,
    Omega_b=0.05,
    h=0.7,
)


_COSMO_PARAM_FIELDS = (
    "A_s_1e9",
    "n_s",
    "Omega_m",
    "Omega_b",
    "h",
    "Omega_k_",
    "w_0_",
    "w_a_",
)


def cosmology_param_names(cosmo):
    names = ["A_s_1e9", "n_s", "Omega_m", "Omega_b", "h"]
    if cosmo.Omega_k_ is not None:
        names.append("Omega_k_")
    if cosmo.w_0_ is not None:
        names.append("w_0_")
    if cosmo.w_a_ is not None:
        names.append("w_a_")
    return tuple(names)


def cosmology_param_values(cosmo, names=None):
    if names is None:
        names = cosmology_param_names(cosmo)
    return tuple(getattr(cosmo, name) for name in names)


def replace_cosmology_params(cosmo, names, values):
    return cosmo.replace(**dict(zip(names, values)))


def zero_cosmology_param_cotangent(cosmo):
    kwargs = {
        "A_s_1e9": jnp.zeros_like(cosmo.A_s_1e9),
        "n_s": jnp.zeros_like(cosmo.n_s),
        "Omega_m": jnp.zeros_like(cosmo.Omega_m),
        "Omega_b": jnp.zeros_like(cosmo.Omega_b),
        "h": jnp.zeros_like(cosmo.h),
        "Omega_k_": None if cosmo.Omega_k_ is None else jnp.zeros_like(cosmo.Omega_k_),
        "w_0_": None if cosmo.w_0_ is None else jnp.zeros_like(cosmo.w_0_),
        "w_a_": None if cosmo.w_a_ is None else jnp.zeros_like(cosmo.w_a_),
        "transfer": None,
        "growth": None,
        "varlin": None,
    }
    return cosmo.replace(**kwargs)


def cosmology_param_cotangent(cosmo, names, values):
    return zero_cosmology_param_cotangent(cosmo).replace(**dict(zip(names, values)))


def project_cosmology_param_cotangent(cot):
    kwargs = {name: getattr(cot, name) for name in _COSMO_PARAM_FIELDS}
    kwargs.update(transfer=None, growth=None, varlin=None)
    return cot.replace(**kwargs)


def _combine_optional_cotangents(lhs, rhs, op):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return op(lhs, rhs)


def add_cosmology_cotangents(lhs, rhs):
    lhs = project_cosmology_param_cotangent(lhs)
    rhs = project_cosmology_param_cotangent(rhs)
    return lhs.replace(
        A_s_1e9=lhs.A_s_1e9 + rhs.A_s_1e9,
        n_s=lhs.n_s + rhs.n_s,
        Omega_m=lhs.Omega_m + rhs.Omega_m,
        Omega_b=lhs.Omega_b + rhs.Omega_b,
        h=lhs.h + rhs.h,
        Omega_k_=_combine_optional_cotangents(lhs.Omega_k_, rhs.Omega_k_, add),
        w_0_=_combine_optional_cotangents(lhs.w_0_, rhs.w_0_, add),
        w_a_=_combine_optional_cotangents(lhs.w_a_, rhs.w_a_, add),
        transfer=None,
        growth=None,
        varlin=None,
    )


def sub_cosmology_cotangents(lhs, rhs):
    lhs = project_cosmology_param_cotangent(lhs)
    rhs = project_cosmology_param_cotangent(rhs)
    return lhs.replace(
        A_s_1e9=lhs.A_s_1e9 - rhs.A_s_1e9,
        n_s=lhs.n_s - rhs.n_s,
        Omega_m=lhs.Omega_m - rhs.Omega_m,
        Omega_b=lhs.Omega_b - rhs.Omega_b,
        h=lhs.h - rhs.h,
        Omega_k_=_combine_optional_cotangents(lhs.Omega_k_, rhs.Omega_k_, sub),
        w_0_=_combine_optional_cotangents(lhs.w_0_, rhs.w_0_, sub),
        w_a_=_combine_optional_cotangents(lhs.w_a_, rhs.w_a_, sub),
        transfer=None,
        growth=None,
        varlin=None,
    )


def scale_cosmology_cotangent(cot, scalar):
    cot = project_cosmology_param_cotangent(cot)
    return cot.replace(
        A_s_1e9=cot.A_s_1e9 * scalar,
        n_s=cot.n_s * scalar,
        Omega_m=cot.Omega_m * scalar,
        Omega_b=cot.Omega_b * scalar,
        h=cot.h * scalar,
        Omega_k_=None if cot.Omega_k_ is None else cot.Omega_k_ * scalar,
        w_0_=None if cot.w_0_ is None else cot.w_0_ * scalar,
        w_a_=None if cot.w_a_ is None else cot.w_a_ * scalar,
        transfer=None,
        growth=None,
        varlin=None,
    )
