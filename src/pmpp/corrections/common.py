"""Shared helpers for PM++ potential corrections."""

import jax
import jax.numpy as jnp
import numpy as np

try:
    import haiku as hk
except ImportError:
    hk = None

try:
    import optax
except ImportError:
    optax = None


def require_haiku(feature_name):
    """Raise a targeted import error when an optional Haiku feature is used.

    Parameters
    ----------
    feature_name
        Name of the optional feature that requires the dependency."""
    if hk is None:
        raise ImportError(
            f"haiku is required for {feature_name}. Install dm-haiku to enable this correction."
        )


def require_optax(feature_name):
    """Raise a targeted import error when an optional Optax utility is used.

    Parameters
    ----------
    feature_name
        Name of the optional feature that requires the dependency."""
    if optax is None:
        raise ImportError(
            f"optax is required for {feature_name}. Install optax to enable this optimizer utility."
        )


HaikuModuleBase = hk.Module if hk is not None else object


def normalized_k_magnitude_transposed(conf):
    """Return ``|k| / k_Nyquist`` on the transposed spectral layout.

    Parameters
    ----------
    conf
        Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers."""
    kx, ky, kz = [jnp.squeeze(a).astype(conf.float_dtype) for a in conf.kvec]
    k_nyquist = jnp.asarray(jnp.pi / conf.cell_size, dtype=conf.float_dtype)
    return jnp.sqrt(
        (kx[:, None, None] / k_nyquist) ** 2
        + (ky[None, :, None] / k_nyquist) ** 2
        + (kz[None, None, :] / k_nyquist) ** 2
    )


def default_cosmo_features(dtype):
    """Default conditioning vector used when no cosmology is available.

    Parameters
    ----------
    dtype
        Floating-point dtype for created arrays or model parameters."""
    return jnp.asarray([0.3, 0.8], dtype=dtype)


def resolve_sigma8(cosmo, dtype, allow_missing_sigma8=False):
    """Extract a finite sigma8 value for correction conditioning.

    Parameters
    ----------
    cosmo
        Cosmology object supplying density, growth, and transfer parameters.
    dtype
        Floating-point dtype for created arrays or model parameters.
    allow_missing_sigma8
        Whether to substitute a default sigma8 feature when it is unavailable on the cosmology."""
    default_sigma8 = float(default_cosmo_features(dtype)[1])
    if cosmo is None:
        return jnp.asarray(default_sigma8, dtype=dtype)
    try:
        sigma8 = float(jax.device_get(jnp.asarray(cosmo.sigma8, dtype=dtype)))
    except Exception:
        if not allow_missing_sigma8:
            raise ValueError(
                "sigma8 is not initialized on the cosmology. "
                "Run boltzmann()/from_sigma8() first, or set allow_missing_sigma8=True "
                "for lightweight test-only correction initialization."
            ) from None
        sigma8 = default_sigma8
    if not np.isfinite(sigma8):
        if not allow_missing_sigma8:
            raise ValueError(
                "sigma8 is non-finite on the cosmology. "
                "The upstream transfer/varlin initialization failed."
            )
        sigma8 = default_sigma8
    return jnp.asarray(sigma8, dtype=dtype)


def cosmo_features(cosmo, dtype, allow_missing_sigma8=False):
    """Pack cosmology conditioning features as ``[Omega_m, sigma8]``.

    Parameters
    ----------
    dtype
        Floating-point dtype for created arrays or model parameters.
    allow_missing_sigma8
        Whether to substitute a default sigma8 feature when it is unavailable on the cosmology."""
    if cosmo is None:
        return default_cosmo_features(dtype)
    if hasattr(cosmo, "shape"):
        return jnp.asarray(cosmo, dtype=dtype)
    sigma8 = resolve_sigma8(cosmo, dtype, allow_missing_sigma8=allow_missing_sigma8)
    return jnp.asarray([cosmo.Omega_m, sigma8], dtype=dtype)


def correction_cosmo_features(correction, cosmo, dtype):
    """Resolve conditioning features, honoring values stored on the correction.

    Parameters
    ----------
    cosmo
        Cosmology object supplying density, growth, and transfer parameters.
    dtype
        Floating-point dtype for created arrays or model parameters."""
    if correction is not None and getattr(correction, "sigma8_value", None) is not None:
        if cosmo is None:
            return jnp.asarray([default_cosmo_features(dtype)[0], correction.sigma8_value], dtype=dtype)
        if hasattr(cosmo, "shape"):
            return jnp.asarray(cosmo, dtype=dtype)
        return jnp.asarray([cosmo.Omega_m, correction.sigma8_value], dtype=dtype)
    allow_missing_sigma8 = False if correction is None else getattr(correction, "allow_missing_sigma8", False)
    return cosmo_features(cosmo, dtype, allow_missing_sigma8=allow_missing_sigma8)


def build_correction_optimizer(
    learning_rate,
    gradient_clip_norm=100.0,
    optimizer_name="adamax",
    apply_if_finite_steps=100,
):
    """Build the default Optax optimizer chain for correction training.

    Parameters
    ----------
    learning_rate : float
        Base optimizer learning rate.
    gradient_clip_norm : float or None, optional
        Global-norm clip applied before the optimizer step. Disable by passing
        ``None`` or a non-positive value.
    optimizer_name : {"adamax", "adam"}, optional
        Underlying Optax optimizer.
    apply_if_finite_steps : int, optional
        Wrap the optimizer with ``optax.apply_if_finite`` for this many
        tolerated consecutive non-finite updates. Set to ``0`` to disable it.

    Returns
    -------
    optax.GradientTransformation
        Optimizer chain ready for training correction parameters.
    """
    require_optax("potential correction optimizer construction")
    transforms = []
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(gradient_clip_norm))

    if optimizer_name == "adamax":
        transforms.append(optax.adamax(learning_rate))
    elif optimizer_name == "adam":
        transforms.append(optax.adam(learning_rate))
    else:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}.")

    optimizer = optax.chain(*transforms) if len(transforms) > 1 else transforms[0]
    if apply_if_finite_steps and apply_if_finite_steps > 0:
        optimizer = optax.apply_if_finite(optimizer, apply_if_finite_steps)
    return optimizer
