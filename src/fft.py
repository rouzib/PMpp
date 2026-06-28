import jax.numpy as jnp


def fftfreq(shape, spacing, dtype=jnp.float64, sparse=True):
    """(Angular) wavevectors for FFT.

    Parameters
    ----------
    shape : tuple of int
        Shape of the real field.
    spacing : float or None, optional
        Grid spacing. None is equivalent to spacing of 2π with angular wavevector period
        of 1, or equivalently spacing of 1 with (non-angular) wavevector period of 1.
    dtype : DTypeLike
    sparse : bool, optional
        Whether to return sparse broadcastable or dense wavevector grids.

    Returns
    -------
    kvec : list of jax.Array
        Wavevectors.

    Notes
    -----

    The angular wavevectors differ from the numpy ``fftfreq`` and ``rfftfreq`` by a
    multiplicative factor of 2π.

    """
    period = 1
    if spacing is not None:
        period = 2 * jnp.pi / spacing

    kvec = []
    for axis, s in enumerate(shape[:-1]):
        k = jnp.fft.fftfreq(s) * period
        kvec.append(k.astype(dtype))

    k = jnp.fft.rfftfreq(shape[-1]) * period
    kvec.append(k.astype(dtype))

    kvec = jnp.meshgrid(*kvec, sparse=sparse, indexing='ij')

    return kvec


def fftfwd(f, shape=None, axes=None, norm=None):
    r"""Forward FFT from a real field to Hermitian Fourier coefficients.

    Parameters
    ----------
    f : ArrayLike
        Input field. See ``a`` in ``numpy.fft.rfftn``.
    shape : sequence of int, optional
        See ``s`` in ``numpy.fft.rfftn``.
    axes : sequence of int, optional
        See ``numpy.fft.rfftn``.
    norm : float or {'backward', 'ortho', 'forward'}, optional
        If a float, interpret it as a grid spacing and apply the corresponding
        physical-volume normalization. Otherwise pass the value through to
        ``jax.numpy.fft.rfftn`` unchanged.

    Returns
    -------
    f : jax.Array
        Output Hermitian Fourier coefficients.

    Raises
    ------
    ValueError
        If input field is not real.

    Notes
    -----
    Given the grid spacing, the normalization convention is

    .. math::

        f(\bm{k}) = \int \mathrm{d}\bm{x} f(\bm(x}) e^{-i \bm{k} \cdot \bm{x}}
                    \approx \frac{V}{N} \sum_\bm{x} f(\bm{x}) e^{-i \bm{k} \cdot \bm{x}}

    where :math:`V/N` is the cell volume, :math:`V` is the box volume, and
    :math:`N` is the number of grid points summed over.

    """
    f = jnp.asarray(f)

    if not jnp.isrealobj(f):
        raise ValueError('input field must be real')

    if norm in {None, 'backward', 'ortho', 'forward'}:
        return jnp.fft.rfftn(f, s=shape, axes=axes, norm=norm)

    d = f.ndim
    if shape is not None:  # len(shape) == len(axes) if both are not None
        d = len(shape)
    if axes is not None:
        d = len(axes)

    return norm ** d * jnp.fft.rfftn(f, s=shape, axes=axes, norm='backward')


def fftinv(f, shape=None, axes=None, norm=None):
    r"""Inverse FFT from Hermitian Fourier coefficients to a real field.

    Parameters
    ----------
    f : ArrayLike
        Input field. See ``a`` in ``numpy.fft.irfftn``.
    shape : sequence of int, optional
        See ``s`` in ``numpy.fft.irfftn``.
    axes : sequence of int, optional
        See ``numpy.fft.irfftn``.
    norm : float or {'backward', 'ortho', 'forward'}, optional
        If a float, interpret it as a grid spacing and apply the inverse
        physical-volume normalization. Otherwise pass the value through to
        ``jax.numpy.fft.irfftn`` unchanged.

    Returns
    -------
    f : jax.Array
        Output real-space field.

    Raises
    ------
    ValueError
        If input field is not complex.

    Notes
    -----
    Given the grid spacing, the normalization convention is

    .. math::

        f(\bm{x}) = \int \frac{\mathrm{d}\bm{k}}{(2\pi)^d} f(\bm(k}) e^{i \bm{k} \cdot \bm{x}}
                    \approx \frac{1}{V} \sum_\bm{k} f(\bm{k}) e^{i \bm{k} \cdot \bm{x}}

    where :math:`d` is the FFT dimension, :math:`V` is the box volume.

    """
    f = jnp.asarray(f)

    if not jnp.iscomplexobj(f):
        raise ValueError('input field must be Hermitian complex')

    if norm in {None, 'backward', 'ortho', 'forward'}:
        return jnp.fft.irfftn(f, s=shape, axes=axes, norm=norm)

    d = f.ndim
    if shape is not None:  # len(shape) == len(axes) if both are not None
        d = len(shape)
    if axes is not None:
        d = len(axes)

    return norm ** -d * jnp.fft.irfftn(f, s=shape, axes=axes, norm='backward')
