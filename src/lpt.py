from functools import partial

import jax
import jax.numpy as jnp
from jax._src.numpy.fft import _fft_norm
from jax.sharding import NamedSharding, PartitionSpec as P

from .cosmo import E2
from .fft import fftinv, fftfwd, fftfreq
from .gravity import laplace, neg_grad
from .growth import growth
from .particles import Particles
from .utils import AXIS_NAME


def _strain(kvec, i, j, pot, conf):
    """LPT strain component sourced by scalar potential only.

     The Nyquist planes are not zeroed when i == j.

    .. _Notes on FFT-based differentiation:
        https://math.mit.edu/~stevenj/fft-deriv.pdf

    """
    k_i, k_j = kvec[i], kvec[j]

    nyquist = jnp.pi / conf.ptcl_spacing
    eps = nyquist * jnp.finfo(conf.float_dtype).eps

    if i != j:
        k_i = jnp.where(jnp.abs(jnp.abs(k_i) - nyquist) <= eps, 0, k_i)
        k_j = jnp.where(jnp.abs(jnp.abs(k_j) - nyquist) <= eps, 0, k_j)

    strain = -k_i * k_j * pot

    strain = conf.mGPU_irfftn(strain)
    strain *= _fft_norm(s=jnp.array(conf.ptcl_grid_shape, dtype=strain.dtype), func_name="rfftn", norm="ortho")
    strain = strain.astype(conf.float_dtype)  # no jnp.complex32

    return strain


def _L(kvec, pot_m, pot_n, conf):
    m_eq_n = pot_n is None
    if m_eq_n:
        pot_n = pot_m

    L = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype,
                  device=NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None)))

    for i in range(conf.dim):
        strain_m = _strain(kvec, i, i, pot_m, conf)

        for j in range(conf.dim - 1, i, -1):
            strain_n = _strain(kvec, j, j, pot_n, conf)

            L += strain_m * strain_n

        if not m_eq_n:
            for j in range(i - 1, -1, -1):
                strain_n = _strain(kvec, j, j, pot_n, conf)

                L += strain_m * strain_n

    if not m_eq_n:
        L *= 0.5

    # Assuming strain sourced by scalar potential only, symmetric about ``i`` and ``j``,
    # for lpt_order <=3, i.e., m, n <= 2
    for i in range(conf.dim - 1):
        for j in range(i + 1, conf.dim):
            strain_m = _strain(kvec, i, j, pot_m, conf)

            strain_n = strain_m
            if not m_eq_n:
                strain_n = _strain(kvec, j, i, pot_n, conf)

            L -= strain_m * strain_n

    return L


@partial(jax.jit, static_argnames=('conf',))
@partial(jax.checkpoint, static_argnums=(2,))
def lpt(modes, cosmo, conf):
    """Lagrangian perturbation theory at ``conf.lpt_order``.

    Parameters
    ----------
    modes : jax.Array
        Linear matter overdensity Fourier modes in [L^3].
    cosmo : Cosmology
    conf : Configuration

    Returns
    -------
    ptcl : Particles
    obsvbl : Observables

    Raises
    ------
    ValueError
        If ``conf.dim`` or ``conf.lpt_order`` is not supported.

    """
    if conf.dim not in (1, 2, 3):
        raise ValueError(f'dim={conf.dim} not supported')
    if conf.lpt_order not in (0, 1, 2, 3):
        raise ValueError(f'lpt_order={conf.lpt_order} not supported')

    modes /= conf.ptcl_cell_vol  # remove volume factor first for convenience

    kvec = conf.kvec_spacing

    pot = []

    if conf.lpt_order > 0:
        src_1 = modes

        pot_1 = laplace(kvec, src_1, conf, cosmo)
        pot.append(pot_1)

    if conf.lpt_order > 1:
        src_2 = _L(kvec, pot_1, None, conf)

        src_2 = conf.mGPU_rfftn(src_2)

        pot_2 = laplace(kvec, src_2, conf, cosmo)
        pot.append(pot_2)

    if conf.lpt_order > 2:
        raise NotImplementedError('TODO')

    a = conf.a_start
    ptcl = Particles.gen_grid(conf, vel=True)
    disp = ptcl.disp
    vel = ptcl.vel
    ptcl_grid_shape = jnp.array(conf.ptcl_grid_shape, dtype=jnp.int32)
    ptcl_grid_coord = jnp.rint(
        (ptcl.pmid.astype(conf.float_dtype) * conf.cell_size + ptcl.disp) / conf.ptcl_spacing
    ).astype(jnp.int32)
    ptcl_grid_coord %= ptcl_grid_shape
    ptcl_idx = (
        (ptcl_grid_coord[:, 0] * ptcl_grid_shape[1] + ptcl_grid_coord[:, 1]) * ptcl_grid_shape[2]
        + ptcl_grid_coord[:, 2]
    )
    ptcl_idx = jnp.where(ptcl.unused_index, jnp.int32(-1), ptcl_idx)
    valid_slots = ~ptcl.unused_index

    for order in range(1, 1 + conf.lpt_order):
        D = growth(a, cosmo, conf, order=order)
        dD_dlna = growth(a, cosmo, conf, order=order, deriv=1)
        a2HDp = a ** 2 * jnp.sqrt(E2(a, cosmo)) * dD_dlna
        D = D.astype(conf.float_dtype)
        a2HDp = a2HDp.astype(conf.float_dtype)

        for i, k in enumerate(kvec):
            grad = neg_grad(k, pot[order - 1], conf.ptcl_spacing)

            grad = conf.mGPU_irfftn(grad)
            grad = grad.astype(conf.float_dtype)  # no jnp.complex32

            grad = jnp.take(grad.reshape(-1), ptcl_idx, mode="wrap")
            grad = jnp.where(valid_slots, grad, 0)

            disp = disp.at[:, i].add(D * grad)
            vel = vel.at[:, i].add(a2HDp * grad)

    scratch_acc = disp[:, :0]
    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl.pmid, disp, vel, scratch_acc, conf.halo_start, conf.halo_end,
        ptcl.halo_mask, ptcl.unused_index, True)

    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=None, halo_mask=halo_mask,
                        unused_index=unused_indexes)
