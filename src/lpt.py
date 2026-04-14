from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from .cosmo import E2
from .fft import fftinv, fftfwd, fftfreq
from .gravity import laplace_transposed, neg_grad
from .growth import growth
from .particles import Particles
from .steps import _halo_move_vjp
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

    if conf.compute_mesh is None:
        strain = jnp.fft.irfftn(strain)
    else:
        strain = conf.mGPU_irfftn_transposed(strain)
    strain = strain.astype(conf.float_dtype)  # no jnp.complex32

    return strain


def _L(kvec, pot_m, pot_n, conf):
    """Quadratic LPT source built from products of strain tensors.

    For second-order LPT this evaluates the invariant
    ``sum_i<j phi_ii psi_jj - phi_ij psi_ji``. When ``pot_n`` is ``None`` the
    same potential is used for both factors, which is the common 2LPT case.
    """
    m_eq_n = pot_n is None
    if m_eq_n:
        pot_n = pot_m

    if conf.compute_mesh is None:
        L = jnp.zeros(conf.ptcl_grid_shape, dtype=conf.float_dtype)
    else:
        L = jnp.zeros(
            conf.ptcl_grid_shape,
            dtype=conf.float_dtype,
            device=NamedSharding(conf.compute_mesh, P(AXIS_NAME, None, None)),
        )

    if conf.lpt_cache_strains:
        # Cache diagonal strains to avoid redundant irfftn calls (conf.lpt_cache_strains=True).
        # Saves dim-1 irfftn calls per _L at the cost of keeping dim extra strain arrays
        # (each of shape ptcl_grid_shape) alive simultaneously. Set lpt_cache_strains=False
        # to recompute instead, trading compute for GPU memory.
        diag_m = [_strain(kvec, i, i, pot_m, conf) for i in range(conf.dim)]
        diag_n = diag_m if m_eq_n else [_strain(kvec, i, i, pot_n, conf) for i in range(conf.dim)]

        for i in range(conf.dim):
            for j in range(conf.dim - 1, i, -1):
                L += diag_m[i] * diag_n[j]

            if not m_eq_n:
                for j in range(i - 1, -1, -1):
                    L += diag_m[i] * diag_n[j]
    else:
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


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _attach_lpt_halo_move_vjp(disp_before, vel_before, disp_after, vel_after, ptcl_before, conf):
    """Attach the halo-move pullback to LPT outputs.

    LPT builds displacement/velocity on the canonical particle grid and then
    calls the same halo-movement machinery as the N-body drift. This wrapper
    makes the backward pass use that halo-move VJP while returning only the
    post-move floating arrays in the primal result.
    """
    return disp_after, vel_after


def _attach_lpt_halo_move_vjp_fwd(disp_before, vel_before, disp_after, vel_after, ptcl_before, conf):
    """Forward rule for the LPT halo-move custom VJP."""
    return (disp_after, vel_after), (disp_before, vel_before, ptcl_before)


def _attach_lpt_halo_move_vjp_bwd(conf, res, cotangents):
    """Backward rule that routes LPT cotangents through halo movement."""
    disp_before, vel_before, ptcl_before = res
    disp_cot, vel_cot = cotangents
    scratch_acc = disp_before[:, :0]

    disp_before_cot, vel_before_cot, _ = _halo_move_vjp(
        ptcl_before,
        disp_before,
        vel_before,
        scratch_acc,
        disp_cot,
        vel_cot,
        scratch_acc,
        conf,
    )

    return (
        disp_before_cot,
        vel_before_cot,
        jnp.zeros_like(disp_cot),
        jnp.zeros_like(vel_cot),
        None,
    )


_attach_lpt_halo_move_vjp.defvjp(
    _attach_lpt_halo_move_vjp_fwd,
    _attach_lpt_halo_move_vjp_bwd,
)


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
    if conf.compute_mesh is not None:
        modes = jax.lax.with_sharding_constraint(
            modes,
            NamedSharding(conf.compute_mesh, P(None, AXIS_NAME, None)),
        )

    pot = []

    if conf.lpt_order > 0:
        src_1 = modes

        pot_1 = laplace_transposed(kvec, src_1, conf, cosmo)
        pot.append(pot_1)

    if conf.lpt_order > 1:
        src_2 = _L(kvec, pot_1, None, conf)

        if conf.compute_mesh is None:
            src_2 = jnp.fft.rfftn(src_2)
        else:
            src_2 = conf.mGPU_rfftn_transposed(src_2)

        pot_2 = laplace_transposed(kvec, src_2, conf, cosmo)
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

            if conf.compute_mesh is None:
                grad = jnp.fft.irfftn(grad)
            else:
                grad = conf.mGPU_irfftn_transposed(grad)
            grad = grad.astype(conf.float_dtype)  # no jnp.complex32

            grad = jnp.take(grad.reshape(-1), ptcl_idx, mode="wrap")
            grad = jnp.where(valid_slots, grad, 0)

            disp = disp.at[:, i].add(D * grad)
            vel = vel.at[:, i].add(a2HDp * grad)

    disp_before_halo = disp
    vel_before_halo = vel
    scratch_acc = disp[:, :0]
    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl.pmid,
        ptcl.disp,
        disp,
        vel,
        scratch_acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.unused_index,
    )
    ptcl_after = ptcl.replace(
        pmid=pmid,
        disp=disp,
        vel=vel,
        acc=None,
        halo_mask=halo_mask,
        unused_index=unused_indexes,
    )
    disp, vel = _attach_lpt_halo_move_vjp(
        disp_before_halo,
        vel_before_halo,
        ptcl_after.disp,
        ptcl_after.vel,
        ptcl,
        conf,
    )

    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=None, halo_mask=halo_mask,
                        unused_index=unused_indexes)
