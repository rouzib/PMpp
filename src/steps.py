import jax
from jax import vjp, value_and_grad
import jax.numpy as jnp

from .configuration import Configuration
from .cosmo import E2, H_deriv
from .gravity import gravity, duplicate_slot_counts
from .growth import growth
from .particles import Particles


def _halo_move_float_outputs(ptcl, disp, vel, acc, share_only_right, conf):
    _, disp, vel, acc, _, _, _, _ = conf.mGPU_halo_moving(
        ptcl.pmid,
        disp,
        vel,
        acc,
        conf.halo_start,
        conf.halo_end,
        ptcl.halo_mask,
        ptcl.unused_index,
        share_only_right,
    )
    return disp, vel, acc


def _halo_move_vjp(ptcl, disp, vel, acc, disp_cot, vel_cot, acc_cot, share_only_right, conf):
    _, halo_move_vjp = vjp(
        lambda disp_in, vel_in, acc_in: _halo_move_float_outputs(
            ptcl, disp_in, vel_in, acc_in, share_only_right, conf
        ),
        disp,
        vel,
        acc,
    )
    return halo_move_vjp((disp_cot, vel_cot, acc_cot))


def _G_D(a, cosmo, conf):
    """Growth factor of ZA canonical velocity in [H_0]."""
    return a ** 2 * jnp.sqrt(E2(a, cosmo)) * growth(a, cosmo, conf, deriv=1)


def _G_K(a, cosmo, conf):
    """Growth factor of ZA accelerations in [H_0^2]."""
    return a ** 3 * E2(a, cosmo) * (
            growth(a, cosmo, conf, deriv=2)
            + (2 + H_deriv(a, cosmo)) * growth(a, cosmo, conf, deriv=1)
    )


def drift_factor(a_vel, a_prev, a_next, cosmo, conf):
    """Drift time step factor of conf.float_dtype in [1/H_0]."""
    factor = growth(a_next, cosmo, conf) - growth(a_prev, cosmo, conf)
    factor /= _G_D(a_vel, cosmo, conf)
    return factor


def kick_factor(a_acc, a_prev, a_next, cosmo, conf):
    """Kick time step factor of conf.float_dtype in [1/H_0]."""
    factor = _G_D(a_next, cosmo, conf) - _G_D(a_prev, cosmo, conf)
    factor /= _G_K(a_acc, cosmo, conf)
    return factor


def drift(a_vel, a_prev, a_next, ptcl: Particles, cosmo, conf: Configuration):
    """Drift."""
    factor = drift_factor(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    share_only_right = conf.num_devices == 2

    disp = ptcl.disp + ptcl.vel * factor

    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl.pmid, disp, ptcl.vel, ptcl.acc, conf.halo_start, conf.halo_end,
        ptcl.halo_mask, ptcl.unused_index, share_only_right)
    return ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=acc, halo_mask=halo_mask, unused_index=unused_indexes)


def drift_adj(a_vel, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf):
    """Drift, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(drift_factor, argnums=3)
    factor, cosmo_cot_drift = factor_valgrad(a_vel, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)
    share_only_right = conf.num_devices == 2

    # drift
    ptcl_before_halo = ptcl
    disp_before_halo = ptcl.disp + ptcl.vel * factor

    pmid, disp, vel, acc, halo_mask, unused_indexes, has_failed, max_ptcl_moved = conf.mGPU_halo_moving(
        ptcl_before_halo.pmid,
        disp_before_halo,
        ptcl_before_halo.vel,
        ptcl_before_halo.acc,
        conf.halo_start,
        conf.halo_end,
        ptcl_before_halo.halo_mask,
        ptcl_before_halo.unused_index,
        share_only_right,
    )
    ptcl = ptcl.replace(pmid=pmid, disp=disp, vel=vel, acc=acc, halo_mask=halo_mask, unused_index=unused_indexes)

    disp, vel, acc = _halo_move_vjp(
        ptcl_before_halo,
        disp_before_halo,
        ptcl_before_halo.vel,
        ptcl_before_halo.acc,
        ptcl_cot.disp,
        ptcl_cot.vel,
        ptcl_cot.acc,
        share_only_right,
        conf,
    )
    ptcl_cot = ptcl_cot.replace(disp=disp, vel=vel, acc=acc)

    # particle adjoint
    vel_cot = ptcl_cot.vel - ptcl_cot.disp * factor
    ptcl_cot = ptcl_cot.replace(vel=vel_cot)

    # cosmology adjoint
    cosmo_cot_drift *= (ptcl_cot.disp * ptcl_before_halo.vel).sum()
    cosmo_cot -= cosmo_cot_drift

    return ptcl, ptcl_cot, cosmo_cot


def kick(a_acc, a_prev, a_next, ptcl, cosmo, conf):
    """Kick."""
    factor = kick_factor(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    vel = ptcl.vel + ptcl.acc * factor

    return ptcl.replace(vel=vel)


def kick_adj(a_acc, a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    """Kick, and particle and cosmology adjoints."""
    factor_valgrad = value_and_grad(kick_factor, argnums=3)
    factor, cosmo_cot_kick = factor_valgrad(a_acc, a_prev, a_next, cosmo, conf)
    factor = factor.astype(conf.float_dtype)

    # kick
    vel = ptcl.vel + ptcl.acc * factor
    ptcl = ptcl.replace(vel=vel)

    # particle adjoint
    disp_cot = ptcl_cot.disp - ptcl_cot.acc * factor
    ptcl_cot = ptcl_cot.replace(disp=disp_cot)

    # cosmology adjoint
    cosmo_cot_kick *= (ptcl_cot.vel * ptcl.acc).sum()
    cosmo_cot_force *= factor
    cosmo_cot -= cosmo_cot_kick + cosmo_cot_force

    return ptcl, ptcl_cot, cosmo_cot


def force(a, ptcl, cosmo, conf):
    """Force."""
    acc = gravity(a, ptcl, cosmo, conf)
    return ptcl.replace(acc=acc)


def force_adj(a, ptcl, ptcl_cot, cosmo, conf):
    """Force, and particle and cosmology vjp."""
    # force
    acc, gravity_vjp = vjp(gravity, a, ptcl, cosmo, conf)

    ptcl = ptcl.replace(acc=acc)

    # particle and cosmology vjp
    _, ptcl_cot_force, cosmo_cot_force, _ = gravity_vjp(ptcl_cot.vel)
    counts = duplicate_slot_counts(ptcl, conf).astype(ptcl_cot_force.disp.dtype)
    acc_cot = jnp.where(counts != 0, ptcl_cot_force.disp / counts, 0)
    ptcl_cot = ptcl_cot.replace(acc=acc_cot)

    return ptcl, ptcl_cot, cosmo_cot_force


def integrate(a_prev, a_next, ptcl, cosmo, conf):
    """Symplectic integration for one step."""
    D = K = 0
    a_disp = a_vel = a_acc = a_prev
    for d, k in conf.symp_splits:
        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ptcl = drift(a_vel, a_disp, a_disp_next, ptcl, cosmo, conf)
            a_disp = a_disp_next
            ptcl = force(a_disp, ptcl, cosmo, conf)
            a_acc = a_disp

        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            ptcl = kick(a_acc, a_vel, a_vel_next, ptcl, cosmo, conf)
            a_vel = a_vel_next

    return ptcl


def integrate_adj(a_prev, a_next, ptcl, ptcl_cot, cosmo, cosmo_cot, cosmo_cot_force, conf):
    """Symplectic integration adjoint for one step."""
    K = D = 0
    a_disp = a_vel = a_acc = a_prev
    for d, k in reversed(conf.symp_splits):
        if k != 0:
            K += k
            a_vel_next = a_prev * (1 - K) + a_next * K
            ptcl, ptcl_cot, cosmo_cot = kick_adj(a_acc, a_vel, a_vel_next, ptcl, ptcl_cot, cosmo, cosmo_cot,
                                                 cosmo_cot_force, conf)
            a_vel = a_vel_next

        if d != 0:
            D += d
            a_disp_next = a_prev * (1 - D) + a_next * D
            ptcl, ptcl_cot, cosmo_cot = drift_adj(a_vel, a_disp, a_disp_next, ptcl, ptcl_cot, cosmo, cosmo_cot, conf)
            a_disp = a_disp_next
            ptcl, ptcl_cot, cosmo_cot_force = force_adj(a_disp, ptcl, ptcl_cot, cosmo, conf)
            a_acc = a_disp

    return ptcl, ptcl_cot, cosmo_cot, cosmo_cot_force
