import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pmwd.boltzmann import boltzmann as boltzmann_pmwd
from pmwd.configuration import Configuration as ConfigurationPMWD
from pmwd.cosmology import SimpleLCDM as SimpleLCDM_PM
from pmwd.lpt import lpt as lpt_pmwd
from pmwd.modes import linear_modes as linear_modes_pmwd
from pmwd.modes import white_noise as white_noise_pmwd
from pmwd.nbody import (
    drift_adj as drift_adj_pmwd,
    force_adj as force_adj_pmwd,
    kick_adj as kick_adj_pmwd,
    nbody as nbody_pmwd,
    nbody_adj_init as nbody_adj_init_pmwd,
    nbody_adj_step as nbody_adj_step_pmwd,
)
from pmwd.scatter import scatter as scatter_pmwd

from src.boltzmann import boltzmann as boltzmann_pmpp
from src.configuration import Configuration
from src.cosmo import SimpleLCDM as SimpleLCDM_PP
from src.lpt import lpt as lpt_pmpp
from src.modes import linear_modes as linear_modes_pmpp
from src.nbody import (
    nbody as nbody_pmpp,
    nbody_adj_init as nbody_adj_init_pmpp,
    nbody_adj_step as nbody_adj_step_pmpp,
)
from src.scatter import scatter as scatter_pmpp
from src.steps import drift_adj as drift_adj_pmpp
from src.steps import force_adj as force_adj_pmpp
from src.steps import kick_adj as kick_adj_pmpp
from src.utils import create_compute_mesh, pmid_to_idx


def _zeros_like_cosmo(cosmo):
    return cosmo.replace(
        A_s_1e9=jnp.zeros_like(cosmo.A_s_1e9),
        n_s=jnp.zeros_like(cosmo.n_s),
        Omega_m=jnp.zeros_like(cosmo.Omega_m),
        Omega_b=jnp.zeros_like(cosmo.Omega_b),
        h=jnp.zeros_like(cosmo.h),
    )


def _create_confs(num_ptcl):
    box_size = 100.0
    ptcl_spacing = box_size / num_ptcl

    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"][:2]
    if len(gpu_devices) < 2:
        raise RuntimeError("This diagnostic requires 2 GPUs")
    compute_mesh = create_compute_mesh(gpu_devices)

    conf_pmpp = Configuration(
        ptcl_spacing,
        (num_ptcl,) * 3,
        mesh_shape=1,
        compute_mesh=compute_mesh,
        max_ptcl_per_slice=int(num_ptcl**3 / len(gpu_devices) * 2.5),
        max_share_ptcl=50000,
        max_share_gather_ptcl=150000,
        a_start=1 / 64,
        a_stop=2 / 64,
        a_nbody_maxstep=1 / 64,
        lpt_order=2,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    conf_pmwd = ConfigurationPMWD(
        ptcl_spacing=conf_pmpp.ptcl_spacing,
        ptcl_grid_shape=conf_pmpp.ptcl_grid_shape,
        mesh_shape=conf_pmpp.mesh_shape,
        a_start=conf_pmpp.a_start,
        a_stop=conf_pmpp.a_stop,
        a_nbody_maxstep=conf_pmpp.a_nbody_maxstep,
        lpt_order=conf_pmpp.lpt_order,
        cosmo_dtype=jnp.float64,
        float_dtype=jnp.float64,
    )
    return conf_pmpp, conf_pmwd


def _build_forward_states(conf_pmpp, conf_pmwd, seed):
    cosmo_pmpp = boltzmann_pmpp(SimpleLCDM_PP(conf_pmpp), conf_pmpp)
    cosmo_pmwd = boltzmann_pmwd(SimpleLCDM_PM(conf_pmwd), conf_pmwd)

    noise_real = white_noise_pmwd(seed, conf_pmwd, real=True)
    modes_pmwd = linear_modes_pmwd(noise_real, cosmo_pmwd, conf_pmwd)
    modes_pmpp = linear_modes_pmpp(noise_real, cosmo_pmpp, conf_pmpp)

    ptcl_lpt_pmwd, _ = lpt_pmwd(modes_pmwd, cosmo_pmwd, conf_pmwd)
    ptcl_lpt_pmpp = lpt_pmpp(
        modes_pmpp,
        cosmo_pmpp,
        conf_pmpp.replace(max_share_ptcl=conf_pmpp.max_share_ptcl * 4),
    )

    ptcl_final_pmwd, _ = nbody_pmwd(ptcl_lpt_pmwd, None, cosmo_pmwd, conf_pmwd)
    ptcl_final_pmpp = nbody_pmpp(ptcl_lpt_pmpp, cosmo_pmpp, conf_pmpp)
    return cosmo_pmpp, cosmo_pmwd, ptcl_final_pmpp, ptcl_final_pmwd


def _particle_keys_pmwd(ptcl_pmwd, conf_pmwd):
    return np.asarray(jax.device_get(pmid_to_idx(ptcl_pmwd.pmid, conf_pmwd)), dtype=np.int64)


def _slot_keys_pmpp(ptcl_pmpp, conf_pmpp):
    return np.asarray(
        jax.device_get(pmid_to_idx(ptcl_pmpp.pmid, conf_pmpp, ptcl_pmpp.unused_index)),
        dtype=np.int64,
    )


def _first_slot_mapping(ptcl_pmwd, ptcl_pmpp, conf_pmwd, conf_pmpp):
    particle_keys = _particle_keys_pmwd(ptcl_pmwd, conf_pmwd)
    slot_keys = _slot_keys_pmpp(ptcl_pmpp, conf_pmpp)
    key_to_particle = {int(key): pid for pid, key in enumerate(particle_keys)}

    first_slot = np.full(particle_keys.shape[0], -1, dtype=np.int32)
    for slot, key in enumerate(slot_keys):
        if key < 0:
            continue
        pid = key_to_particle.get(int(key))
        if pid is not None and first_slot[pid] < 0:
            first_slot[pid] = slot

    if np.any(first_slot < 0):
        missing = np.flatnonzero(first_slot < 0)
        raise RuntimeError(f"Missing PMPP slots for PMWD particle ids: {missing[:8].tolist()}")
    return first_slot


def _aggregate_slot_field(field_pmpp, ptcl_pmpp, conf_pmpp, ptcl_pmwd, conf_pmwd, mode):
    particle_keys = _particle_keys_pmwd(ptcl_pmwd, conf_pmwd)
    key_to_particle = {int(key): pid for pid, key in enumerate(particle_keys)}
    slot_keys = _slot_keys_pmpp(ptcl_pmpp, conf_pmpp)
    field_pmpp = np.asarray(jax.device_get(field_pmpp))

    out = np.zeros((particle_keys.shape[0],) + field_pmpp.shape[1:], dtype=field_pmpp.dtype)
    counts = np.zeros(particle_keys.shape[0], dtype=np.int32)
    for slot, key in enumerate(slot_keys):
        if key < 0:
            continue
        pid = key_to_particle.get(int(key))
        if pid is not None:
            out[pid] += field_pmpp[slot]
            counts[pid] += 1

    if mode == "sum":
        return out
    if mode == "mean":
        reshape = (counts.shape[0],) + (1,) * (field_pmpp.ndim - 1)
        counts = np.maximum(counts.reshape(reshape), 1)
        return out / counts
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def _field_metrics(ref, val):
    ref = np.asarray(ref)
    val = np.asarray(val)
    diff = val - ref
    denom = np.maximum(np.abs(ref), 1e-30)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "max_rel": float(np.max(np.abs(diff) / denom)),
    }


def _print_metrics(label, ref, val):
    metrics = _field_metrics(ref, val)
    print(
        f"{label:28s}  max_abs={metrics['max_abs']:.3e}  "
        f"mean_abs={metrics['mean_abs']:.3e}  max_rel={metrics['max_rel']:.3e}"
    )


def _print_cosmo_diff(label, ref, val):
    print(f"\n{label}")
    for name in ("A_s_1e9", "n_s", "Omega_m", "Omega_b", "h"):
        ref_val = np.asarray(jax.device_get(getattr(ref, name)))
        val_val = np.asarray(jax.device_get(getattr(val, name)))
        _print_metrics(name, ref_val, val_val)


def _compare_ptcl_fields(label, ptcl_ref, ptcl_val_agg):
    print(f"\n{label}")
    _print_metrics("disp", np.asarray(jax.device_get(ptcl_ref.disp)), ptcl_val_agg["disp"])
    _print_metrics("vel", np.asarray(jax.device_get(ptcl_ref.vel)), ptcl_val_agg["vel"])
    _print_metrics("acc", np.asarray(jax.device_get(ptcl_ref.acc)), ptcl_val_agg["acc"])


def _aggregate_ptcl_cot(ptcl_cot_pmpp, ptcl_pmpp, conf_pmpp, ptcl_pmwd, conf_pmwd, mode):
    return {
        "disp": _aggregate_slot_field(ptcl_cot_pmpp.disp, ptcl_pmpp, conf_pmpp, ptcl_pmwd, conf_pmwd, mode),
        "vel": _aggregate_slot_field(ptcl_cot_pmpp.vel, ptcl_pmpp, conf_pmpp, ptcl_pmwd, conf_pmwd, mode),
        "acc": _aggregate_slot_field(ptcl_cot_pmpp.acc, ptcl_pmpp, conf_pmpp, ptcl_pmwd, conf_pmwd, mode),
    }


def _linear_probe(shape, dtype):
    n = np.prod(shape)
    probe = np.linspace(-1.0, 1.0, int(n), dtype=np.float64).reshape(shape)
    return jnp.asarray(probe, dtype=dtype)


def _coerce_ptcl_cot(ptcl_cot, dtype):
    return ptcl_cot.replace(
        disp=jnp.asarray(ptcl_cot.disp, dtype=dtype),
        vel=jnp.asarray(ptcl_cot.vel, dtype=dtype),
        acc=jnp.asarray(ptcl_cot.acc, dtype=dtype),
    )


def main():
    num_ptcl = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    conf_pmpp, conf_pmwd = _create_confs(num_ptcl)
    cosmo_pmpp, cosmo_pmwd, ptcl_final_pmpp, ptcl_final_pmwd = _build_forward_states(
        conf_pmpp, conf_pmwd, seed
    )
    first_slot = _first_slot_mapping(ptcl_final_pmwd, ptcl_final_pmpp, conf_pmwd, conf_pmpp)

    dens_pmwd, dens_vjp_pmwd = jax.vjp(lambda p: scatter_pmwd(p, conf_pmwd), ptcl_final_pmwd)
    dens_pmpp, dens_vjp_pmpp = jax.vjp(lambda p: scatter_pmpp(p, conf_pmpp), ptcl_final_pmpp)
    probe = _linear_probe(dens_pmwd.shape, conf_pmwd.float_dtype)
    ptcl_cot_final_pmwd, = dens_vjp_pmwd(probe)
    ptcl_cot_final_pmpp, = dens_vjp_pmpp(probe)

    print("One-step full N-body forward compare")
    _print_metrics("density", np.asarray(jax.device_get(dens_pmwd)), np.asarray(jax.device_get(dens_pmpp)))
    _print_metrics(
        "disp[first-slot]",
        np.asarray(jax.device_get(ptcl_final_pmwd.disp)),
        np.asarray(jax.device_get(ptcl_final_pmpp.disp))[first_slot],
    )
    _print_metrics(
        "vel[first-slot]",
        np.asarray(jax.device_get(ptcl_final_pmwd.vel)),
        np.asarray(jax.device_get(ptcl_final_pmpp.vel))[first_slot],
    )
    _print_metrics(
        "acc[first-slot]",
        np.asarray(jax.device_get(ptcl_final_pmwd.acc)),
        np.asarray(jax.device_get(ptcl_final_pmpp.acc))[first_slot],
    )

    ptcl_cot_final_pmpp_agg = _aggregate_ptcl_cot(
        ptcl_cot_final_pmpp, ptcl_final_pmpp, conf_pmpp, ptcl_final_pmwd, conf_pmwd, "mean"
    )
    _compare_ptcl_fields("Scatter VJP particle cotangent", ptcl_cot_final_pmwd, ptcl_cot_final_pmpp_agg)

    a_prev = conf_pmwd.a_nbody[0]
    a_next = conf_pmwd.a_nbody[-1]

    pmwd_init = nbody_adj_init_pmwd(a_next, ptcl_final_pmwd, ptcl_cot_final_pmwd, None, cosmo_pmwd, conf_pmwd)
    pmpp_init = nbody_adj_init_pmpp(a_next, ptcl_final_pmpp, ptcl_cot_final_pmpp, cosmo_pmpp, conf_pmpp)
    ptcl_init_pmwd, ptcl_cot_init_pmwd, cosmo_cot_init_pmwd, cosmo_force_init_pmwd = pmwd_init
    ptcl_init_pmpp, ptcl_cot_init_pmpp, cosmo_cot_init_pmpp, cosmo_force_init_pmpp = pmpp_init

    _compare_ptcl_fields(
        "After nbody_adj_init ptcl_cot",
        ptcl_cot_init_pmwd,
        _aggregate_ptcl_cot(ptcl_cot_init_pmpp, ptcl_init_pmpp, conf_pmpp, ptcl_init_pmwd, conf_pmwd, "mean"),
    )
    _print_cosmo_diff("After nbody_adj_init cosmo_cot", cosmo_cot_init_pmwd, cosmo_cot_init_pmpp)
    _print_cosmo_diff(
        "After nbody_adj_init cosmo_cot_force",
        cosmo_force_init_pmwd,
        cosmo_force_init_pmpp,
    )

    pmwd_step = nbody_adj_step_pmwd(
        a_prev,
        a_next,
        ptcl_init_pmwd,
        ptcl_cot_init_pmwd,
        None,
        cosmo_pmwd,
        cosmo_cot_init_pmwd,
        cosmo_force_init_pmwd,
        conf_pmwd,
    )
    pmpp_step = nbody_adj_step_pmpp(
        a_prev,
        a_next,
        ptcl_init_pmpp,
        ptcl_cot_init_pmpp,
        cosmo_pmpp,
        cosmo_cot_init_pmpp,
        cosmo_force_init_pmpp,
        conf_pmpp,
    )
    ptcl_step_pmwd, ptcl_cot_step_pmwd, cosmo_cot_step_pmwd, cosmo_force_step_pmwd = pmwd_step
    ptcl_step_pmpp, ptcl_cot_step_pmpp, cosmo_cot_step_pmpp, cosmo_force_step_pmpp = pmpp_step

    _compare_ptcl_fields(
        "After nbody_adj_step ptcl_cot",
        ptcl_cot_step_pmwd,
        _aggregate_ptcl_cot(ptcl_cot_step_pmpp, ptcl_step_pmpp, conf_pmpp, ptcl_step_pmwd, conf_pmwd, "sum"),
    )
    _print_cosmo_diff("After nbody_adj_step cosmo_cot", cosmo_cot_step_pmwd, cosmo_cot_step_pmpp)
    _print_cosmo_diff(
        "After nbody_adj_step cosmo_cot_force",
        cosmo_force_step_pmwd,
        cosmo_force_step_pmpp,
    )

    print("\nManual reverse substeps inside integrate_adj")
    K = D = 0
    a_disp_pmwd = a_vel_pmwd = a_acc_pmwd = a_prev
    a_disp_pmpp = a_vel_pmpp = a_acc_pmpp = a_prev
    ptcl_pw = ptcl_init_pmwd
    ptcl_pp = ptcl_init_pmpp
    ptcl_cot_pw = ptcl_cot_init_pmwd
    ptcl_cot_pp = ptcl_cot_init_pmpp
    cosmo_cot_pw = cosmo_cot_init_pmwd
    cosmo_cot_pp = cosmo_cot_init_pmpp
    cosmo_force_pw = cosmo_force_init_pmwd
    cosmo_force_pp = cosmo_force_init_pmpp

    for split_index, (d, k) in enumerate(reversed(conf_pmwd.symp_splits), start=1):
        if k != 0:
            K += k
            a_vel_next_pw = a_prev * (1 - K) + a_next * K
            a_vel_next_pp = a_prev * (1 - K) + a_next * K
            ptcl_pw, ptcl_cot_pw, cosmo_cot_pw = kick_adj_pmwd(
                a_acc_pmwd,
                a_vel_pmwd,
                a_vel_next_pw,
                ptcl_pw,
                ptcl_cot_pw,
                cosmo_pmwd,
                cosmo_cot_pw,
                cosmo_force_pw,
                conf_pmwd,
            )
            ptcl_pp, ptcl_cot_pp, cosmo_cot_pp = kick_adj_pmpp(
                a_acc_pmpp,
                a_vel_pmpp,
                a_vel_next_pp,
                ptcl_pp,
                ptcl_cot_pp,
                cosmo_pmpp,
                cosmo_cot_pp,
                cosmo_force_pp,
                conf_pmpp,
            )
            a_vel_pmwd = a_vel_next_pw
            a_vel_pmpp = a_vel_next_pp
            _compare_ptcl_fields(
                f"split {split_index} after kick_adj ptcl_cot",
                ptcl_cot_pw,
                _aggregate_ptcl_cot(ptcl_cot_pp, ptcl_pp, conf_pmpp, ptcl_pw, conf_pmwd, "mean"),
            )
            _print_cosmo_diff(f"split {split_index} after kick_adj cosmo_cot", cosmo_cot_pw, cosmo_cot_pp)

        if d != 0:
            D += d
            a_disp_next_pw = a_prev * (1 - D) + a_next * D
            a_disp_next_pp = a_prev * (1 - D) + a_next * D
            ptcl_pw, ptcl_cot_pw, cosmo_cot_pw = drift_adj_pmwd(
                a_vel_pmwd,
                a_disp_pmwd,
                a_disp_next_pw,
                ptcl_pw,
                ptcl_cot_pw,
                cosmo_pmwd,
                cosmo_cot_pw,
                conf_pmwd,
            )
            ptcl_pp, ptcl_cot_pp, cosmo_cot_pp = drift_adj_pmpp(
                a_vel_pmpp,
                a_disp_pmpp,
                a_disp_next_pp,
                ptcl_pp,
                ptcl_cot_pp,
                cosmo_pmpp,
                cosmo_cot_pp,
                conf_pmpp,
            )
            a_disp_pmwd = a_disp_next_pw
            a_disp_pmpp = a_disp_next_pp
            _compare_ptcl_fields(
                f"split {split_index} after drift_adj ptcl_cot",
                ptcl_cot_pw,
                _aggregate_ptcl_cot(ptcl_cot_pp, ptcl_pp, conf_pmpp, ptcl_pw, conf_pmwd, "sum"),
            )
            _print_cosmo_diff(f"split {split_index} after drift_adj cosmo_cot", cosmo_cot_pw, cosmo_cot_pp)

            ptcl_pw, ptcl_cot_pw, cosmo_force_pw = force_adj_pmwd(
                a_disp_pmwd,
                ptcl_pw,
                _coerce_ptcl_cot(ptcl_cot_pw, conf_pmwd.float_dtype),
                cosmo_pmwd,
                conf_pmwd,
            )
            try:
                ptcl_pp, ptcl_cot_pp, cosmo_force_pp = force_adj_pmpp(
                    a_disp_pmpp,
                    ptcl_pp,
                    _coerce_ptcl_cot(ptcl_cot_pp, conf_pmpp.float_dtype),
                    cosmo_pmpp,
                    conf_pmpp,
                )
            except TypeError as exc:
                print(f"\nsplit {split_index} after force_adj PMPP standalone call hit float0: {exc}")
                break
            a_acc_pmwd = a_disp_pmwd
            a_acc_pmpp = a_disp_pmpp
            _compare_ptcl_fields(
                f"split {split_index} after force_adj ptcl_cot",
                ptcl_cot_pw,
                _aggregate_ptcl_cot(ptcl_cot_pp, ptcl_pp, conf_pmpp, ptcl_pw, conf_pmwd, "sum"),
            )
            _print_cosmo_diff(f"split {split_index} after force_adj cosmo_cot_force", cosmo_force_pw, cosmo_force_pp)


if __name__ == "__main__":
    main()
