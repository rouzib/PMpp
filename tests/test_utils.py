import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax

from pmpp.configuration import Configuration
from pmpp.multigpu_configuration import MultiGPUConfiguration
from pmpp.utils import create_compute_mesh


def init_conf(num_ptcl, mesh_shape, box_size, num_devices=None, max_ptcl_per_slice=1.2, max_share_ptcl=20000,
              max_share_gather_ptcl=50000, multigpu_mode="mesh_halo",
              particle_halo_gather_mesh_halo=False):
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / ptcl_grid_shape[0]

    if num_devices:
        num_devices = min(num_devices, len(jax.devices()))
    else:
        num_devices = len(jax.devices())

    compute_mesh = create_compute_mesh(jax.devices()[:num_devices])
    conf_mGPU = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=mesh_shape,
                              multigpu=MultiGPUConfiguration(
                                  compute_mesh=compute_mesh,
                                  mode=multigpu_mode,
                              ),
                              max_ptcl_per_slice=int(num_ptcl * num_ptcl * num_ptcl / num_devices * max_ptcl_per_slice),
                              max_share_ptcl=max_share_ptcl,
                              max_share_gather_ptcl=max_share_gather_ptcl,
                              particle_halo_gather_mesh_halo=particle_halo_gather_mesh_halo,
                              to_save_z=[1, 2 / 3, 1 / 3, 0],
                              a_start=1 / 60, a_nbody_maxstep=1 / 60, )

    return conf_mGPU
