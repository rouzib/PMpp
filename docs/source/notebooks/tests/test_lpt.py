#%%
#%%
import os

import sys
import os

try:
    os.chdir("/home/r/rouzib/links/projects/aip-lplevass/rouzib/pmpp/")
    local = False
except:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    local = True


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
#%%
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import matplotlib.pyplot as plt
#%%
# from pmwd import white_noise as white_noise_pmwd, linear_modes as linear_modes_pmwd, lpt as lpt_pmwd
# from pmwd import Particles as Particles_pmwd, Configuration as Configuration_pmwd, scatter as scatter_pmwd, SimpleLCDM, \
#     boltzmann as boltzmann_pmwd, nbody as nbody_pmwd

from src.boltzmann import boltzmann
from src.configuration import Configuration
from src.cosmo import SimpleLCDM
from src.scatter import scatter
from src.lpt import lpt
from src.nbody import nbody
from src.particles import Particles
from src.utils import create_compute_mesh
from src.modes import linear_modes, white_noise
from src.utils import AXIS_NAME
#%%
box_size = 1000
seed = 0
mesh_shape = 1
#%%
def init_conf(num_ptcl, mesh_shape, num_devices=None, max_ptcl_per_slice=1.2, max_share_ptcl=20000,
              max_share_gather_ptcl=50000):
    ptcl_grid_shape = (num_ptcl,) * 3
    ptcl_spacing = box_size / ptcl_grid_shape[0]

    if num_devices:
        num_devices = min(num_devices, len(jax.devices()))
    else:
        num_devices = len(jax.devices())

    compute_mesh = create_compute_mesh(jax.devices()[:num_devices])
    conf_mGPU = Configuration(ptcl_spacing, ptcl_grid_shape, mesh_shape=mesh_shape, compute_mesh=compute_mesh,
                              max_ptcl_per_slice=int(num_ptcl * num_ptcl * num_ptcl / num_devices * max_ptcl_per_slice),
                              max_share_ptcl=max_share_ptcl,
                              max_share_gather_ptcl=max_share_gather_ptcl, lpt_order=1)

    return conf_mGPU


conf = init_conf(512, mesh_shape, num_devices=None, max_ptcl_per_slice=1.05, max_share_ptcl=80000,
                 max_share_gather_ptcl=120000)
#%%
#%%
@jax.jit
def get_lpt(conf, seed):
    cosmo = SimpleLCDM(conf)
    cosmo = boltzmann(cosmo, conf)
    modes = white_noise(seed, conf)
    modes = linear_modes(modes, cosmo, conf)
    conf_mGPU_lpt = conf.replace(max_share_ptcl=conf.max_share_ptcl * 2)
    ptcl_lpt_mGPU = lpt(modes, cosmo, conf_mGPU_lpt)
    return scatter(ptcl_lpt_mGPU, conf).sum(1)
#%%
ptcl_lpt = get_lpt(conf, 0)
#%%
#%%
# #%%
# @jax.jit
# def get_lpt_pmwd(conf, seed):
#     cosmo_pmwd = SimpleLCDM(conf)
#     cosmo_pmwd = boltzmann_pmwd(cosmo_pmwd, conf)
#     modes_pmwd = white_noise_pmwd(seed, conf)
#     modes_pmwd = linear_modes_pmwd(modes_pmwd, cosmo_pmwd, conf)
#     ptcl_lpt_pmwd, _ = lpt_pmwd(modes_pmwd, cosmo_pmwd, conf)
#     conf_pmwd = Configuration_pmwd(ptcl_spacing=conf.ptcl_spacing, ptcl_grid_shape=conf.ptcl_grid_shape,
#                                mesh_shape=conf.mesh_shape)
#     return scatter_pmwd(ptcl_lpt_pmwd, conf_pmwd).sum(1)
# #%%
# ptcl_lpt_pmwd = get_lpt_pmwd(conf, 0)
# #%%
# #%%
# #%%
# print(ptcl_lpt.shape)
# print(ptcl_lpt_pmwd.shape)
# #%%
# plt.imshow(ptcl_lpt)
# plt.colorbar()
# #%%
# plt.imshow(ptcl_lpt_pmwd)
# plt.colorbar()
# #%%
# plt.imshow(jnp.log(jnp.abs(ptcl_lpt_pmwd - ptcl_lpt)))
# plt.colorbar()
# plt.show()
# #%%
