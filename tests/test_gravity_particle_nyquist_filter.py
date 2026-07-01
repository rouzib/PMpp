import numpy as np

import jax
import jax.numpy as jnp

from pmpp.configuration import Configuration
from pmpp.cosmo import SimpleLCDM
from pmpp.gravity import gravity
from pmpp.particles import Particles


def _sinusoid_positions(res, box_size, amplitude):
    axis = np.arange(res, dtype=np.float32) * np.float32(box_size / res)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    pos = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    pos[:, 0] = (
        pos[:, 0]
        + np.float32(amplitude) * np.sin(np.float32(2 * np.pi / box_size) * pos[:, 0])
    ) % np.float32(box_size)
    return pos


def _x_force_modes(mesh_shape):
    res = 16
    box_size = 1000.0
    conf = Configuration(
        box_size / res,
        (res, res, res),
        mesh_shape=mesh_shape,
        a_start=1 / 128,
        a_stop=1.0,
        a_nbody_maxstep=1 / 20,
    )
    pos = _sinusoid_positions(res, box_size, amplitude=0.5)
    ptcl = Particles.from_ordered_pos(
        conf,
        jnp.asarray(pos),
        vel=jnp.zeros_like(jnp.asarray(pos)),
    )
    acc = np.asarray(jax.device_get(gravity(jnp.asarray(1.0), ptcl, SimpleLCDM(conf), conf)))
    mean_x_force = acc[:, 0].reshape(res, res, res).mean(axis=(1, 2))
    return np.abs(np.fft.rfft(mean_x_force)) / (res / 2)


def test_refined_force_mesh_does_not_alias_particle_lattice():
    base_modes = _x_force_modes(mesh_shape=1)
    base_fundamental = base_modes[1]

    for mesh_shape in (2, 4):
        modes = _x_force_modes(mesh_shape=mesh_shape)
        assert np.isclose(modes[1], base_fundamental, rtol=0.03)
        assert modes[2] < 0.10 * base_fundamental
