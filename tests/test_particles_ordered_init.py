import numpy as np

import jax.numpy as jnp

from src.configuration import Configuration
from src.particles import Particles
from src.utils import pmid_to_idx


def test_from_ordered_pos_preserves_unique_grid_anchor_under_collisions():
    conf = Configuration(
        ptcl_spacing=1.0,
        ptcl_grid_shape=(4, 4, 4),
        mesh_shape=1,
        float_dtype=jnp.float32,
    )

    axes = [jnp.arange(n, dtype=conf.float_dtype) * conf.ptcl_spacing for n in conf.ptcl_grid_shape]
    pos = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, conf.dim)

    # Particles 0 and 16 share the same y/z coordinates but differ in x.
    # Force them into the same rounded Eulerian mesh cell to trigger a from_pos collision.
    pos = pos.at[0, 0].set(jnp.float32(0.6))
    pos = pos.at[16, 0].set(jnp.float32(0.6))

    ptcl_colliding = Particles.from_pos(conf, pos)
    idx_colliding = np.asarray(pmid_to_idx(ptcl_colliding.pmid, conf))
    assert np.unique(idx_colliding).size < conf.ptcl_num

    ptcl_ordered = Particles.from_ordered_pos(conf, pos)
    idx_ordered = np.asarray(pmid_to_idx(ptcl_ordered.pmid, conf))
    assert np.unique(idx_ordered).size == conf.ptcl_num

    pos_ordered = np.asarray(ptcl_ordered.pos(dtype=conf.float_dtype))
    assert np.allclose(pos_ordered, np.asarray(pos), atol=1e-6, rtol=1e-6)
