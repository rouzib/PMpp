"""Composite potential corrections."""

from dataclasses import field
from functools import partial

import jax.numpy as jnp

from ..utils import pytree_dataclass
from .mesh_cnn import MeshCNNPotentialCorrection
from .radial import RadialPotentialCorrection


@partial(
    pytree_dataclass,
    aux_fields=("dtype",),
    frozen=True,
    eq=False,
)
class CombinedPotentialCorrection:
    radial: RadialPotentialCorrection
    mesh_cnn: MeshCNNPotentialCorrection
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))
