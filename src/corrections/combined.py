"""Composite potential corrections."""

from dataclasses import field
from functools import partial
from typing import Optional

import jax.numpy as jnp

from ..utils import pytree_dataclass
from .mesh_cnn import MeshCNNPotentialCorrection
from .radial import RadialPotentialCorrection
from .window import PMWindowCompensationCorrection


@partial(
    pytree_dataclass,
    aux_fields=("dtype",),
    frozen=True,
    eq=False,
)
class CombinedPotentialCorrection:
    """Composite correction made from independent correction blocks.

    The Fourier-space blocks, ``window`` and ``radial``, are multiplicative and
    commute. The optional mesh CNN is applied after those transfers because it
    predicts an additive real-space potential residual.
    """

    radial: Optional[RadialPotentialCorrection] = None
    mesh_cnn: Optional[MeshCNNPotentialCorrection] = None
    window: Optional[PMWindowCompensationCorrection] = None
    dtype: jnp.dtype = field(default=jnp.float32, repr=False)

    def __post_init__(self):
        if self._is_transforming():
            return
        object.__setattr__(self, "dtype", jnp.dtype(self.dtype))
