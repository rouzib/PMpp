import math
from functools import partial
from typing import ClassVar, Optional, Tuple, Union, List, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax.sharding import Mesh
from jax.typing import ArrayLike, DTypeLike
from mcfit import TophatVar

from .FFT_distributed import create_ffts
from .fft import fftfreq
from .gather import initialize_mGPU_gather
from .halo_moving import (
    initialize_mGPU_compute_halo_mask,
    initialize_mGPU_halo_move_pullback,
    initialize_mGPU_halo_movement_canonical,
    initialize_mGPU_reconstruct_pre_drift,
)
from .scatter import initialize_mGPU_scatter
from .utils import pytree_dataclass, build_ring_permutations


@partial(pytree_dataclass,
         aux_fields=Ellipsis,
         frozen=True,
         eq=False)
class Configuration:
    """Configuration parameters, "immutable" as a frozen dataclass.

    Parameters
    ----------
    ptcl_spacing : float
        Lagrangian particle grid cell size in [L].
    ptcl_grid_shape : tuple of int
        Lagrangian particle grid shape, in ``len(ptcl_grid_shape)`` spatial dimensions.
    mesh_shape : int, float, or tuple of int, optional
        Mesh shape. If an int or float, it is used as the 1D mesh to particle grid shape
        ratio, to determine the mesh shape from that of the particle grid. The mesh grid
        cannot be smaller than the particle grid (int or float values must not be
        smaller than 1) and the two grids must have the same aspect ratio.
    cosmo_dtype : DTypeLike, optional
        Float dtype for Cosmology and Configuration.
    pmid_dtype : DTypeLike, optional
        Signed integer dtype for particle or mesh grid indices.
    float_dtype : DTypeLike, optional
        Float dtype for other particle and mesh quantities.
    k_pivot_Mpc : float, optional
        Primordial scalar power spectrum pivot scale in 1/Mpc.
    T_cmb : float, optional
        CMB temperature in K.
    M : float, optional
        Mass unit defined in kg/h. Default is 1e10 M☉/h.
    L : float, optional
        Length unit defined in m/h. Default is Mpc/h.
    T : float, optional
        Time unit defined in s/h. Default is Hubble time 1/H_0 ~ 1e10 years/h ~ age of
        the Universe.
    transfer_fit : bool, optional
        Whether to use Eisenstein & Hu fit to transfer function. Default is True
        (subject to change when False is implemented).
    transfer_fit_nowiggle : bool, optional
        Whether to use non-oscillatory transfer function fit.
    transfer_lgk_min : float, optional
        Minimum transfer function wavenumber in [1/L] in log10.
    transfer_lgk_max : float, optional
        Maximum transfer function wavenumber in [1/L] in log10.
    transfer_lgk_maxstep : float, optional
        Maximum transfer function wavenumber step size in [1/L] in log10. It determines
        the number of wavenumbers ``transfer_k_num``, the actual step size
        ``transfer_lgk_step``, and the wavenumbers ``transfer_k``.
    growth_rtol : float, optional
        Relative tolerance for solving the growth ODEs.
    growth_atol : float, optional
        Absolute tolerance for solving the growth ODEs.
    growth_inistep: float, None, or 2-tuple of float or None, optional
        The initial step size for solving the growth ODEs. If None, use estimation. If a
        tuple, use the two step sizes for forward and reverse integrations,
        respectively.
    lpt_order : int, optional
        LPT order, with 1 for Zel'dovich approximation, 2 for 2LPT, and 3 for 3LPT.
    lpt_cache_strains : bool, optional
        Whether to cache the diagonal strain arrays in the 2LPT tidal tensor computation.
        When True (default), each unique strain is computed once and reused, saving
        ``dim - 1`` redundant 3D irfftn calls per ``_L`` evaluation at the cost of keeping
        ``dim`` extra strain arrays (each of shape ``ptcl_grid_shape``) alive simultaneously
        on the GPU. Set to False to trade compute for memory when GPU memory is tight.
    a_start : float, optional
        LPT scale factor and N-body starting time.
    a_stop : float, optional
        N-body stopping time (scale factor).
    a_lpt_maxstep : float, optional
        Maximum LPT light cone scale factor step size. It determines the number of steps
        ``a_lpt_num``, the actual step size ``a_lpt_step``, and the steps ``a_lpt``.
    a_nbody_maxstep : float, optional
        Maximum N-body time integration scale factor step size. It determines the number
        of steps ``a_nbody_num``, the actual step size ``a_nbody_step``, and the steps
        ``a_nbody``.
    symp_splits : tuple of float 2-tuples, optional
        Symplectic splitting method composition, with each 2-tuples being drift and then
        kick coefficients. Its adjoint has the same splits in reverse nested orders,
        i.e., kick and then drift. Default is the Newton-Störmer-Verlet-leapfrog method.
    chunk_size : int, optional
        Chunk size to split particles in batches in scatter and gather to save memory.

    Raises
    ------
    ValueError
        Incorrect or inconsistent parameter values.

    """

    ptcl_spacing: float
    ptcl_grid_shape: Tuple[int, ...]  # tuple[int, ...] for python >= 3.9 (PEP 585)

    # mGPU compute_mesh
    compute_mesh: Mesh = None

    mesh_shape: Union[float, Tuple[int, ...]] = 1
    nMesh: int = None

    cosmo_dtype: DTypeLike = jnp.dtype(jnp.float64)
    pmid_dtype: DTypeLike = jnp.dtype(jnp.int16)
    float_dtype: DTypeLike = jnp.dtype(jnp.float32)

    k_pivot_Mpc: float = 0.05

    T_cmb: float = 2.7255  # Fixsen 2009, arXiv:0911.1955

    # constants in SI units, as class variables
    M_sun_SI: ClassVar[float] = 1.98847e30  # solar mass in kg
    Mpc_SI: ClassVar[float] = 3.0856775815e22  # Mpc in m
    H_0_SI: ClassVar[float] = 1e5 / Mpc_SI  # Hubble constant in h/s
    c_SI: ClassVar[int] = 299792458  # speed of light in m/s
    G_SI: ClassVar[float] = 6.67430e-11  # Gravitational constant in m^3/kg/s^2

    # Units
    M: float = 1e10 * M_sun_SI
    L: float = Mpc_SI
    T: float = 1 / H_0_SI

    transfer_fit: bool = True
    transfer_fit_nowiggle: bool = False
    transfer_lgk_min: float = -4
    transfer_lgk_max: float = 3
    transfer_lgk_maxstep: float = 1 / 128

    growth_rtol: Optional[float] = None
    growth_atol: Optional[float] = None
    growth_inistep: Union[float, None,
    Tuple[Optional[float], Optional[float]]] = (1, None)

    lpt_order: int = 2
    lpt_cache_strains: bool = True

    a_start: float = 1 / 64
    a_start: float = 1 / 64
    a_stop: float = 1
    a_lpt_maxstep: float = 1 / 128
    a_nbody_maxstep: float = 1 / 64
    a_custom: Optional[List[float]] = None

    symp_splits: Tuple[Tuple[float, float], ...] = ((0, 0.5), (1, 0.5))

    chunk_size: int = 2 ** 24

    to_save_z: List[int] = None
    to_save_a: List[int] = None
    slice_to_save: List[int] = None
    max_slice_width: int = None

    # mGPU setup
    use_mGPU: bool = False
    num_devices: int = None
    devices: List[jax.Device] = None
    devices_index: List[int] = None

    local_mesh_shape: Tuple[int, ...] = None

    # mGPU halos
    ptcl_halo_width: int = None
    slice_start: List[int] = None
    slice_end: List[int] = None
    halo_start: ArrayLike = None
    halo_end: ArrayLike = None
    offsets: List[int] = None
    scatter_offsets: List[List[int]] = None

    max_ptcl_per_slice: int = None
    max_share_ptcl: int = 50000
    max_share_gather_ptcl: int = 200000

    left_perm: Tuple[Tuple[int, int]] = None
    right_perm: Tuple[Tuple[int, int]] = None

    # mGPU functions that need initialization
    mGPU_halo_moving: Callable = lambda x: x
    mGPU_reconstruct_pre_drift: Callable = lambda x: x
    mGPU_halo_move_pullback: Callable = lambda x: x
    mGPU_compute_halo_mask: Callable = lambda x: x
    mGPU_rfftn: Callable = lambda x: x
    mGPU_irfftn: Callable = lambda x: x
    mGPU_scatter: Callable = lambda x: x
    mGPU_gather: Callable = lambda x: x

    def __post_init__(self):
        if self._is_transforming():
            return

        if isinstance(self.mesh_shape, (int, float)):
            object.__setattr__(self, 'init_mesh_shape', self.mesh_shape)
            mesh_shape = tuple(round(s * self.mesh_shape) for s in self.ptcl_grid_shape)
            object.__setattr__(self, 'mesh_shape', mesh_shape)
        if len(self.ptcl_grid_shape) != len(self.mesh_shape):
            raise ValueError('particle and mesh grid dimensions differ')
        if any(sm < sp for sp, sm in zip(self.ptcl_grid_shape, self.mesh_shape)):
            raise ValueError('mesh grid cannot be smaller than particle grid')
        if any(self.ptcl_grid_shape[0] * sm != self.mesh_shape[0] * sp
               for sp, sm in zip(self.ptcl_grid_shape[1:], self.mesh_shape[1:])):
            raise ValueError('particle and mesh grid aspect ratios differ')

        object.__setattr__(self, 'cosmo_dtype', jnp.dtype(self.cosmo_dtype))
        object.__setattr__(self, 'pmid_dtype', jnp.dtype(self.pmid_dtype))
        object.__setattr__(self, 'float_dtype', jnp.dtype(self.float_dtype))
        if not jnp.issubdtype(self.cosmo_dtype, jnp.floating):
            raise ValueError('cosmo_dtype must be floating point numbers')
        if not jnp.issubdtype(self.pmid_dtype, jnp.signedinteger):
            raise ValueError('pmid_dtype must be signed integers')
        if not jnp.issubdtype(self.float_dtype, jnp.floating):
            raise ValueError('float_dtype must be floating point numbers')

        if self.a_custom is not None:
            object.__setattr__(self, 'a_custom', jnp.array(self.a_custom))

        # ~ 1.5e-8 for float64, 3.5e-4 for float32
        growth_tol = math.sqrt(jnp.finfo(self.cosmo_dtype).eps)
        if self.growth_rtol is None:
            object.__setattr__(self, 'growth_rtol', growth_tol)
        if self.growth_atol is None:
            object.__setattr__(self, 'growth_atol', growth_tol)

        if any(len(s) != 2 for s in self.symp_splits):
            raise ValueError(f'symp_splits={self.symp_splits} not supported')
        symp_splits_sum = tuple(sum(s) for s in zip(*self.symp_splits))
        if symp_splits_sum != (1, 1):
            raise ValueError(f'sum of symplectic splits = {symp_splits_sum} != (1, 1)')

        with jax.ensure_compile_time_eval():
            object.__setattr__(self, "kvec", fftfreq(self.mesh_shape, self.cell_size, dtype=self.float_dtype))
            object.__setattr__(self, "kvec_spacing",
                               fftfreq(self.ptcl_grid_shape, self.ptcl_spacing, dtype=self.float_dtype))

        object.__setattr__(self, "nMesh", self.mesh_shape[0])

        if self.to_save_z is None:
            object.__setattr__(self, "to_save_z", None)

        if self.to_save_a is None and self.to_save_z is not None:
            object.__setattr__(self, "to_save_a", list(1 / (1 + z) for z in self.to_save_z))

        if self.slice_to_save is not None:
            object.__setattr__(self, "slice_to_save", jnp.array(self.slice_to_save))


        # with jax.ensure_compile_time_eval():
        #     object.__setattr__(
        #         self,
        #         'var_tophat',
        #         TophatVar(self.transfer_k[1:], lowring=True, backend='jax'),
        #     )

        # mGPU setup
        if self.compute_mesh is not None:
            if self.compute_mesh.size < 1:
                raise ValueError(
                    f"mGPU used, but less than 1 device was set in the compute_mesh (actual: {self.compute_mesh.size}).")
            with jax.ensure_compile_time_eval():
                object.__setattr__(self, "use_mGPU", True)
                object.__setattr__(self, "num_devices", self.compute_mesh.size)
                object.__setattr__(self, "devices", self.compute_mesh.devices)
                object.__setattr__(self, "devices_index", list(device.id for device in self.devices))

                object.__setattr__(self, "local_mesh_shape",
                                   (self.mesh_shape[0] // self.num_devices, self.mesh_shape[1], self.mesh_shape[2]))

                global_nMesh = self.mesh_shape[0]
                ptcl_halo_width = 0 if self.num_devices == 1 else max(
                    1,
                    int(round(self.mesh_shape[0] / self.ptcl_grid_shape[0])),
                )
                object.__setattr__(self, "ptcl_halo_width", ptcl_halo_width)
                slice_start = list(
                    (global_nMesh // self.num_devices * device_idx) % global_nMesh for device_idx in self.devices_index)
                slice_end = list((global_nMesh // self.num_devices * (device_idx + 1)) % global_nMesh for device_idx in
                                 self.devices_index)
                halo_start = list(
                    [(slice_s - ptcl_halo_width) % global_nMesh, slice_s] for slice_s in
                    slice_start)
                halo_end = list([(slice_e - ptcl_halo_width) % global_nMesh, slice_e] for slice_e in slice_end)
                object.__setattr__(self, "slice_start", jnp.array(halo_start)[:, 0])
                object.__setattr__(self, "slice_end",
                                   jnp.array(halo_end)[:, 1] if self.num_devices > 1 else jnp.array([global_nMesh]))
                object.__setattr__(self, "halo_start", jnp.array(halo_start))
                object.__setattr__(self, "halo_end", jnp.array(halo_end))

                offsets = jnp.array(slice_start)
                object.__setattr__(self, "offsets", offsets)
                object.__setattr__(self, "scatter_offsets", jnp.array([[o, 0., 0.] for o in offsets]) * self.cell_size)

                """print(f"slice_start = {self.slice_start}")
                print(f"slice_end = {self.slice_end}")
                print(f"halo_start = {self.halo_start}")
                print(f"halo_end = {self.halo_end}")"""

                if self.max_ptcl_per_slice is None:
                    scaling = 1.3 + (self.num_devices.bit_length() - 3) * 0.1
                    if self.num_devices == 1:
                        scaling = 1
                    object.__setattr__(self, "max_ptcl_per_slice",
                                       math.floor(self.ptcl_num // self.num_devices * scaling))
                if self.num_devices == 1:
                    object.__setattr__(self, "max_ptcl_per_slice",
                                       min(self.max_ptcl_per_slice, self.ptcl_num))
                object.__setattr__(self, "max_share_ptcl", min(self.max_share_ptcl, self.max_ptcl_per_slice // 2))
                object.__setattr__(self, "max_share_gather_ptcl", min(self.max_share_gather_ptcl,
                                                                      self.max_ptcl_per_slice // 2))

                left_perm, right_perm = build_ring_permutations(self.num_devices)
                object.__setattr__(self, "left_perm", left_perm)
                object.__setattr__(self, "right_perm", right_perm)

                # initialization of mGPU functions
                rfftn_jit, irfftn_jit, _, _ = create_ffts(self.compute_mesh)
                object.__setattr__(self, "mGPU_rfftn", rfftn_jit)
                object.__setattr__(self, "mGPU_irfftn", irfftn_jit)
                object.__setattr__(self, "mGPU_halo_moving", initialize_mGPU_halo_movement_canonical(self))
                object.__setattr__(self, "mGPU_reconstruct_pre_drift", initialize_mGPU_reconstruct_pre_drift(self))
                object.__setattr__(self, "mGPU_halo_move_pullback", initialize_mGPU_halo_move_pullback(self))
                object.__setattr__(self, "mGPU_compute_halo_mask", initialize_mGPU_compute_halo_mask(self))
                object.__setattr__(self, "mGPU_scatter", initialize_mGPU_scatter(self))
                object.__setattr__(self, "mGPU_gather", initialize_mGPU_gather(self))
        else:
            object.__setattr__(self, "local_mesh_shape", self.mesh_shape)
            object.__setattr__(self, "ptcl_halo_width", 0)

        # finalize
        dtype = self.cosmo_dtype
        for name, value in self.named_children():
            value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            object.__setattr__(self, name, value)

    @property
    def dim(self):
        """Spatial dimension."""
        return len(self.ptcl_grid_shape)

    @property
    def ptcl_cell_vol(self):
        """Lagrangian particle grid cell volume in [L^dim]."""
        return self.ptcl_spacing ** self.dim

    @property
    def ptcl_num(self):
        """Number of particles."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.ptcl_grid_shape).prod().item()

    @property
    def box_size(self):
        """Simulation box size tuple in [L]."""
        return tuple(self.ptcl_spacing * s for s in self.ptcl_grid_shape)

    @property
    def box_vol(self):
        """Simulation box volume in [L^dim]."""
        with jax.ensure_compile_time_eval():
            return jnp.array(self.box_size).prod().item()

    @property
    def cell_size(self):
        """Mesh cell size in [L]."""
        return self.ptcl_spacing * self.ptcl_grid_shape[0] / self.mesh_shape[0]

    @property
    def disp_size(self):
        """Displacement size in [L]"""
        return 1 / self.cell_size

    @property
    def mesh_size(self):
        """Number of mesh grid points."""
        return math.prod(self.mesh_shape)

    @property
    def local_mesh_size(self):
        """Number of mesh grid points."""
        return math.prod(self.local_mesh_shape)

    @property
    def V(self):
        """Velocity unit as [L/T]. Default is 100 km/s."""
        return self.L / self.T

    @property
    def H_0(self):
        """Hubble constant H_0 in [1/T]."""
        return self.H_0_SI * self.T

    @property
    def c(self):
        """Speed of light in [L/T]."""
        return self.c_SI / self.V

    @property
    def G(self):
        """Gravitational constant in [L^3 / M / T^2]."""
        return self.G_SI * self.M / (self.L * self.V ** 2)

    @property
    def rho_crit(self):
        """Critical density in [M / L^3]."""
        return 3 * self.H_0 ** 2 / (8 * jnp.pi * self.G)

    @property
    def transfer_k_num(self):
        """Number of transfer function wavenumbers, including a leading 0."""
        return 1 + math.ceil((self.transfer_lgk_max - self.transfer_lgk_min)
                             / self.transfer_lgk_maxstep) + 1

    @property
    def transfer_lgk_step(self):
        """Transfer function wavenumber step size in [1/L] in log10."""
        return ((self.transfer_lgk_max - self.transfer_lgk_min)
                / (self.transfer_k_num - 2))

    @property
    def transfer_k(self):
        """Transfer function wavenumbers in [1/L], of ``cosmo_dtype``."""
        k = jnp.logspace(self.transfer_lgk_min, self.transfer_lgk_max,
                         num=self.transfer_k_num - 1, dtype=self.cosmo_dtype)
        return jnp.concatenate((jnp.array([0]), k))

    @property
    def a_lpt_num(self):
        """Number of LPT light cone scale factor steps, excluding ``a_start``."""
        return math.ceil(self.a_start / self.a_lpt_maxstep)

    @property
    def a_lpt_step(self):
        """LPT light cone scale factor step size."""
        return self.a_start / self.a_lpt_num

    @property
    def a_nbody_num(self):
        """Number of N-body time integration scale factor steps, excluding ``a_start``."""
        return math.ceil((self.a_stop - self.a_start) / self.a_nbody_maxstep)

    @property
    def a_nbody_step(self):
        """N-body time integration scale factor step size."""
        return (self.a_stop - self.a_start) / self.a_nbody_num

    @property
    def a_lpt(self):
        """LPT light cone scale factor steps, including ``a_start``, of ``cosmo_dtype``."""
        return jnp.linspace(0, self.a_start, num=self.a_lpt_num + 1,
                            dtype=self.cosmo_dtype)

    @property
    def a_nbody(self):
        """N-body time integration scale factor steps, including ``a_start``, of ``cosmo_dtype``."""
        if self.a_custom is not None:
            return self.a_custom
        return jnp.linspace(self.a_start, self.a_stop, num=1 + self.a_nbody_num,
                            dtype=self.cosmo_dtype)

    @property
    def growth_a(self):
        """Growth function scale factors, for both LPT and N-body, of ``cosmo_dtype``."""
        return jnp.concatenate((self.a_lpt, self.a_nbody[1:]))

    @property
    def var_tophat(self):
        with jax.ensure_compile_time_eval():
            return TophatVar(self.transfer_k[1:], lowring=True, backend='jax')

    @property
    def varlin_R(self):
        """Linear matter overdensity variance in a top-hat window of radius R in [L], of ``cosmo_dtype``."""
        return self.var_tophat.y
