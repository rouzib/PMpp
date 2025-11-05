from dataclasses import field
from functools import partial
from itertools import accumulate
from operator import add, sub, itemgetter, mul
from typing import Optional, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import NamedSharding
from jax.tree_util import tree_map
from jax.experimental.shard_map import shard_map
from jax.typing import ArrayLike
from jax.sharding import PartitionSpec as P

from .plotting_utils import plot_particle_bins_callback
from .utils import pytree_dataclass, is_float0_array, raise_error, distribute_array_on_gpus, AXIS_NAME


@partial(pytree_dataclass, aux_fields=("conf",), frozen=True, eq=False)
class Particles:
    """Particle state.

    Particles are indexable.

    Array-likes are converted to ``jax.Array`` of ``conf.pmid_dtype`` or
    ``conf.float_dtype`` at instantiation.

    Parameters
    ----------
    conf : Configuration
        Configuration parameters.
    pmid : ArrayLike
        Particle IDs by mesh indices, of signed int dtype. They are the nearest mesh
        grid points from particles' Lagrangian positions. It can save memory compared to
        the raveled particle IDs, e.g., 6 bytes for 3 times int16 versus 8 bytes for
        uint64. Call ``raveled_id`` for the raveled IDs.
    disp : ArrayLike
        Particle comoving displacements from pmid in [L]. For displacements from
        particles' grid Lagrangian positions, use ``ptcl_rpos(ptcl,
        Particles.gen_grid(ptcl.conf), ptcl.conf)``. It can save the particle locations
        with much more uniform precision than positions, whereever they are. Call
        ``pos`` for the positions.
    vel : ArrayLike, optional
        Particle canonical velocities in [H_0 L].
    acc : ArrayLike, optional
        Particle accelerations in [H_0^2 L].
    attr : pytree, optional
        Particle attributes (custom features).

    """

    conf: "Configuration" = field(repr=False)

    pmid: ArrayLike
    disp: ArrayLike
    vel: Optional[ArrayLike] = None
    acc: Optional[ArrayLike] = None

    # mGPU attributes
    unused_index: Optional[ArrayLike] = None
    halo_mask: Optional[ArrayLike] = None
    idx: Optional[ArrayLike] = None

    attr: Any = None

    def __post_init__(self):

        def get_dtype_by_name(name):
            if name == "pmid":
                return conf.pmid_dtype
            elif name == "disp":
                return conf.float_dtype
            elif (name == "unused_index") | (name == "halo_mask"):
                return jnp.bool
            elif name == "idx":
                return jnp.int32
            else:
                return conf.float_dtype

        if self._is_transforming():
            return

        conf = self.conf
        for name, value in self.named_children():
            # dtype = conf.pmid_dtype if name == 'pmid' else conf.float_dtype
            dtype = get_dtype_by_name(name)
            if name == 'attr':
                value = tree_map(lambda x: jnp.asarray(x, dtype=dtype), value)
            else:
                value = (value if value is None or is_float0_array(value)
                         else jnp.asarray(value, dtype=dtype))
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.pmid)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)

    @staticmethod
    @jax.jit
    def particles_in_slice_mask(x_mod, slice_start, slice_end):
        """
        Compute a boolean mask for particles within a specified range of the global mesh.

        This function uses modular arithmetic and JAX's conditional operation
        to determine whether particles fall within a specified start and
        end range, considering cyclic boundaries.

        :param p: ndarray
            Particle positions. The first column of this array represents the
            x-coordinates of the particles.
        :param slice_start: int
            Start of the mesh slice.
        :param slice_end: int
            End of the mesh slice.
        :param global_nMesh: int
            The total size of the mesh, used for modular arithmetic.
        :return: ndarray
            A boolean mask indicating whether each particle lies within the
            specified slice.
        """
        """ This uses jax.lax.cond which is not replicated.
        return jax.lax.cond(
            slice_start > slice_end,
            lambda _: ((p[:, 0] % global_nMesh) >= slice_start) |
                      ((p[:, 0] % global_nMesh) < slice_end),
            lambda _: ((p[:, 0] % global_nMesh) >= slice_start) &
                      ((p[:, 0] % global_nMesh) < slice_end),
            operand=None
        )"""
        within_slice = (x_mod >= slice_start) & (x_mod < slice_end)
        across_boundary = (x_mod >= slice_start) | (x_mod < slice_end)
        return jnp.where(slice_start > slice_end, across_boundary, within_slice)

    @staticmethod
    @jax.jit
    def compute_halo_mask(x_mod, halo_start, halo_end, unused_indexes):
        """
        Computes a mask for particles based on their positions and halo boundaries.

        This function calculates a boolean mask for a set of particles represented
        by their positions. The mask is determined by checking whether the particles
        fall within certain defined halo regions (start or end). This computation
        takes into account periodic boundary conditions, leveraging modular arithmetic
        to support wrapping around the edges of the simulation space. The function
        also ensures particles with zero positions in the dataset are excluded from
        the resulting mask.

        :param p: Position array of particles with shape (n, m), where `n` is the
                  number of particles and `m` is the dimensionality of positions.
        :param halo_start: A tuple of two integers that define the start halo
                           boundaries as `(start, end)` in periodic space.
        :param halo_end: A tuple of two integers that define the end halo
                         boundaries as `(start, end)` in periodic space.
        :param global_nMesh: The size of the global periodic domain along the
                             position axis.
        :return: A boolean mask of shape `(n,)` indicating whether each particle
                 satisfies the given halo conditions.
        :rtype: jax.numpy.ndarray
        """

        def slice_mask(xm, start, end):
            """ Not replicated branching
            return jax.lax.cond(
                start > end,
                lambda _: (xm >= start) | (xm < end),
                lambda _: (xm >= start) & (xm < end),
                operand=None
            )"""
            within_range = (xm >= start) & (xm < end)
            across_boundary = (xm >= start) | (xm < end)
            return jnp.where(start > end, across_boundary, within_range)

        mask_start = slice_mask(x_mod, halo_start[0], halo_start[1])
        mask_end = slice_mask(x_mod, halo_end[0], halo_end[1])

        return (mask_start | mask_end) & ~unused_indexes

    @staticmethod
    def distribute_ptcl_pos(pmid, disp, vel, acc, conf, gpu_id):
        x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % conf.nMesh
        in_slice_mask = Particles.particles_in_slice_mask(x_mod, conf.slice_start[gpu_id], conf.slice_end[gpu_id])
        indices = jnp.compress(in_slice_mask, jnp.arange(pmid.shape[0]), axis=0, size=conf.max_ptcl_per_slice,
                               fill_value=-1)

        _ = jax.lax.cond(
            jnp.sum(in_slice_mask) > conf.max_ptcl_per_slice,
            lambda _: raise_error(
                "[ERROR] [GPU {a}] Exceeded max_ptcl_per_slice: "
                "max_ptcl_per_slice={x}, actual max_ptcl_per_slice={y}. Some particles may have "
                f"disappeared. Consider making 'conf.max_ptcl_per_slice' bigger so that this does not happen again.",
                a=gpu_id, x=conf.max_ptcl_per_slice, y=jnp.sum(in_slice_mask)),
            lambda _: None,
            operand=None
        )

        if vel is None:
            vel = jnp.zeros_like(disp)
        if acc is None:
            acc = jnp.zeros_like(disp)

        valid_count = jnp.minimum(jnp.sum(in_slice_mask), conf.max_ptcl_per_slice)

        def slice_particles(indices):
            pmid_sliced = jax.lax.gather(
                pmid,
                indices[:, None],
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,)
                ),
                slice_sizes=(1, pmid.shape[1])
            )  # Output shape: (indices.shape[0], pmid.shape[1])

            disp_sliced = jax.lax.gather(
                disp,
                indices[:, None],
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,)
                ),
                slice_sizes=(1, disp.shape[1])
            )  # Output shape: (indices.shape[0], disp.shape[1])

            vel_sliced = jax.lax.gather(
                vel,
                indices[:, None],
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,)
                ),
                slice_sizes=(1, vel.shape[1])
            )  # Output shape: (indices.shape[0], vel.shape[1])

            acc_sliced = jax.lax.gather(
                acc,
                indices[:, None],
                dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1,),
                    collapsed_slice_dims=(0,),
                    start_index_map=(0,)
                ),
                slice_sizes=(1, acc.shape[1])
            )  # Output shape: (indices.shape[0], acc.shape[1])

            # Replace invalid entries (-1 index) with zeros
            pmid_sliced = jax.lax.select(
                jnp.broadcast_to(indices[:, None] >= 0, pmid_sliced.shape),  # Broadcast condition
                pmid_sliced,
                jnp.zeros_like(pmid_sliced)
            )
            disp_sliced = jax.lax.select(
                jnp.broadcast_to(indices[:, None] >= 0, disp_sliced.shape),  # Broadcast condition
                disp_sliced,
                jnp.zeros_like(disp_sliced)
            )
            vel_sliced = jax.lax.select(
                jnp.broadcast_to(indices[:, None] >= 0, vel_sliced.shape),  # Broadcast condition
                vel_sliced,
                jnp.zeros_like(vel_sliced)
            )
            acc_sliced = jax.lax.select(
                jnp.broadcast_to(indices[:, None] >= 0, acc_sliced.shape),  # Broadcast condition
                acc_sliced,
                jnp.zeros_like(acc_sliced)
            )

            return pmid_sliced, disp_sliced, vel_sliced, acc_sliced

        # Define the no-particles branch
        def zero_slices():
            empty_pmid = jnp.zeros((conf.max_ptcl_per_slice, pmid.shape[1]),
                                   dtype=pmid.dtype)
            empty_disp = jnp.zeros((conf.max_ptcl_per_slice, disp.shape[1]),
                                   dtype=disp.dtype)
            empty_vel = jnp.zeros((conf.max_ptcl_per_slice, disp.shape[1]),
                                  dtype=vel.dtype)
            empty_acc = jnp.zeros((conf.max_ptcl_per_slice, disp.shape[1]),
                                  dtype=acc.dtype)

            # Use jax.lax.pvary on each output to annotate with no partitioning (empty tuple)
            empty_pmid = jax.lax.pvary(empty_pmid, (AXIS_NAME,))
            empty_disp = jax.lax.pvary(empty_disp, (AXIS_NAME,))
            empty_vel = jax.lax.pvary(empty_vel, (AXIS_NAME,))
            empty_acc = jax.lax.pvary(empty_acc, (AXIS_NAME,))

            return empty_pmid, empty_disp, empty_vel, empty_acc

        # Use lax.cond to handle the two cases (particles exist vs no particles)
        pmid_sliced, disp_sliced, vel_sliced, acc_sliced = jax.lax.cond(
            valid_count > 0,  # Condition: there are particles in the slice
            lambda _indices: slice_particles(indices),
            lambda _: zero_slices(),
            indices
        )

        unused_index = jnp.all(pmid_sliced == 0, axis=1) & jnp.all(disp_sliced == 0, axis=1)
        unused_index = unused_index.at[0].set(False)
        # unused_index = (indices == -1)
        x_mod = (pmid_sliced[:, 0] + disp_sliced[:, 0] * conf.disp_size) % conf.nMesh
        temp = [conf.halo_start[gpu_id, 0], (conf.halo_start[gpu_id, 1] - 1) % conf.nMesh]
        halo_mask = Particles.compute_halo_mask(x_mod, temp,
                                                conf.halo_end[gpu_id], unused_index)

        unused_index.astype(jnp.bool)
        halo_mask.astype(jnp.bool)
        indices.astype(jnp.int32)

        return pmid_sliced, disp_sliced, vel_sliced, acc_sliced, unused_index, halo_mask, indices

    @classmethod
    def from_pos(cls, conf, pos, vel=None, acc=None, wrap=True):
        """Construct particle state of ``pmid`` and ``disp`` from positions.

        There may be collisions in particle ``pmid``.

        Parameters
        ----------
        conf : Configuration
        pos : ArrayLike
            Particle positions in [L].
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        """
        pos = jnp.asarray(pos)

        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        unused_index, halo_mask, indices = None, None, None
        if conf.use_mGPU:
            def stack_and_shard(array: ArrayLike, dtype=jnp.float32):
                return distribute_array_on_gpus(jnp.concatenate(array, axis=0, dtype=dtype), conf.compute_mesh,
                                                P(AXIS_NAME))

            pmid_sliced_list, disp_sliced_list, vel_sliced_list, acc_sliced_list, unused_index_list, halo_mask_list, indices_list = [], [], [], [], [], [], []
            for i in range(conf.num_devices):
                gpu_id = conf.devices_index[i]  # jax.lax.axis_index('gpus')
                pmid_sliced, disp_sliced, vel_sliced, acc_sliced, unused_index, halo_mask, indices = cls.distribute_ptcl_pos(
                    pmid, disp, vel, acc, conf, gpu_id)

                pmid_sliced_list.append(pmid_sliced)
                disp_sliced_list.append(disp_sliced)
                vel_sliced_list.append(vel_sliced)
                acc_sliced_list.append(acc_sliced)
                unused_index_list.append(unused_index)
                halo_mask_list.append(halo_mask)
                indices_list.append(indices)

            pmid = stack_and_shard(pmid_sliced_list, conf.pmid_dtype)
            disp = stack_and_shard(disp_sliced_list, conf.float_dtype)
            unused_index = stack_and_shard(unused_index_list, jnp.bool)
            halo_mask = stack_and_shard(halo_mask_list, jnp.bool)
            indices = stack_and_shard(indices_list, jnp.int32)

            if vel is not None:
                vel = stack_and_shard(vel_sliced_list, conf.float_dtype)
            if acc is not None:
                acc = stack_and_shard(acc_sliced_list, conf.float_dtype)

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask, idx=indices)

    @classmethod
    def from_pos_sharded(cls, conf, pos, vel=None, acc=None, wrap=True):
        """Construct particle state of ``pmid`` and ``disp`` from positions."""
        pos = jnp.asarray(pos)

        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        unused_index, halo_mask, indices = None, None, None
        if conf.use_mGPU:
            @partial(shard_map, mesh=conf.compute_mesh, in_specs=(
                    P(None, None),
                    P(None, None),
                    P(None, None),
                    P(None, None),
                    None
            ), out_specs=(P(AXIS_NAME),
                          P(AXIS_NAME),
                          P(AXIS_NAME),
                          P(AXIS_NAME),
                          P(AXIS_NAME),
                          P(AXIS_NAME),
                          P(AXIS_NAME)
            ))
            def sharded_slicing(pmid, disp, vel, acc, conf):
                gpu_id = jax.lax.axis_index(AXIS_NAME)
                return cls.distribute_ptcl_pos(pmid, disp, vel, acc, conf, gpu_id)

            pmid, disp, vel, acc, unused_index, halo_mask, indices = sharded_slicing(pmid, disp, vel, acc, conf)
        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask, idx=indices)

    @classmethod
    def from_pmid(cls, conf, pmid, disp, vel=None, acc=None):
        """Construct particle state of ``pmid`` and ``disp`` from positions.

        There may be collisions in particle ``pmid``.

        Parameters
        ----------
        conf : Configuration
        pos : ArrayLike
            Particle positions in [L].
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        """
        pmid = jnp.asarray(pmid)
        disp = jnp.asarray(disp)

        unused_index, halo_mask, indices = None, None, None
        if conf.use_mGPU:
            def stack_and_shard(array: ArrayLike, dtype=jnp.float32):
                return distribute_array_on_gpus(jnp.concatenate(array, axis=0, dtype=dtype), conf.compute_mesh,
                                                P(AXIS_NAME))

            pmid_sliced_list, disp_sliced_list, vel_sliced_list, acc_sliced_list, unused_index_list, halo_mask_list, indices_list = [], [], [], [], [], [], []
            for i in range(conf.num_devices):
                gpu_id = conf.devices_index[i]  # jax.lax.axis_index('gpus')
                pmid_sliced, disp_sliced, vel_sliced, acc_sliced, unused_index, halo_mask, indices = cls.distribute_ptcl_pos(
                    pmid, disp, vel, acc, conf, gpu_id)

                pmid_sliced_list.append(pmid_sliced)
                disp_sliced_list.append(disp_sliced)
                vel_sliced_list.append(vel_sliced)
                acc_sliced_list.append(acc_sliced)
                unused_index_list.append(unused_index)
                halo_mask_list.append(halo_mask)
                indices_list.append(indices)

            pmid = stack_and_shard(pmid_sliced_list, conf.pmid_dtype)
            disp = stack_and_shard(disp_sliced_list, conf.float_dtype)
            unused_index = stack_and_shard(unused_index_list, jnp.bool)
            halo_mask = stack_and_shard(halo_mask_list, jnp.bool)
            indices = stack_and_shard(indices_list, jnp.int32)

            if vel is not None:
                vel = stack_and_shard(vel_sliced_list, conf.float_dtype)
            if acc is not None:
                acc = stack_and_shard(acc_sliced_list, conf.float_dtype)

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask, idx=indices)

    @classmethod
    def from_ptcl(cls, ptcl, conf=None, wrap=True):
        """Construct particle state of ``pmid`` and ``disp`` from positions.

        There may be collisions in particle ``pmid``.

        Parameters
        ----------
        conf : Configuration
        pos : ArrayLike
            Particle positions in [L].
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        """
        if conf is None:
            conf = ptcl.conf
        pmid = ptcl.pmid
        disp = ptcl.disp
        vel = ptcl.vel
        acc = ptcl.vel

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        unused_index, halo_mask, indices = None, None, None
        if conf.use_mGPU:
            def stack_and_shard(array: ArrayLike, dtype=jnp.float32):
                return distribute_array_on_gpus(jnp.concatenate(array, axis=0, dtype=dtype), conf.compute_mesh,
                                                P(AXIS_NAME))

            pmid_sliced_list, disp_sliced_list, vel_sliced_list, acc_sliced_list, unused_index_list, halo_mask_list, indices_list = [], [], [], [], [], [], []
            for i in range(conf.num_devices):
                gpu_id = conf.devices_index[i]  # jax.lax.axis_index('gpus')
                pmid_sliced, disp_sliced, vel_sliced, acc_sliced, unused_index, halo_mask, indices = cls.distribute_ptcl_pos(
                    pmid, disp, vel, acc, conf, gpu_id)

                pmid_sliced_list.append(pmid_sliced)
                disp_sliced_list.append(disp_sliced)
                vel_sliced_list.append(vel_sliced)
                acc_sliced_list.append(acc_sliced)
                unused_index_list.append(unused_index)
                halo_mask_list.append(halo_mask)
                indices_list.append(indices)

            pmid = stack_and_shard(pmid_sliced_list, conf.pmid_dtype)
            disp = stack_and_shard(disp_sliced_list, conf.float_dtype)
            unused_index = stack_and_shard(unused_index_list, jnp.bool)
            halo_mask = stack_and_shard(halo_mask_list, jnp.bool)
            indices = stack_and_shard(indices_list, jnp.int32)

            if vel is not None:
                vel = stack_and_shard(vel_sliced_list, conf.float_dtype)
            if acc is not None:
                acc = stack_and_shard(acc_sliced_list, conf.float_dtype)

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask, idx=indices)

    @classmethod
    def gen_grid(cls, conf, vel=False, acc=False):
        """Generate particles on a uniform grid with zero velocities.

        Parameters
        ----------
        conf : Configuration
        vel : bool, optional
            Whether to initialize velocities to zeros.
        acc : bool, optional
            Whether to initialize accelerations to zeros.

        """

        def build_local(gpu_id):
            pmid, disp = [], []

            sp = conf.ptcl_grid_shape[0]
            sm = conf.local_mesh_shape[0] + conf.mesh_shape[0] / sp
            n_devices = conf.num_devices
            n_ptcl = sp // n_devices + 1
            pmid_x = jnp.linspace(0, sm, num=n_ptcl, endpoint=False)
            offset = conf.offsets[gpu_id]
            pmid_x = jnp.rint(pmid_x)
            pmid_x = pmid_x.astype(conf.pmid_dtype)

            disp_x = jnp.arange(n_ptcl) * conf.mesh_shape[0] - pmid_x.astype(int) * sp
            disp_x *= conf.cell_size / sp
            disp_x = disp_x.astype(conf.float_dtype)

            pmid_x = pmid_x + offset - 1
            pmid_x = jnp.mod(pmid_x, conf.mesh_shape[0])

            pmid.append(pmid_x)
            disp.append(disp_x)

            for i, (sp, sm) in enumerate(zip(conf.ptcl_grid_shape[1:], conf.mesh_shape[1:])):
                pmid_yz = jnp.linspace(0, sm, num=sp, endpoint=False)
                pmid_yz = jnp.rint(pmid_yz)
                pmid_yz = pmid_yz.astype(conf.pmid_dtype)
                pmid.append(pmid_yz)

                # exact int arithmetic
                disp_yz = jnp.arange(sp) * sm - pmid_yz.astype(int) * sp
                disp_yz *= conf.cell_size / sp
                disp_yz = disp_yz.astype(conf.float_dtype)
                disp.append(disp_yz)

            pmid = jnp.meshgrid(*pmid, indexing='ij')
            pmid = jnp.stack(pmid, axis=-1).reshape(-1, conf.dim)
            pmid = jnp.pad(pmid, ((0, conf.max_ptcl_per_slice - pmid.shape[0]), (0, 0)), mode='constant')
            disp = jnp.meshgrid(*disp, indexing='ij')
            disp = jnp.stack(disp, axis=-1).reshape(-1, conf.dim)
            disp = jnp.pad(disp, ((0, conf.max_ptcl_per_slice - disp.shape[0]), (0, 0)), mode='constant')

            unused_index = jnp.zeros_like(disp[:, 0], dtype=jnp.bool,
                                          device=NamedSharding(conf.compute_mesh, P(AXIS_NAME))).at[
                           n_ptcl * sp * sp:].set(True)

            x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % conf.nMesh
            temp = [conf.halo_start[gpu_id, 0], (conf.halo_start[gpu_id, 1] - 1) % conf.nMesh]

            halo_mask = Particles.compute_halo_mask(x_mod, temp,
                                                    conf.halo_end[gpu_id], unused_index)
            # print(pmid[halo_mask])

            return pmid, disp, unused_index, halo_mask

        @partial(shard_map, mesh=conf.compute_mesh, in_specs=(P()),
                 out_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)))
        def build_all():
            axis = jax.lax.axis_index(AXIS_NAME)
            return build_local(axis)

        pmid, disp, unused_index, halo_mask = build_all()

        vel = jnp.zeros_like(disp, device=NamedSharding(conf.compute_mesh, P(AXIS_NAME))) if vel else None
        acc = jnp.zeros_like(disp, device=NamedSharding(conf.compute_mesh, P(AXIS_NAME))) if acc else None

        idx_dtype = jnp.int32
        mesh_shape = jnp.array(conf.mesh_shape, dtype=idx_dtype)  # (3,)
        ix = (pmid[:, 0].astype(idx_dtype)) % mesh_shape[0]
        iy = (pmid[:, 1].astype(idx_dtype)) % mesh_shape[1]
        iz = (pmid[:, 2].astype(idx_dtype)) % mesh_shape[2]

        # Flatten to a unique global ID in C-order: ((x * Ny) + y) * Nz + z
        idx = (ix * mesh_shape[1] + iy) * mesh_shape[2] + iz  # shape (N,)

        # Mask out padded entries so they don't collide with real IDs
        idx = jnp.where(unused_index, idx_dtype(-1), idx)

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask, idx=idx)

    def raveled_id(self, dtype=jnp.uint64, wrap=False):
        """Particle raveled IDs, flattened from ``pmid``.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Output int dtype.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        Returns
        -------
        raveled_id : jax.Array
            Particle raveled IDs.

        """
        conf = self.conf

        pmid = self.pmid
        if wrap:
            pmid = pmid % jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        strides = tuple(accumulate((1,) + conf.mesh_shape[:0:-1], mul))[::-1]

        raveled_id = sum(i.astype(dtype) * s for i, s in zip(pmid.T, strides))

        return raveled_id

    def pos(self, dtype=jnp.float64, wrap=True):
        """Particle positions.

        Parameters
        ----------
        dtype : DTypeLike, optional
            Output float dtype.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries.

        Returns
        -------
        pos : jax.Array
            Particle positions in [L].

        """
        conf = self.conf

        pos = self.pmid.astype(dtype)
        pos *= conf.cell_size
        pos += self.disp.astype(dtype)

        if wrap:
            pos %= jnp.array(conf.box_size, dtype=dtype)

        return pos

    @staticmethod
    def pmid_to_pos(pmid, disp, conf):
        pos = pmid * conf.cell_size + disp
        pos %= jnp.array(conf.box_size)
        return pos

    @staticmethod
    def pos_to_pmid(pos, conf):
        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)
        return pmid, disp

    def values_on_device(self, device_id):
        start_id = self.conf.max_ptcl_per_slice * device_id
        end_id = self.conf.max_ptcl_per_slice * (device_id + 1)

        pmid = self.pmid[start_id:end_id]
        disp = self.disp[start_id:end_id]
        vel = self.vel[start_id:end_id] if self.vel is not None else None
        acc = self.acc[start_id:end_id] if self.acc is not None else None

        # mGPU attributes
        unused_index = self.unused_index[start_id:end_id] if self.unused_index is not None else None
        halo_mask = self.halo_mask[start_id:end_id] if self.halo_mask is not None else None
        idx = self.idx[start_id:end_id] if self.idx is not None else None

        return pmid, disp, vel, acc, unused_index, halo_mask, idx

    @staticmethod
    @jax.jit
    def remove_particles(pmid, disp, vel, acc, particle_indices, mask, unused_index):
        """
        Removes particles from the given data arrays based on a boolean mask. This function
        operates on position vectors, velocity vectors, particle indices, and unused indices.
        Positions and velocities corresponding to masked particles are set to zero, particle
        indices are set to -1, and unused indices are set to True.

        :param pos: Array of particle positions. The shape of the array should align with
            the expected broadcast dimension for operations.
        :param vel: Array of particle velocities. Should match the shape and type specifications
            of `pos`.
        :param particle_indices: Array representing the unique indices assigned to each particle.
            Portions of the array will be modified for particles identified by the mask.
        :param mask: Boolean array indicating which particles should be removed. `True` values
            flag particles for removal at corresponding positions in `pos`, `vel`, and `particle_indices`.
        :param unused_index: Array of unused indices, which will also be updated when particles are
            removed as per the mask.
        :return: A tuple containing the updated position array (`pos`), velocity array (`vel`),
            particle indices array (`particle_indices`), and unused index array (`unused_index`)
            after particles are removed based on the mask.
        """
        broadcasted_mask = jnp.broadcast_to(jnp.expand_dims(mask, axis=-1), pmid.shape)
        zeros = jnp.zeros_like(disp)
        pmid = jax.lax.select(broadcasted_mask, jnp.zeros_like(pmid), pmid)
        disp = jax.lax.select(broadcasted_mask, zeros, disp)
        vel = jax.lax.select(broadcasted_mask, zeros, vel)
        acc = jax.lax.select(broadcasted_mask, zeros, acc)
        particle_indices = jax.lax.select(mask, -jnp.ones_like(particle_indices), particle_indices)
        unused_index = jax.lax.select(mask, jnp.ones_like(unused_index), unused_index)
        return pmid, disp, vel, acc, particle_indices, unused_index

    @staticmethod
    @partial(jax.jit, static_argnames=["max_values_to_add"])
    def add_particles(pmid, disp, vel, acc, particle_indices, unused_indexes, new_pmid, new_disp, new_vel, new_acc,
                      new_particle_indices,
                      max_values_to_add):
        """
        Add a specified number of new particle positions, velocities, and indices to the existing particle
        system, while ensuring that the number of added particles does not exceed the maximum limit. The
        function updates the particle system and the unused slot indices accordingly.

        :param pos: A 2D array representing the positions of existing particles. Each row corresponds to
                    a particle, and each column corresponds to a coordinate.
        :param vel: A 2D array representing the velocities of existing particles. Each row corresponds to
                    a particle, and each column corresponds to a speed component.
        :param particle_indices: A 1D array representing the indices of particles currently in use
                                 within the system.
        :param unused_indexes: A 1D Boolean array where True indicates the presence of available slots
                               for new particles and False indicates occupied slots.
        :param new_pos: A 2D array representing the positions of the new particles to be added. Each row
                        corresponds to a particle, and each column corresponds to a coordinate.
        :param new_vel: A 2D array representing the velocities of the new particles to be added. Each row
                        corresponds to a particle, and each column corresponds to a speed component.
        :param new_particle_indices: A 1D array representing the indices of the new particles to be added.
        :param max_values_to_add: An integer representing the maximum number of new particles to add
                                  to the system.
        :return: A tuple containing the updated positions, velocities, indices of the particles, and the
                 updated array of unused indices.
        """
        _ = jax.lax.cond(
            jnp.sum(unused_indexes) < max_values_to_add,
            lambda _: raise_error(
                "[ERROR] Exceeded max_amount_particles_per_slice. Available slots: {x}, values_to_add: {y}. Consider making 'max_amount_particles_per_slice' bigger.",
                x=jnp.sum(unused_indexes), y=max_values_to_add),
            lambda _: None,
            operand=None
        )

        all_indices = jnp.nonzero(unused_indexes, size=pmid.shape[0], fill_value=-1)[0]

        real_indices = jax.lax.dynamic_slice_in_dim(
            all_indices,
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )

        chosen_new_pmid = jax.lax.dynamic_slice_in_dim(
            new_pmid,
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_disp = jax.lax.dynamic_slice_in_dim(
            new_disp,
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_vel = jax.lax.dynamic_slice_in_dim(
            new_vel,
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_acc = jax.lax.dynamic_slice_in_dim(
            new_acc,
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_indices = jax.lax.dynamic_slice_in_dim(
            new_particle_indices,
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )

        pmid = pmid.at[real_indices].set(chosen_new_pmid)
        disp = disp.at[real_indices].set(chosen_new_disp)
        vel = vel.at[real_indices].set(chosen_new_vel)
        acc = acc.at[real_indices].set(chosen_new_acc)
        particle_indices = particle_indices.at[real_indices].set(chosen_new_indices)

        # unused_indexes = particle_indices == -1
        unused_indexes = jnp.all(pmid == 0, axis=1) & jnp.all(disp == 0, axis=1)

        return pmid, disp, vel, acc, particle_indices, unused_indexes

    @staticmethod
    def move_particles_shard_map(pmid, disp, vel, acc, particle_indices, halo_start, halo_end, previous_halo_mask,
                                 unused_indexes, share_only_right, global_nMesh, max_values_to_share, left_perm,
                                 right_perm, num_gpus,
                                 disp_size):
        """
        Finds and processes particles that need to move between GPUs based on their position relative to halo regions.
        This function handles particle sharing and removal to ensure a consistent distribution of particles across GPUs.

        :param pos: Current positions of particles.
        :param vel: Current velocities of particles.
        :param particle_indices: Indices associated with each particle.
        :param halo_start: Start boundaries of the halo regions.
        :param halo_end: End boundaries of the halo regions.
        :param previous_halo_mask: Mask indicating particles previously present in the halo region.
        :param global_nMesh: The global mesh size used for calculations.
        :param unused_indexes: Indices that can be reused for new particles.
        :param max_values_to_share: Maximum number of particles to share between neighboring GPUs.
        :param left_perm: The permutation mapping for particle exchange with the left neighboring GPU.
        :param right_perm: The permutation mapping for particle exchange with the right neighboring GPU.
        :param num_gpus: The number of GPUs that are used in the computation.
        :return: Updated positions, velocities, indices, previous halo mask, and unused indexes after processing.
        """
        halo_start = halo_start.squeeze()
        halo_end = halo_end.squeeze()

        halo_start_fix = [halo_start[0], (halo_start[1] - 1) % global_nMesh]

        offset = global_nMesh // num_gpus
        # dummy_mask = jnp.all(pmid == 0, axis=1) & jnp.all(disp == 0, axis=1)  # [n_particles], bool
        dummy_mask = unused_indexes

        x_mod = (pmid[:, 0] + disp[:, 0] * disp_size) % global_nMesh

        particles_in_halo = Particles.compute_halo_mask(x_mod, halo_start_fix, halo_end, unused_indexes)
        """diff = particles_in_halo.astype(int) - previous_halo_mask.astype(
            int)  # 1 if entered the halo and should be shared to the other gpus, -1 if particle exited the halo slice
        """
        in_gpu_slice = Particles.particles_in_slice_mask(x_mod, halo_start_fix[1],
                                                         halo_end[0])  # in the gpu slice, but not in the halo

        exited_halo = previous_halo_mask & ~particles_in_halo  # Particles that exited the halo_slice
        entered_halo = particles_in_halo & ~previous_halo_mask  # Particles that entered the halo_slice
        stayed_in_halo = particles_in_halo == previous_halo_mask  # Particles that stayed in the halo_slice
        # to_keep = (diff == -1) & in_gpu_slice  # do nothing with this
        to_share_and_remove = stayed_in_halo & (~in_gpu_slice) & ~dummy_mask & ~particles_in_halo
        to_remove = (
                            exited_halo & ~in_gpu_slice) | to_share_and_remove  # remove since this is the to_keep of another GPU
        to_share = entered_halo | to_share_and_remove

        to_share_left = to_share & Particles.particles_in_slice_mask(x_mod,
                                                                     (halo_start_fix[0] - offset) % global_nMesh,
                                                                     halo_start_fix[1])
        # jax.debug.print("{x}", x=((halo_start_fix[0] - offset) % global_nMesh, halo_start_fix[1]))
        to_share_right = to_share & Particles.particles_in_slice_mask(x_mod, halo_end[0],
                                                                      (halo_end[
                                                                           1] + offset) % global_nMesh) & ~share_only_right

        """jax.debug.print("[GPU {a}] Halo_start: {x}, Halo_end: {y}.", a=jax.lax.axis_index("gpus"),
                        x=halo_start_fix, y=halo_end)

        def plotting():
            p = (pmid + disp * disp_size) % global_nMesh

            plot_particle_bins_callback(p, None, global_nMesh, title_idx=0)  # "All particles"
            plot_particle_bins_callback(p, particles_in_halo, global_nMesh, title_idx=1)  # "Particles in halo"
            plot_particle_bins_callback(p, exited_halo, global_nMesh, title_idx=2)  # "Particles that exited halo_slice"
            plot_particle_bins_callback(p, entered_halo, global_nMesh, title_idx=3)  # "Particles that entered halo_slice"
            plot_particle_bins_callback(p, stayed_in_halo, global_nMesh,
                                        title_idx=4)  # "Particles that stayed in halo_slice"
            plot_particle_bins_callback(p, in_gpu_slice, global_nMesh, title_idx=5)  # "In GPU slice"
            plot_particle_bins_callback(p, ~in_gpu_slice & stayed_in_halo, global_nMesh,
                                        title_idx=6)  # "Out of GPU slice and not in halo"
            plot_particle_bins_callback(p, to_remove, global_nMesh, title_idx=7)  # "Particles that need to be removed"
            plot_particle_bins_callback(p, to_share, global_nMesh, title_idx=8)  # "Particles that need to be shared"
            plot_particle_bins_callback(p, to_share_left, global_nMesh,
                                        title_idx=9)  # "Particles that need to be shared to the left slice"
            plot_particle_bins_callback(p, to_share_right, global_nMesh,
                                        title_idx=10)  # "Particles that need to be shared to the right slice"

        jax.lax.cond(
            jax.lax.axis_index(AXIS_NAME) == 1,
            lambda: plotting(),
            lambda: None
        )"""

        check_fraction_and_share = (
                (jnp.sum(to_share_right) > max_values_to_share) |
                (jnp.sum(to_share_left) > max_values_to_share)
        )

        _ = jax.lax.cond(
            check_fraction_and_share,
            lambda _: raise_error(
                "[ERROR] [GPU {a}] Exceeded max_values_to_share: "
                "to_share_right={x}, to_share_left={y}, max_values_to_share={z}. Some particles may have "
                f"disappeared during the simulation. Consider making 'max_values_to_share' bigger so that this does not happen again.",
                a=jax.lax.axis_index('gpus'), x=jnp.sum(to_share_right), y=jnp.sum(to_share_left),
                z=max_values_to_share),
            lambda _: None,
            operand=None
        )

        to_share_right_pmid = jnp.compress(to_share_right, pmid, axis=0, size=max_values_to_share, fill_value=0)
        to_share_right_disp = jnp.compress(to_share_right, disp, axis=0, size=max_values_to_share, fill_value=0)
        to_share_right_vel = jnp.compress(to_share_right, vel, axis=0, size=max_values_to_share, fill_value=0)
        to_share_right_acc = jnp.compress(to_share_right, acc, axis=0, size=max_values_to_share, fill_value=0)
        to_share_right_idx = jnp.compress(to_share_right, particle_indices, axis=0, size=max_values_to_share,
                                          fill_value=-1)

        to_share_left_pmid = jnp.compress(to_share_left, pmid, axis=0, size=max_values_to_share, fill_value=0)
        to_share_left_disp = jnp.compress(to_share_left, disp, axis=0, size=max_values_to_share, fill_value=0)
        to_share_left_vel = jnp.compress(to_share_left, vel, axis=0, size=max_values_to_share, fill_value=0)
        to_share_left_acc = jnp.compress(to_share_left, acc, axis=0, size=max_values_to_share, fill_value=0)
        to_share_left_idx = jnp.compress(to_share_left, particle_indices, axis=0, size=max_values_to_share,
                                         fill_value=-1)

        incoming_from_left_pmid, incoming_from_left_disp, incoming_from_left_vel, incoming_from_left_acc, incoming_from_left_idx = jax.lax.ppermute(
            (to_share_right_pmid, to_share_right_disp, to_share_right_vel, to_share_right_acc, to_share_right_idx),
            axis_name=AXIS_NAME, perm=right_perm)
        incoming_from_right_pmid, incoming_from_right_disp, incoming_from_right_vel, incoming_from_right_acc, incoming_from_right_idx = jax.lax.ppermute(
            (to_share_left_pmid, to_share_left_disp, to_share_left_vel, to_share_left_acc, to_share_left_idx),
            axis_name=AXIS_NAME, perm=left_perm)

        """jax.debug.print("[GPU {a}] to_share_left: {z}, to_share_left_pmid: {x}, to_share_left_disp: {y}.",
                        a=jax.lax.axis_index("gpus"), x=to_share_left_pmid, y=to_share_left_disp,
                        z=jnp.sum(to_share_left))

        jax.debug.print("[GPU {a}] incoming_from_right_pmid: {x}, incoming_from_right_disp: {y}.",
                        a=jax.lax.axis_index("gpus"), x=incoming_from_right_pmid, y=incoming_from_right_disp)"""

        incoming_pmid = jnp.concatenate((incoming_from_right_pmid, incoming_from_left_pmid), axis=0)
        incoming_disp = jnp.concatenate((incoming_from_right_disp, incoming_from_left_disp), axis=0)
        incoming_vel = jnp.concatenate((incoming_from_right_vel, incoming_from_left_vel), axis=0)
        incoming_acc = jnp.concatenate((incoming_from_right_acc, incoming_from_left_acc), axis=0)
        incoming_idx = jnp.concatenate((incoming_from_right_idx, incoming_from_left_idx), axis=0)

        pmid, disp, vel, acc, particle_indices, unused_index = Particles.remove_particles(pmid, disp, vel, acc,
                                                                                          particle_indices, to_remove,
                                                                                          unused_indexes)

        pmid, disp, vel, acc, particle_indices, unused_indexes = Particles.add_particles(pmid, disp, vel, acc,
                                                                                         particle_indices,
                                                                                         unused_indexes,
                                                                                         incoming_pmid,
                                                                                         incoming_disp,
                                                                                         incoming_vel,
                                                                                         incoming_acc,
                                                                                         incoming_idx,
                                                                                         max_values_to_share * 2)

        x_mod = (pmid[:, 0] + disp[:, 0] * disp_size) % global_nMesh
        previous_halo_mask = Particles.compute_halo_mask(x_mod, halo_start_fix, halo_end, unused_indexes)

        return pmid, disp, vel, acc, particle_indices, previous_halo_mask, unused_indexes, check_fraction_and_share, jnp.maximum(
            jnp.sum(to_share_right), jnp.sum(to_share_left))

    @staticmethod
    def initialize_mGPU_halo_movement(conf):
        """return shard_map(
            Particles.move_particles_shard_map,
            mesh=conf.compute_mesh,
            in_specs=(
                P(AXIS_NAME, None),  # pmid
                P(AXIS_NAME, None),  # disp
                P(AXIS_NAME, None),  # vel
                P(AXIS_NAME, None),  # acc
                P(AXIS_NAME),  # particle_indices
                P(AXIS_NAME),  # halo_start
                P(AXIS_NAME),  # halo_end
                P(AXIS_NAME),  # previous_halo_mask
                P(AXIS_NAME),  # unused_indexes
                None,  # nMesh (not a tracer)
                None,  # max_values_to_share (not a tracer)
                None,  # left_perm (not a tracer)
                None,  # right_perm (not a tracer)
                None,  # num_gpus
            ),
            out_specs=(
                P(AXIS_NAME, None),  # pmid
                P(AXIS_NAME, None),  # disp
                P(AXIS_NAME, None),  # vel
                P(AXIS_NAME, None),  # acc
                P(AXIS_NAME),  # particle_indices
                P(AXIS_NAME),  # previous_halo_mask
                P(AXIS_NAME),  # unused_indexes
                P(),  # has_failed
                P()  # max_particles_moved
            ),
            check_rep=False
        )"""
        func = partial(Particles.move_particles_shard_map,
                       global_nMesh=conf.nMesh,
                       max_values_to_share=conf.max_share_ptcl,
                       left_perm=conf.left_perm,
                       right_perm=conf.right_perm,
                       num_gpus=conf.num_devices,
                       disp_size=conf.disp_size)
        return shard_map(
            func,
            mesh=conf.compute_mesh,
            in_specs=(
                P(AXIS_NAME, None),  # pmid
                P(AXIS_NAME, None),  # disp
                P(AXIS_NAME, None),  # vel
                P(AXIS_NAME, None),  # acc
                P(AXIS_NAME),  # particle_indices
                P(AXIS_NAME),  # halo_start
                P(AXIS_NAME),  # halo_end
                P(AXIS_NAME),  # previous_halo_mask
                P(AXIS_NAME),  # unused_indexes
                P()
            ),
            out_specs=(
                P(AXIS_NAME, None),  # pmid
                P(AXIS_NAME, None),  # disp
                P(AXIS_NAME, None),  # vel
                P(AXIS_NAME, None),  # acc
                P(AXIS_NAME),  # particle_indices
                P(AXIS_NAME),  # previous_halo_mask
                P(AXIS_NAME),  # unused_indexes
                P(),  # has_failed
                P()  # max_particles_moved
            ),
            check_rep=False
        )
