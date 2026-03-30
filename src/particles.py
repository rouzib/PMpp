from dataclasses import field
from functools import partial
from itertools import accumulate
from operator import itemgetter, mul
from typing import Optional, Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax.experimental.shard_map import shard_map
from jax.typing import ArrayLike
from jax.sharding import NamedSharding, PartitionSpec as P

from .halo_moving import (
    compute_halo_mask as halo_compute_halo_mask,
    compute_halo_mask_shard_map as halo_compute_halo_mask_shard_map,
    initialize_mGPU_compute_halo_mask as halo_initialize_mGPU_compute_halo_mask,
    initialize_mGPU_halo_move_pullback as halo_initialize_mGPU_halo_move_pullback,
    initialize_mGPU_halo_movement_canonical as halo_initialize_mGPU_halo_movement_canonical,
    initialize_mGPU_reconstruct_pre_drift as halo_initialize_mGPU_reconstruct_pre_drift,
    particles_in_slice_mask as halo_particles_in_slice_mask,
)
from .utils import pytree_dataclass, is_float0_array, raise_error, AXIS_NAME, pmid_to_idx


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
        with much more uniform precision than positions, wherever they are. Call
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

    attr: Any = None

    def __post_init__(self):

        def get_dtype_by_name(name):
            if name == "pmid":
                return conf.pmid_dtype
            elif name == "disp":
                return conf.float_dtype
            elif (name == "unused_index") | (name == "halo_mask"):
                return jnp.bool
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
        """Compatibility wrapper for the standalone halo-moving helper."""
        return halo_particles_in_slice_mask(x_mod, slice_start, slice_end)

    @staticmethod
    @jax.jit
    def compute_halo_mask(x_mod, halo_start, halo_end, unused_indexes):
        """Compatibility wrapper for the standalone halo-moving helper."""
        return halo_compute_halo_mask(x_mod, halo_start, halo_end, unused_indexes)

    @staticmethod
    def _host_particles_in_slice_mask(x_mod, slice_start, slice_end):
        if slice_start > slice_end:
            return (x_mod >= slice_start) | (x_mod < slice_end)
        return (x_mod >= slice_start) & (x_mod < slice_end)

    @staticmethod
    def _host_compute_halo_mask(x_mod, halo_start, halo_end, unused_index):
        mask_start = Particles._host_particles_in_slice_mask(x_mod, halo_start[0], halo_start[1])
        mask_end = Particles._host_particles_in_slice_mask(x_mod, halo_end[0], halo_end[1])
        return (mask_start | mask_end) & ~unused_index

    @staticmethod
    def _shard_host_slices(conf, slices: List[np.ndarray], dtype):
        slices = [jnp.asarray(s, dtype=dtype) for s in slices]
        total_shape = (conf.num_devices * slices[0].shape[0],) + slices[0].shape[1:]
        partition = P(AXIS_NAME, *([None] * (slices[0].ndim - 1)))
        sharding = NamedSharding(conf.compute_mesh, partition)
        mesh_devices = list(conf.compute_mesh.devices.flat)
        device_arrays = [
            jax.device_put(slices[i], device=mesh_devices[i])
            for i in range(conf.num_devices)
        ]
        return jax.make_array_from_single_device_arrays(total_shape, sharding, device_arrays)

    @staticmethod
    def _partition_and_shard_particle_fields(conf, pmid, disp, vel, acc):
        runtime = conf.multigpu
        if runtime is None:
            raise ValueError("Host-side particle partitioning requires an initialized multi-GPU runtime.")

        store_particle_halos = runtime.store_particle_halos
        pmid_host = np.asarray(jax.device_get(pmid), dtype=np.dtype(conf.pmid_dtype))
        disp_host = np.asarray(jax.device_get(disp), dtype=np.dtype(conf.float_dtype))
        vel_host = None if vel is None else np.asarray(jax.device_get(vel), dtype=np.dtype(conf.float_dtype))
        acc_host = None if acc is None else np.asarray(jax.device_get(acc), dtype=np.dtype(conf.float_dtype))

        slice_start = runtime.slice_start if store_particle_halos else runtime.owned_slice_start
        slice_end = runtime.slice_end if store_particle_halos else runtime.owned_slice_end
        slice_start = np.asarray(jax.device_get(slice_start), dtype=np.int64)
        slice_end = np.asarray(jax.device_get(slice_end), dtype=np.int64)
        halo_start = np.asarray(jax.device_get(runtime.halo_start), dtype=np.int64)
        halo_end = np.asarray(jax.device_get(runtime.halo_end), dtype=np.int64)

        x_mod = (pmid_host[:, 0] + disp_host[:, 0] * conf.disp_size) % conf.nMesh
        capacity = conf.max_ptcl_per_slice
        spatial_ndim = pmid_host.shape[1]

        pmid_slices, disp_slices = [], []
        vel_slices = [] if vel_host is not None else None
        acc_slices = [] if acc_host is not None else None
        unused_slices, halo_slices = [], []

        for slice_idx in range(conf.num_devices):
            in_slice_mask = Particles._host_particles_in_slice_mask(
                x_mod,
                slice_start[slice_idx],
                slice_end[slice_idx],
            )
            indices = np.flatnonzero(in_slice_mask)
            count = indices.size
            if count > capacity:
                raise ValueError(
                    "[ERROR] [GPU {gpu}] Exceeded max_ptcl_per_slice: max_ptcl_per_slice={cap}, "
                    "actual particles in slice={count}. Consider increasing 'conf.max_ptcl_per_slice'."
                    .format(gpu=slice_idx, cap=capacity, count=count)
                )

            count = min(count, capacity)
            selected = indices[:count]

            pmid_slice = np.zeros((capacity, spatial_ndim), dtype=pmid_host.dtype)
            disp_slice = np.zeros((capacity, spatial_ndim), dtype=disp_host.dtype)
            if count:
                pmid_slice[:count] = pmid_host[selected]
                disp_slice[:count] = disp_host[selected]

            unused_index = np.ones((capacity,), dtype=np.bool_)
            unused_index[:count] = False
            if store_particle_halos:
                x_mod_local = (pmid_slice[:, 0] + disp_slice[:, 0] * conf.disp_size) % conf.nMesh
                halo_mask = Particles._host_compute_halo_mask(
                    x_mod_local,
                    halo_start[slice_idx],
                    halo_end[slice_idx],
                    unused_index,
                )
            else:
                halo_mask = np.zeros((capacity,), dtype=np.bool_)

            pmid_slices.append(pmid_slice)
            disp_slices.append(disp_slice)
            unused_slices.append(unused_index)
            halo_slices.append(halo_mask)

            if vel_slices is not None:
                vel_slice = np.zeros((capacity, spatial_ndim), dtype=vel_host.dtype)
                if count:
                    vel_slice[:count] = vel_host[selected]
                vel_slices.append(vel_slice)

            if acc_slices is not None:
                acc_slice = np.zeros((capacity, spatial_ndim), dtype=acc_host.dtype)
                if count:
                    acc_slice[:count] = acc_host[selected]
                acc_slices.append(acc_slice)

        pmid = Particles._shard_host_slices(conf, pmid_slices, conf.pmid_dtype)
        disp = Particles._shard_host_slices(conf, disp_slices, conf.float_dtype)
        unused_index = Particles._shard_host_slices(conf, unused_slices, jnp.bool_)
        halo_mask = Particles._shard_host_slices(conf, halo_slices, jnp.bool_)

        vel = None if vel_slices is None else Particles._shard_host_slices(conf, vel_slices, conf.float_dtype)
        acc = None if acc_slices is None else Particles._shard_host_slices(conf, acc_slices, conf.float_dtype)
        return pmid, disp, vel, acc, unused_index, halo_mask

    @staticmethod
    def distribute_ptcl_pos(pmid, disp, vel, acc, conf, slice_idx):
        runtime = conf.multigpu
        store_particle_halos = runtime is not None and runtime.store_particle_halos
        if runtime is None:
            slice_start = conf.slice_start[slice_idx]
            slice_end = conf.slice_end[slice_idx]
        elif store_particle_halos:
            slice_start = runtime.slice_start[slice_idx]
            slice_end = runtime.slice_end[slice_idx]
        else:
            slice_start = runtime.owned_slice_start[slice_idx]
            slice_end = runtime.owned_slice_end[slice_idx]

        x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % conf.nMesh
        in_slice_mask = Particles.particles_in_slice_mask(x_mod, slice_start, slice_end)
        indices = jnp.compress(in_slice_mask, jnp.arange(pmid.shape[0]), axis=0,
                               size=min(conf.max_ptcl_per_slice, pmid.shape[0]),
                               fill_value=-1)

        _ = jax.lax.cond(
            jnp.sum(in_slice_mask) > conf.max_ptcl_per_slice,
            lambda _: raise_error(
                "[ERROR] [GPU {a}] Exceeded max_ptcl_per_slice: "
                "max_ptcl_per_slice={x}, actual max_ptcl_per_slice={y}. Some particles may have "
                f"disappeared. Consider making 'conf.max_ptcl_per_slice' bigger so that this does not happen again.",
                a=slice_idx, x=conf.max_ptcl_per_slice, y=jnp.sum(in_slice_mask)),
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
        if runtime is not None and not store_particle_halos:
            halo_mask = jnp.zeros_like(unused_index)
        else:
            x_mod = (pmid_sliced[:, 0] + disp_sliced[:, 0] * conf.disp_size) % conf.nMesh
            halo_mask = Particles.compute_halo_mask(x_mod, conf.halo_start[slice_idx],
                                                    conf.halo_end[slice_idx], unused_index)

        unused_index.astype(jnp.bool)
        halo_mask.astype(jnp.bool)

        return pmid_sliced, disp_sliced, vel_sliced, acc_sliced, unused_index, halo_mask

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
        if conf.use_mGPU:
            pos_host = np.asarray(jax.device_get(pos), dtype=np.dtype(conf.float_dtype))
            pmid = np.rint(pos_host / conf.cell_size).astype(np.dtype(conf.pmid_dtype))
            disp = (pos_host - pmid * conf.cell_size).astype(np.dtype(conf.float_dtype))
            if wrap:
                pmid = np.mod(pmid, np.asarray(conf.mesh_shape, dtype=pmid.dtype))
            pmid, disp, vel, acc, unused_index, halo_mask = cls._partition_and_shard_particle_fields(
                conf, pmid, disp, vel, acc
            )
            return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask)

        pos = jnp.asarray(pos)
        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=None, halo_mask=None)

    @classmethod
    def from_pos_sharded(cls, conf, pos, vel=None, acc=None, wrap=True):
        """Construct particle state of ``pmid`` and ``disp`` from positions."""
        return cls.from_pos(conf, pos, vel=vel, acc=acc, wrap=wrap)

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
        if conf.use_mGPU:
            pmid, disp, vel, acc, unused_index, halo_mask = cls._partition_and_shard_particle_fields(
                conf, pmid, disp, vel, acc
            )
            return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask)

        pmid = jnp.asarray(pmid)
        disp = jnp.asarray(disp)
        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=None, halo_mask=None)

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
        acc = ptcl.acc

        if conf.use_mGPU:
            pmid = np.asarray(jax.device_get(pmid), dtype=np.dtype(conf.pmid_dtype))
            disp = np.asarray(jax.device_get(disp), dtype=np.dtype(conf.float_dtype))
            if wrap:
                pmid = np.mod(pmid, np.asarray(conf.mesh_shape, dtype=pmid.dtype))
            pmid, disp, vel, acc, unused_index, halo_mask = cls._partition_and_shard_particle_fields(
                conf, pmid, disp, vel, acc
            )
            return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask)

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        if wrap:
            pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=None, halo_mask=None)

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
            runtime = conf.multigpu
            store_particle_halos = runtime is not None and runtime.store_particle_halos
            n_devices = conf.num_devices
            if store_particle_halos:
                sm = conf.local_mesh_shape[0] + conf.ptcl_halo_width
                n_ptcl = sp // n_devices + (1 if n_devices > 1 else 0)
                pmid_shift = -conf.ptcl_halo_width
            else:
                sm = conf.local_mesh_shape[0]
                n_ptcl = sp // n_devices
                pmid_shift = 0
            pmid_x = jnp.linspace(0, sm, num=n_ptcl, endpoint=False)
            offset = conf.offsets[gpu_id]
            pmid_x = jnp.rint(pmid_x)
            pmid_x = pmid_x.astype(conf.pmid_dtype)

            disp_x = jnp.arange(n_ptcl) * conf.mesh_shape[0] - pmid_x.astype(int) * sp
            disp_x *= conf.cell_size / sp
            disp_x = disp_x.astype(conf.float_dtype)

            pmid_x = pmid_x + offset + pmid_shift
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

            if runtime is not None and not store_particle_halos:
                halo_mask = jnp.zeros_like(unused_index)
            else:
                x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % conf.nMesh
                halo_mask = Particles.compute_halo_mask(x_mod, conf.halo_start[gpu_id],
                                                        conf.halo_end[gpu_id], unused_index)

            return pmid, disp, unused_index, halo_mask

        @partial(shard_map, mesh=conf.compute_mesh, in_specs=(P()),
                 out_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME)))
        def build_all():
            axis = jax.lax.axis_index(AXIS_NAME)
            return build_local(axis)

        pmid, disp, unused_index, halo_mask = build_all()

        vel = jnp.zeros_like(disp) if vel else None
        acc = jnp.zeros_like(disp) if acc else None

        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask)

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
        return pmid, disp, vel, acc, unused_index, halo_mask

    @staticmethod
    @jax.jit
    def remove_particles(pmid, disp, vel, acc, mask, unused_index):
        """
        Removes particles from the given data arrays based on a boolean mask. This function
        operates on position vectors, velocity vectors, particle indices, and unused indices.
        Positions and velocities corresponding to masked particles are set to zero, particle
        indices are set to -1, and unused indices are set to True.

        :param pos: Array of particle positions. The shape of the array should align with
            the expected broadcast dimension for operations.
        :param vel: Array of particle velocities. Should match the shape and type specifications
            of `pos`.
        :param mask: Boolean array indicating which particles should be removed. `True` values
            flag particles for removal at corresponding positions in `pmid`, `disp`, `vel`, and `acc`.
        :param unused_index: Array of unused indices, which will also be updated when particles are
            removed as per the mask.
        :return: A tuple containing the updated position array (`pos`), velocity array (`vel`),
            acceleration array (`acc`), and unused index array (`unused_index`) after particles are
            removed based on the mask.
        """
        mask_2d = jnp.expand_dims(mask, axis=-1)
        pmid = jax.lax.select(jnp.broadcast_to(mask_2d, pmid.shape), jnp.zeros_like(pmid), pmid)
        disp = jax.lax.select(jnp.broadcast_to(mask_2d, disp.shape), jnp.zeros_like(disp), disp)
        vel = jax.lax.select(jnp.broadcast_to(mask_2d, vel.shape), jnp.zeros_like(vel), vel)
        acc = jax.lax.select(jnp.broadcast_to(mask_2d, acc.shape), jnp.zeros_like(acc), acc)
        unused_index = jax.lax.select(mask, jnp.ones_like(unused_index), unused_index)
        return pmid, disp, vel, acc, unused_index

    @staticmethod
    @partial(jax.jit, static_argnames=["max_values_to_add"])
    def add_particles(
        pmid,
        disp,
        vel,
        acc,
        unused_indexes,
        new_pmid,
        new_disp,
        new_vel,
        new_acc,
        new_valid,
        max_values_to_add,
    ):
        max_values_to_add = min(max_values_to_add, pmid.shape[0])
        """
        Add a specified number of new particle positions, velocities, and indices to the existing particle
        system, while ensuring that the number of added particles does not exceed the maximum limit. The
        function updates the particle system and the unused slot indices accordingly.

        :param pos: A 2D array representing the positions of existing particles. Each row corresponds to
                    a particle, and each column corresponds to a coordinate.
        :param vel: A 2D array representing the velocities of existing particles. Each row corresponds to
                    a particle, and each column corresponds to a speed component.
        :param unused_indexes: A 1D Boolean array where True indicates the presence of available slots
                               for new particles and False indicates occupied slots.
        :param new_pos: A 2D array representing the positions of the new particles to be added. Each row
                        corresponds to a particle, and each column corresponds to a coordinate.
        :param new_vel: A 2D array representing the velocities of the new particles to be added. Each row
                        corresponds to a particle, and each column corresponds to a speed component.
        :param max_values_to_add: An integer representing the maximum number of new particles to add
                                  to the system.
        :return: A tuple containing the updated positions, velocities, indices of the particles, and the
                 updated array of unused indices.
        """
        num_values_to_add = jnp.sum(new_valid)

        _ = jax.lax.cond(
            jnp.sum(unused_indexes) < num_values_to_add,
            lambda _: raise_error(
                "[ERROR] Exceeded max_amount_particles_per_slice. Available slots: {x}, values_to_add: {y}. Consider making 'max_amount_particles_per_slice' bigger.",
                x=jnp.sum(unused_indexes), y=num_values_to_add),
            lambda _: None,
            operand=None
        )

        # Pad with a stable sentinel slot instead of 0. Using 0 here can produce repeated
        # destination indices after the real unused slots, and those padded `.at[...].set`
        # writes can overwrite an earlier successful insertion on slot 0.
        real_indices = jnp.nonzero(
            unused_indexes,
            size=max_values_to_add,
            fill_value=pmid.shape[0] - 1,
        )[0]

        valid_new_indices = jnp.nonzero(new_valid, size=max_values_to_add, fill_value=0)[0]

        chosen_new_pmid = jax.lax.dynamic_slice_in_dim(
            new_pmid[valid_new_indices],
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_disp = jax.lax.dynamic_slice_in_dim(
            new_disp[valid_new_indices],
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_vel = jax.lax.dynamic_slice_in_dim(
            new_vel[valid_new_indices],
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        chosen_new_acc = jax.lax.dynamic_slice_in_dim(
            new_acc[valid_new_indices],
            start_index=0,
            slice_size=max_values_to_add,
            axis=0
        )
        update_mask = jnp.arange(max_values_to_add) < num_values_to_add
        current_pmid = pmid[real_indices]
        current_disp = disp[real_indices]
        current_vel = vel[real_indices]
        current_acc = acc[real_indices]

        pmid = pmid.at[real_indices].set(jnp.where(update_mask[:, None], chosen_new_pmid, current_pmid))
        disp = disp.at[real_indices].set(jnp.where(update_mask[:, None], chosen_new_disp, current_disp))
        vel = vel.at[real_indices].set(jnp.where(update_mask[:, None], chosen_new_vel, current_vel))
        acc = acc.at[real_indices].set(jnp.where(update_mask[:, None], chosen_new_acc, current_acc))
        unused_indexes = jnp.all(pmid == 0, axis=1) & jnp.all(disp == 0, axis=1)

        return pmid, disp, vel, acc, unused_indexes

    @staticmethod
    def _key_fill_value(conf):
        return jnp.asarray(conf.mesh_size, dtype=jnp.int32)

    @staticmethod
    def _owned_slice_bounds(global_nMesh, num_gpus, offsets):
        owned_start = offsets[jax.lax.axis_index(AXIS_NAME)]
        owned_end = (owned_start + global_nMesh // num_gpus) % global_nMesh
        return owned_start, owned_end

    @staticmethod
    def _x_mod_from_disp(pmid, disp, global_nMesh, disp_size):
        return (pmid[:, 0] + disp[:, 0] * disp_size) % global_nMesh

    @staticmethod
    def _capacity_check(count, capacity, message):
        _ = jax.lax.cond(
            count > capacity,
            lambda _: raise_error(message, x=count, y=capacity),
            lambda _: None,
            operand=None,
        )

    @staticmethod
    def _compact_sorted_particles(keys, pmid, disp, vel, acc, mask, capacity, key_fill, error_message):
        count = jnp.sum(mask)
        Particles._capacity_check(count, capacity, error_message)
        # Canonical callers preserve packed-key order already: they compact from
        # a globally sorted authoritative sequence or from masks applied to that
        # same sorted sequence. Mask-compaction therefore preserves the order and
        # does not need an additional argsort.
        keys_compact = jnp.compress(mask, keys, axis=0, size=capacity, fill_value=key_fill)
        pmid_compact = jnp.compress(mask, pmid, axis=0, size=capacity, fill_value=0)
        disp_compact = jnp.compress(mask, disp, axis=0, size=capacity, fill_value=0)
        vel_compact = jnp.compress(mask, vel, axis=0, size=capacity, fill_value=0)
        acc_compact = jnp.compress(mask, acc, axis=0, size=capacity, fill_value=0)
        valid = jnp.arange(capacity) < count
        return keys_compact, pmid_compact, disp_compact, vel_compact, acc_compact, valid

    @staticmethod
    def _sorted_merge_two(
        keys_a,
        pmid_a,
        disp_a,
        vel_a,
        acc_a,
        valid_a,
        keys_b,
        pmid_b,
        disp_b,
        vel_b,
        acc_b,
        valid_b,
        capacity,
        key_fill,
        error_message,
    ):
        count_a = jnp.sum(valid_a)
        count_b = jnp.sum(valid_b)
        total = count_a + count_b
        Particles._capacity_check(total, capacity, error_message)
        keys_cat = jnp.concatenate((jnp.where(valid_a, keys_a, key_fill), jnp.where(valid_b, keys_b, key_fill)), axis=0)
        pmid_cat = jnp.concatenate((pmid_a, pmid_b), axis=0)
        disp_cat = jnp.concatenate((disp_a, disp_b), axis=0)
        vel_cat = jnp.concatenate((vel_a, vel_b), axis=0)
        acc_cat = jnp.concatenate((acc_a, acc_b), axis=0)
        order = jnp.argsort(keys_cat, stable=True)[:capacity]
        out_keys = keys_cat[order]
        out_pmid = pmid_cat[order]
        out_disp = disp_cat[order]
        out_vel = vel_cat[order]
        out_acc = acc_cat[order]
        out_valid = jnp.arange(capacity) < total
        out_keys = jnp.where(out_valid, out_keys, key_fill)
        return out_keys, out_pmid, out_disp, out_vel, out_acc, out_valid

    @staticmethod
    def _sorted_merge_three(
        keys_a,
        pmid_a,
        disp_a,
        vel_a,
        acc_a,
        valid_a,
        keys_b,
        pmid_b,
        disp_b,
        vel_b,
        acc_b,
        valid_b,
        keys_c,
        pmid_c,
        disp_c,
        vel_c,
        acc_c,
        valid_c,
        capacity,
        key_fill,
        error_message,
    ):
        merged_ab = Particles._sorted_merge_two(
            keys_a, pmid_a, disp_a, vel_a, acc_a, valid_a,
            keys_b, pmid_b, disp_b, vel_b, acc_b, valid_b,
            capacity, key_fill, error_message,
        )
        return Particles._sorted_merge_two(
            *merged_ab,
            keys_c, pmid_c, disp_c, vel_c, acc_c, valid_c,
            capacity, key_fill, error_message,
        )

    @staticmethod
    def _pack_left_halo_and_authoritative(
        left_keys,
        left_pmid,
        left_disp,
        left_vel,
        left_acc,
        left_valid,
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        max_ptcl_per_slice,
        halo_start,
        halo_end,
        global_nMesh,
        disp_size,
    ):
        left_count = jnp.sum(left_valid)
        auth_count = jnp.sum(auth_valid)
        total = left_count + auth_count
        Particles._capacity_check(
            total,
            max_ptcl_per_slice,
            "[ERROR] Exceeded canonical particle storage capacity. "
            "required_slots={x}, max_ptcl_per_slice={y}.",
        )

        pmid = jnp.zeros((max_ptcl_per_slice, left_pmid.shape[1]), dtype=left_pmid.dtype)
        disp = jnp.zeros((max_ptcl_per_slice, left_disp.shape[1]), dtype=left_disp.dtype)
        vel = jnp.zeros((max_ptcl_per_slice, left_vel.shape[1]), dtype=left_vel.dtype)
        acc = jnp.zeros((max_ptcl_per_slice, left_acc.shape[1]), dtype=left_acc.dtype)
        slots = jnp.arange(max_ptcl_per_slice, dtype=jnp.int32)
        left_mask = slots < left_count
        auth_mask = (slots >= left_count) & (slots < total)
        left_idx = jnp.minimum(slots, left_pmid.shape[0] - 1)
        auth_idx = jnp.maximum(slots - left_count.astype(jnp.int32), 0)
        auth_idx = jnp.minimum(auth_idx, auth_pmid.shape[0] - 1)

        pmid = jnp.where(left_mask[:, None], left_pmid[left_idx], pmid)
        disp = jnp.where(left_mask[:, None], left_disp[left_idx], disp)
        vel = jnp.where(left_mask[:, None], left_vel[left_idx], vel)
        acc = jnp.where(left_mask[:, None], left_acc[left_idx], acc)

        pmid = jnp.where(auth_mask[:, None], auth_pmid[auth_idx], pmid)
        disp = jnp.where(auth_mask[:, None], auth_disp[auth_idx], disp)
        vel = jnp.where(auth_mask[:, None], auth_vel[auth_idx], vel)
        acc = jnp.where(auth_mask[:, None], auth_acc[auth_idx], acc)

        unused_index = jnp.arange(max_ptcl_per_slice) >= total
        x_mod = Particles._x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
        halo_mask = Particles.compute_halo_mask(x_mod, halo_start.squeeze(), halo_end.squeeze(), unused_index)
        return pmid, disp, vel, acc, halo_mask, unused_index

    @staticmethod
    def _canonical_authoritative_from_full(
        pmid,
        source_disp,
        carried_disp,
        vel,
        acc,
        unused_index,
        global_nMesh,
        disp_size,
        num_gpus,
        offsets,
        conf,
    ):
        owned_start, owned_end = Particles._owned_slice_bounds(global_nMesh, num_gpus, offsets)
        x_mod = Particles._x_mod_from_disp(pmid, source_disp, global_nMesh, disp_size)
        owned_mask = Particles.particles_in_slice_mask(x_mod, owned_start, owned_end) & ~unused_index
        keys = pmid_to_idx(pmid, conf)
        return Particles._compact_sorted_particles(
            keys,
            pmid,
            carried_disp,
            vel,
            acc,
            owned_mask,
            pmid.shape[0],
            Particles._key_fill_value(conf),
            "[ERROR] Exceeded authoritative compact capacity. "
            "authoritative_particles={x}, compact_capacity={y}.",
        )

    @staticmethod
    def _canonical_authoritative_from_full_with_slots(
        pmid,
        source_disp,
        carried_disp,
        vel,
        acc,
        unused_index,
        global_nMesh,
        disp_size,
        num_gpus,
        offsets,
        conf,
    ):
        owned_start, owned_end = Particles._owned_slice_bounds(global_nMesh, num_gpus, offsets)
        x_mod = Particles._x_mod_from_disp(pmid, source_disp, global_nMesh, disp_size)
        owned_mask = Particles.particles_in_slice_mask(x_mod, owned_start, owned_end) & ~unused_index
        keys = pmid_to_idx(pmid, conf)
        auth = Particles._compact_sorted_particles(
            keys,
            pmid,
            carried_disp,
            vel,
            acc,
            owned_mask,
            pmid.shape[0],
            Particles._key_fill_value(conf),
            "[ERROR] Exceeded authoritative compact capacity. "
            "authoritative_particles={x}, compact_capacity={y}.",
        )
        slot_index = jnp.arange(pmid.shape[0], dtype=jnp.int32)
        auth_slots = jnp.compress(
            owned_mask,
            slot_index,
            axis=0,
            size=pmid.shape[0],
            fill_value=jnp.asarray(-1, slot_index.dtype),
        )
        return (*auth, auth_slots)

    @staticmethod
    def _scatter_compact_to_dense(compact_values, compact_slots, compact_valid, out_size):
        out = jnp.zeros((out_size,) + compact_values.shape[1:], dtype=compact_values.dtype)
        slots = jnp.where(compact_valid, compact_slots, 0)
        mask = compact_valid.reshape((compact_valid.shape[0],) + (1,) * (compact_values.ndim - 1))
        values = compact_values * mask.astype(compact_values.dtype)
        return out.at[slots].add(values)

    @staticmethod
    def _sorted_merge_three_with_provenance(
        keys_a,
        pmid_a,
        disp_a,
        vel_a,
        acc_a,
        valid_a,
        keys_b,
        pmid_b,
        disp_b,
        vel_b,
        acc_b,
        valid_b,
        keys_c,
        pmid_c,
        disp_c,
        vel_c,
        acc_c,
        valid_c,
        capacity,
        key_fill,
        error_message,
    ):
        count_a = jnp.sum(valid_a)
        count_b = jnp.sum(valid_b)
        count_c = jnp.sum(valid_c)
        total = count_a + count_b + count_c
        Particles._capacity_check(total, capacity, error_message)

        keys_cat = jnp.concatenate((
            jnp.where(valid_a, keys_a, key_fill),
            jnp.where(valid_b, keys_b, key_fill),
            jnp.where(valid_c, keys_c, key_fill),
        ), axis=0)
        pmid_cat = jnp.concatenate((pmid_a, pmid_b, pmid_c), axis=0)
        disp_cat = jnp.concatenate((disp_a, disp_b, disp_c), axis=0)
        vel_cat = jnp.concatenate((vel_a, vel_b, vel_c), axis=0)
        acc_cat = jnp.concatenate((acc_a, acc_b, acc_c), axis=0)

        src_a = jnp.arange(keys_a.shape[0], dtype=jnp.int32)
        src_b = jnp.arange(keys_b.shape[0], dtype=jnp.int32)
        src_c = jnp.arange(keys_c.shape[0], dtype=jnp.int32)
        src_idx = jnp.concatenate((src_a, src_b, src_c), axis=0)
        src_tag = jnp.concatenate((
            jnp.where(valid_a, jnp.int32(0), jnp.int32(3)),
            jnp.where(valid_b, jnp.int32(1), jnp.int32(3)),
            jnp.where(valid_c, jnp.int32(2), jnp.int32(3)),
        ), axis=0)

        order = jnp.argsort(keys_cat, stable=True)[:capacity]
        out_valid = jnp.arange(capacity) < total
        out_keys = keys_cat[order]
        out_pmid = pmid_cat[order]
        out_disp = disp_cat[order]
        out_vel = vel_cat[order]
        out_acc = acc_cat[order]
        out_src_idx = jnp.where(out_valid, src_idx[order], -1)
        out_src_tag = jnp.where(out_valid, src_tag[order], 3)
        out_keys = jnp.where(out_valid, out_keys, key_fill)
        return (
            out_keys,
            out_pmid,
            out_disp,
            out_vel,
            out_acc,
            out_valid,
            out_src_tag,
            out_src_idx,
        )

    @staticmethod
    def _canonical_route_authoritative(
        keys,
        pmid,
        disp,
        vel,
        acc,
        valid,
        global_nMesh,
        max_values_to_share,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    ):
        # Start from the authoritative owned-particle block only. Those particles
        # are already sorted by packed key, so each routing branch below stays
        # sorted after mask-compaction.
        owned_start, owned_end = Particles._owned_slice_bounds(global_nMesh, num_gpus, offsets)
        slice_width = global_nMesh // num_gpus
        left_start = (owned_start - slice_width) % global_nMesh
        right_end = (owned_end + slice_width) % global_nMesh

        x_mod = Particles._x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
        stay_mask = valid & Particles.particles_in_slice_mask(x_mod, owned_start, owned_end)
        send_left_mask = valid & Particles.particles_in_slice_mask(x_mod, left_start, owned_start)
        send_right_mask = valid & Particles.particles_in_slice_mask(x_mod, owned_end, right_end)
        if num_gpus == 2:
            send_right_mask = jnp.zeros_like(send_right_mask)
        dropped_mask = valid & ~(stay_mask | send_left_mask | send_right_mask)

        _ = jax.lax.cond(
            jnp.any(dropped_mask),
            lambda _: raise_error(
                "[ERROR] Canonical halo move only supports same-slab or neighboring-slab migration. "
                "particles_outside_neighbor_range={x}.",
                x=jnp.sum(dropped_mask),
            ),
            lambda _: None,
            operand=None,
        )

        key_fill = Particles._key_fill_value(conf)
        # Split the authoritative sequence into particles that stay local and
        # particles that migrate one slab to the left or right. Each compacted
        # output keeps the original packed-key order.
        stay = Particles._compact_sorted_particles(
            keys, pmid, disp, vel, acc, stay_mask, pmid.shape[0], key_fill,
            "[ERROR] Exceeded stay-particle compact capacity. stay_particles={x}, capacity={y}.",
        )
        send_left = Particles._compact_sorted_particles(
            keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share, key_fill,
            "[ERROR] Exceeded left-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
        )
        send_right = Particles._compact_sorted_particles(
            keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share, key_fill,
            "[ERROR] Exceeded right-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
        )

        # Neighbor exports arrive already sorted, so only the stay/local-authority
        # merge needs to restore the canonical packed-key order.
        incoming_from_left = jax.lax.ppermute(send_right, axis_name=AXIS_NAME, perm=right_perm)
        incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)

        merged = Particles._sorted_merge_three(
            *stay,
            *incoming_from_left,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        max_particles_moved = jnp.maximum(jnp.sum(send_left[-1]), jnp.sum(send_right[-1]))
        return merged, max_particles_moved

    @staticmethod
    def _canonical_route_authoritative_with_aux(
        keys,
        pmid,
        disp,
        vel,
        acc,
        valid,
        global_nMesh,
        max_values_to_share,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    ):
        owned_start, owned_end = Particles._owned_slice_bounds(global_nMesh, num_gpus, offsets)
        slice_width = global_nMesh // num_gpus
        left_start = (owned_start - slice_width) % global_nMesh
        right_end = (owned_end + slice_width) % global_nMesh

        x_mod = Particles._x_mod_from_disp(pmid, disp, global_nMesh, disp_size)
        stay_mask = valid & Particles.particles_in_slice_mask(x_mod, owned_start, owned_end)
        send_left_mask = valid & Particles.particles_in_slice_mask(x_mod, left_start, owned_start)
        send_right_mask = valid & Particles.particles_in_slice_mask(x_mod, owned_end, right_end)
        if num_gpus == 2:
            send_right_mask = jnp.zeros_like(send_right_mask)

        auth_pos = jnp.arange(pmid.shape[0], dtype=jnp.int32)
        key_fill = Particles._key_fill_value(conf)

        stay = Particles._compact_sorted_particles(
            keys, pmid, disp, vel, acc, stay_mask, pmid.shape[0], key_fill,
            "[ERROR] Exceeded stay-particle compact capacity. stay_particles={x}, capacity={y}.",
        )
        send_left = Particles._compact_sorted_particles(
            keys, pmid, disp, vel, acc, send_left_mask, max_values_to_share, key_fill,
            "[ERROR] Exceeded left-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
        )
        send_right = Particles._compact_sorted_particles(
            keys, pmid, disp, vel, acc, send_right_mask, max_values_to_share, key_fill,
            "[ERROR] Exceeded right-migration share capacity. particles_to_share={x}, max_share_ptcl={y}.",
        )

        stay_pos = jnp.compress(
            stay_mask, auth_pos, axis=0, size=pmid.shape[0], fill_value=jnp.asarray(-1, auth_pos.dtype)
        )
        send_left_pos = jnp.compress(
            send_left_mask, auth_pos, axis=0, size=max_values_to_share, fill_value=jnp.asarray(-1, auth_pos.dtype)
        )
        send_right_pos = jnp.compress(
            send_right_mask, auth_pos, axis=0, size=max_values_to_share, fill_value=jnp.asarray(-1, auth_pos.dtype)
        )

        incoming_from_left = jax.lax.ppermute(send_right, axis_name=AXIS_NAME, perm=right_perm)
        incoming_from_right = jax.lax.ppermute(send_left, axis_name=AXIS_NAME, perm=left_perm)

        merged = Particles._sorted_merge_three_with_provenance(
            *stay,
            *incoming_from_left,
            *incoming_from_right,
            pmid.shape[0],
            key_fill,
            "[ERROR] Exceeded canonical authoritative capacity after migration. "
            "required_particles={x}, max_ptcl_per_slice={y}.",
        )
        route_aux = (
            stay_pos,
            stay[-1],
            send_left_pos,
            send_left[-1],
            send_right_pos,
            send_right[-1],
            merged[-2],
            merged[-1],
        )
        return merged[:6], route_aux

    @staticmethod
    def _reverse_build_full_cot(
        full_cot,
        auth_pmid,
        auth_disp,
        auth_valid,
        halo_end,
        max_ptcl_per_slice,
        max_halo_values_to_share,
        global_nMesh,
        left_perm,
        right_perm,
        disp_size,
    ):
        auth_pos = jnp.arange(auth_pmid.shape[0], dtype=jnp.int32)
        x_mod = Particles._x_mod_from_disp(auth_pmid, auth_disp, global_nMesh, disp_size)
        right_halo_mask = auth_valid & Particles.particles_in_slice_mask(
            x_mod, halo_end.squeeze()[0], halo_end.squeeze()[1]
        )
        right_halo_pos = jnp.compress(
            right_halo_mask,
            auth_pos,
            axis=0,
            size=max_halo_values_to_share,
            fill_value=jnp.asarray(-1, auth_pos.dtype),
        )
        right_halo_valid = jnp.arange(max_halo_values_to_share) < jnp.sum(right_halo_mask)
        left_halo_valid = jax.lax.ppermute(right_halo_valid, axis_name=AXIS_NAME, perm=right_perm)

        left_count = jnp.sum(left_halo_valid)
        auth_count = jnp.sum(auth_valid)
        slots = jnp.arange(max_ptcl_per_slice, dtype=jnp.int32)
        left_mask = slots < left_count
        auth_mask = (slots >= left_count) & (slots < (left_count + auth_count))

        left_cot = jnp.compress(
            left_mask,
            full_cot,
            axis=0,
            size=max_halo_values_to_share,
            fill_value=jnp.asarray(0, full_cot.dtype),
        )
        auth_cot = jnp.compress(
            auth_mask,
            full_cot,
            axis=0,
            size=auth_pmid.shape[0],
            fill_value=jnp.asarray(0, full_cot.dtype),
        )

        outbound_right_cot = jax.lax.ppermute(left_cot, axis_name=AXIS_NAME, perm=left_perm)
        valid_mask = right_halo_valid.reshape((right_halo_valid.shape[0],) + (1,) * (full_cot.ndim - 1))
        auth_cot = auth_cot.at[jnp.where(right_halo_valid, right_halo_pos, 0)].add(
            outbound_right_cot * valid_mask.astype(full_cot.dtype)
        )
        return auth_cot

    @staticmethod
    def _reverse_route_cot(
        merged_cot,
        stay_pos,
        stay_valid,
        send_left_pos,
        send_left_valid,
        send_right_pos,
        send_right_valid,
        merge_src_tag,
        merge_src_idx,
        auth_size,
        max_values_to_share,
        left_perm,
        right_perm,
    ):
        dtype = merged_cot.dtype
        cot_shape = merged_cot.shape[1:]
        stay_cot = jnp.zeros((stay_pos.shape[0],) + cot_shape, dtype=dtype)
        incoming_from_left_cot = jnp.zeros((max_values_to_share,) + cot_shape, dtype=dtype)
        incoming_from_right_cot = jnp.zeros((max_values_to_share,) + cot_shape, dtype=dtype)

        stay_mask = (merge_src_tag == 0)
        incoming_left_mask = (merge_src_tag == 1)
        incoming_right_mask = (merge_src_tag == 2)
        broadcast_shape = (merged_cot.shape[0],) + (1,) * (merged_cot.ndim - 1)
        stay_scale = stay_mask.reshape(broadcast_shape).astype(dtype)
        incoming_left_scale = incoming_left_mask.reshape(broadcast_shape).astype(dtype)
        incoming_right_scale = incoming_right_mask.reshape(broadcast_shape).astype(dtype)

        stay_cot = stay_cot.at[jnp.where(stay_mask, merge_src_idx, 0)].add(
            merged_cot * stay_scale
        )
        incoming_from_left_cot = incoming_from_left_cot.at[jnp.where(incoming_left_mask, merge_src_idx, 0)].add(
            merged_cot * incoming_left_scale
        )
        incoming_from_right_cot = incoming_from_right_cot.at[jnp.where(incoming_right_mask, merge_src_idx, 0)].add(
            merged_cot * incoming_right_scale
        )

        send_right_cot = jax.lax.ppermute(incoming_from_left_cot, axis_name=AXIS_NAME, perm=left_perm)
        send_left_cot = jax.lax.ppermute(incoming_from_right_cot, axis_name=AXIS_NAME, perm=right_perm)

        auth_cot = jnp.zeros((auth_size,) + cot_shape, dtype=dtype)
        stay_valid_scale = stay_valid.reshape((stay_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)).astype(dtype)
        send_left_valid_scale = send_left_valid.reshape((send_left_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)).astype(dtype)
        send_right_valid_scale = send_right_valid.reshape((send_right_valid.shape[0],) + (1,) * (merged_cot.ndim - 1)).astype(dtype)
        auth_cot = auth_cot.at[jnp.where(stay_valid, stay_pos, 0)].add(
            stay_cot * stay_valid_scale
        )
        auth_cot = auth_cot.at[jnp.where(send_left_valid, send_left_pos, 0)].add(
            send_left_cot * send_left_valid_scale
        )
        auth_cot = auth_cot.at[jnp.where(send_right_valid, send_right_pos, 0)].add(
            send_right_cot * send_right_valid_scale
        )
        return auth_cot

    @staticmethod
    def halo_move_pullback_from_prestate_shard_map(
        pmid,
        source_disp,
        carried_disp,
        vel,
        acc,
        halo_end,
        unused_indexes,
        disp_cot,
        vel_cot,
        acc_cot,
        global_nMesh,
        max_values_to_share,
        max_halo_values_to_share,
        max_ptcl_per_slice,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    ):
        # Reverse of the canonical halo move for the float payloads only.
        # The slot layout is deterministic, so we can:
        # 1. rebuild the authoritative pre-move sequence,
        # 2. replay the routing metadata in packed-key order,
        # 3. transpose the build+route operations with direct scatters/permutations.
        (
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            auth_valid,
            auth_slots,
        ) = Particles._canonical_authoritative_from_full_with_slots(
            pmid,
            source_disp,
            carried_disp,
            vel,
            acc,
            unused_indexes,
            global_nMesh,
            disp_size,
            num_gpus,
            offsets,
            conf,
        )
        (
            merged_keys,
            merged_pmid,
            merged_disp,
            merged_vel,
            merged_acc,
            merged_valid,
        ), route_aux = Particles._canonical_route_authoritative_with_aux(
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            auth_valid,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
        (
            stay_pos,
            stay_valid,
            send_left_pos,
            send_left_valid,
            send_right_pos,
            send_right_valid,
            merge_src_tag,
            merge_src_idx,
        ) = route_aux
        # Reverse the canonical build+route once for the whole float payload
        # stack instead of replaying the same metadata three times for disp,
        # vel, and acc independently.
        payload_cot = jnp.stack((disp_cot, vel_cot, acc_cot), axis=-1)
        merged_payload_cot = Particles._reverse_build_full_cot(
            payload_cot,
            merged_pmid,
            merged_disp,
            merged_valid,
            halo_end,
            max_ptcl_per_slice,
            max_halo_values_to_share,
            global_nMesh,
            left_perm,
            right_perm,
            disp_size,
        )

        auth_payload_cot = Particles._reverse_route_cot(
            merged_payload_cot,
            stay_pos,
            stay_valid,
            send_left_pos,
            send_left_valid,
            send_right_pos,
            send_right_valid,
            merge_src_tag,
            merge_src_idx,
            auth_pmid.shape[0],
            max_values_to_share,
            left_perm,
            right_perm,
        )

        input_payload_cot = Particles._scatter_compact_to_dense(
            auth_payload_cot,
            auth_slots,
            auth_valid,
            pmid.shape[0],
        )
        return input_payload_cot[..., 0], input_payload_cot[..., 1], input_payload_cot[..., 2]

    @staticmethod
    def _canonical_build_full_from_authoritative(
        auth_keys,
        auth_pmid,
        auth_disp,
        auth_vel,
        auth_acc,
        auth_valid,
        halo_start,
        halo_end,
        max_ptcl_per_slice,
        max_halo_values_to_share,
        global_nMesh,
        right_perm,
        disp_size,
        conf,
    ):
        # Rebuild the full duplicated storage from the canonical authoritative
        # block: export the local right boundary, receive the neighbor's right
        # boundary as our left halo, then pack left halo + authoritative slots.
        x_mod = Particles._x_mod_from_disp(auth_pmid, auth_disp, global_nMesh, disp_size)
        right_halo_mask = auth_valid & Particles.particles_in_slice_mask(
            x_mod, halo_end.squeeze()[0], halo_end.squeeze()[1]
        )
        key_fill = Particles._key_fill_value(conf)
        outbound_right_halo = Particles._compact_sorted_particles(
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            right_halo_mask,
            max_halo_values_to_share,
            key_fill,
            "[ERROR] Exceeded halo-share capacity while rebuilding canonical storage. "
            "particles_to_share={x}, max_halo_share_ptcl={y}.",
        )
        incoming_left_halo = jax.lax.ppermute(outbound_right_halo, axis_name=AXIS_NAME, perm=right_perm)
        return Particles._pack_left_halo_and_authoritative(
            *incoming_left_halo,
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            auth_valid,
            max_ptcl_per_slice,
            halo_start,
            halo_end,
            global_nMesh,
            disp_size,
        )

    @staticmethod
    def move_particles_canonical_shard_map(
        pmid,
        disp_before,
        disp_after,
        vel,
        acc,
        halo_start,
        halo_end,
        unused_indexes,
        global_nMesh,
        max_values_to_share,
        max_halo_values_to_share,
        max_ptcl_per_slice,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    ):
        # Forward drift halo move:
        # 1. Keep only authoritative owned particles from the previous full state.
        # 2. Re-route those particles according to the post-drift positions.
        # 3. Rebuild the duplicated left-halo + authoritative storage layout.
        auth = Particles._canonical_authoritative_from_full(
            pmid,
            disp_before,
            disp_after,
            vel,
            acc,
            unused_indexes,
            global_nMesh,
            disp_size,
            num_gpus,
            offsets,
            conf,
        )
        (auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid), max_particles_moved = (
            Particles._canonical_route_authoritative(
                *auth,
                global_nMesh,
                max_values_to_share,
                left_perm,
                right_perm,
                num_gpus,
                disp_size,
                offsets,
                conf,
            )
        )
        pmid, disp, vel, acc, halo_mask, unused_indexes = Particles._canonical_build_full_from_authoritative(
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            auth_valid,
            halo_start,
            halo_end,
            max_ptcl_per_slice,
            max_halo_values_to_share,
            global_nMesh,
            right_perm,
            disp_size,
            conf,
        )
        return pmid, disp, vel, acc, halo_mask, unused_indexes, jnp.bool_(False), max_particles_moved

    @staticmethod
    def reconstruct_pre_drift_canonical_shard_map(
        pmid,
        disp,
        vel,
        acc,
        halo_start,
        halo_end,
        unused_indexes,
        drift_factor,
        global_nMesh,
        max_values_to_share,
        max_halo_values_to_share,
        max_ptcl_per_slice,
        left_perm,
        right_perm,
        num_gpus,
        disp_size,
        offsets,
        conf,
    ):
        # Reverse drift reconstruction uses the same canonical routing contract as
        # the forward move, but starts from the post-drift authoritative block and
        # shifts displacements back before rerouting/repacking.
        auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid = Particles._canonical_authoritative_from_full(
            pmid,
            disp,
            disp,
            vel,
            acc,
            unused_indexes,
            global_nMesh,
            disp_size,
            num_gpus,
            offsets,
            conf,
        )
        auth_disp = auth_disp - auth_vel * drift_factor.astype(auth_disp.dtype)
        (auth_keys, auth_pmid, auth_disp, auth_vel, auth_acc, auth_valid), _ = Particles._canonical_route_authoritative(
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            auth_valid,
            global_nMesh,
            max_values_to_share,
            left_perm,
            right_perm,
            num_gpus,
            disp_size,
            offsets,
            conf,
        )
        pmid, disp, vel, acc, halo_mask, unused_index = Particles._canonical_build_full_from_authoritative(
            auth_keys,
            auth_pmid,
            auth_disp,
            auth_vel,
            auth_acc,
            auth_valid,
            halo_start,
            halo_end,
            max_ptcl_per_slice,
            max_halo_values_to_share,
            global_nMesh,
            right_perm,
            disp_size,
            conf,
        )
        return pmid, disp, vel, acc, unused_index, halo_mask

    @staticmethod
    @partial(jax.jit, static_argnames=["global_nMesh", "disp_size"])
    def compute_halo_mask_shard_map(pmid, disp, unused_indexes, halo_start, halo_end, global_nMesh, disp_size):
        return halo_compute_halo_mask_shard_map(
            pmid, disp, unused_indexes, halo_start, halo_end, global_nMesh, disp_size
        )

    @staticmethod
    def initialize_mGPU_halo_movement_canonical(conf):
        return halo_initialize_mGPU_halo_movement_canonical(conf)

    @staticmethod
    def initialize_mGPU_reconstruct_pre_drift(conf):
        return halo_initialize_mGPU_reconstruct_pre_drift(conf)

    @staticmethod
    def initialize_mGPU_halo_move_pullback(conf):
        return halo_initialize_mGPU_halo_move_pullback(conf)

    @staticmethod
    def initialize_mGPU_compute_halo_mask(conf):
        return halo_initialize_mGPU_compute_halo_mask(conf)
