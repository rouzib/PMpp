from dataclasses import field
from functools import partial
from itertools import accumulate
from operator import itemgetter, mul
from typing import Optional, Any, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from jax import shard_map
from jax.typing import ArrayLike
from jax.sharding import NamedSharding, PartitionSpec as P

from ..distributed.routing import (
    compute_halo_mask as halo_compute_halo_mask, particles_in_slice_mask as halo_particles_in_slice_mask,
)
from ..core.utils import pytree_dataclass, is_float0_array, raise_error, AXIS_NAME


def _mark_varying(value):
    """Mark a shard-map value as varying across the PM++ device axis."""
    if hasattr(jax.lax, "pcast"):
        try:
            return jax.lax.pcast(value, AXIS_NAME, to="varying")
        except NameError:
            # ``distribute_ptcl_pos`` is also used directly by host-side test
            # and conversion helpers, where no shard-map axis is bound.
            return value
    return jax.lax.pvary(value, (AXIS_NAME, ))


@partial(pytree_dataclass, aux_fields=("conf", ), frozen=True, eq=False)
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
    unused_index : ArrayLike, optional
        Boolean padding mask used by the static-capacity multi-GPU layouts.
        True marks inactive slots.
    halo_mask : ArrayLike, optional
        Boolean mask marking duplicated halo-particle slots in
        ``particle_halo`` mode.
    attr : pytree, optional
        Particle attributes (custom features).

    Notes
    -----
    In multi-GPU runs the leading slot dimension is a fixed-capacity storage
    layout, not necessarily the exact number of physical particles per device.
    ``unused_index`` and ``halo_mask`` distinguish active authoritative slots,
    padding, and duplicated halo slots.
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
            """Return the configured dtype for a particle-field name.

            Parameters
            ----------
            name
                Particle field name whose dtype is being resolved.
            """
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
                value = (value if value is None or is_float0_array(value) else jnp.asarray(value, dtype=dtype))
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.pmid)

    def __getitem__(self, key):
        return tree_map(itemgetter(key), self)

    @staticmethod
    @jax.jit
    def particles_in_slice_mask(x_mod, slice_start, slice_end):
        """Compatibility wrapper for the standalone halo-moving helper.

        Parameters
        ----------
        x_mod
            Particle x-coordinate wrapped into the periodic global mesh.
        slice_start
            Inclusive start of the x-slice in periodic mesh coordinates.
        slice_end
            Exclusive end of the x-slice in periodic mesh coordinates."""
        return halo_particles_in_slice_mask(x_mod, slice_start, slice_end)

    @staticmethod
    @jax.jit
    def compute_halo_mask(x_mod, halo_start, halo_end, unused_indexes):
        """Compatibility wrapper for the standalone halo-moving helper.

        Parameters
        ----------
        x_mod
            Particle x-coordinate wrapped into the periodic global mesh.
        halo_start
            Inclusive start of the halo interval in periodic mesh coordinates.
        halo_end
            Exclusive end of the halo interval in periodic mesh coordinates.
        unused_indexes
            Boolean mask marking padded or inactive particle slots."""
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
        total_shape = (conf.num_devices * slices[0].shape[0], ) + slices[0].shape[1:]
        partition = P(AXIS_NAME, *([None] * (slices[0].ndim - 1)))
        sharding = NamedSharding(conf.compute_mesh, partition)
        mesh_devices = list(conf.compute_mesh.devices.flat)
        device_arrays = [jax.device_put(slices[i], device=mesh_devices[i]) for i in range(conf.num_devices)]
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
        mesh_shape_host = np.asarray(jax.device_get(conf.mesh_shape), dtype=np.int64)

        # Use float32 arithmetic to match the GPU-side x_mod computation in
        # _x_mod_from_disp (JAX 32-bit mode). Using float64 here can place
        # boundary particles on the wrong slab, causing them to be dropped
        # during the first authoritative extraction in _canonical_authoritative_from_full.
        # `pmid` is already stored in mesh-grid coordinates because it is
        # derived from `pos / conf.cell_size`.
        x_mod = (pmid_host[:, 0].astype(np.float32) +
                 disp_host[:, 0].astype(np.float32) * np.float32(conf.disp_size)) % np.float32(conf.nMesh)
        capacity = conf.max_ptcl_per_slice
        spatial_ndim = pmid_host.shape[1]

        pmid_slices, disp_slices = [], []
        vel_slices = [] if vel_host is not None else None
        acc_slices = [] if acc_host is not None else None
        unused_slices, halo_slices = [], []

        for slice_idx in range(conf.num_devices):
            in_slice_mask = Particles._host_particles_in_slice_mask(
                x_mod, slice_start[slice_idx], slice_end[slice_idx],
            )
            indices = np.flatnonzero(in_slice_mask)
            count = indices.size
            if count > capacity:
                raise ValueError(
                    "[ERROR] [GPU {gpu}] Exceeded max_ptcl_per_slice: max_ptcl_per_slice={cap}, "
                    "actual particles in slice={count}. Consider increasing 'conf.max_ptcl_per_slice'.".format(
                        gpu=slice_idx, cap=capacity, count=count
                    )
                )

            count = min(count, capacity)
            selected = indices[:count]
            if count:
                pmid_selected = pmid_host[selected].astype(np.int64, copy=False)
                keys_selected = ((pmid_selected[:, 0] % mesh_shape_host[0]) * mesh_shape_host[1] +
                                 (pmid_selected[:, 1] % mesh_shape_host[1])
                                 ) * mesh_shape_host[2] + (pmid_selected[:, 2] % mesh_shape_host[2])
                selected = selected[np.argsort(keys_selected, kind="stable")]

            pmid_slice = np.zeros((capacity, spatial_ndim), dtype=pmid_host.dtype)
            disp_slice = np.zeros((capacity, spatial_ndim), dtype=disp_host.dtype)
            if count:
                pmid_slice[:count] = pmid_host[selected]
                disp_slice[:count] = disp_host[selected]

            unused_index = np.ones((capacity, ), dtype=np.bool_)
            unused_index[:count] = False
            if store_particle_halos:
                x_mod_local = (pmid_slice[:, 0] + disp_slice[:, 0] * conf.disp_size) % conf.nMesh
                halo_mask = Particles._host_compute_halo_mask(
                    x_mod_local, halo_start[slice_idx], halo_end[slice_idx], unused_index,
                )
            else:
                halo_mask = np.zeros((capacity, ), dtype=np.bool_)

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
        """Build per-device padded particle arrays from host particle data.

        Parameters
        ----------
        pmid
            Integer particle mesh cell identifiers.
        disp
            Particle displacement vectors relative to ``pmid`` cells.
        vel
            Particle velocity vectors.
        acc
            Particle acceleration vectors.
        conf
            Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
        slice_idx
            Index of the target device slice.
        """
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
        indices = jnp.compress(
            in_slice_mask, jnp.arange(pmid.shape[0]), axis=0, size=min(conf.max_ptcl_per_slice, pmid.shape[0]),
            fill_value=-1
        )

        _ = jax.lax.cond(
            jnp.sum(in_slice_mask) > conf.max_ptcl_per_slice, lambda _: raise_error(
                "[ERROR] [GPU {a}] Exceeded max_ptcl_per_slice: "
                "max_ptcl_per_slice={x}, actual max_ptcl_per_slice={y}. Some particles may have "
                f"disappeared. Consider making 'conf.max_ptcl_per_slice' bigger so that this does not happen again.", a=
                slice_idx, x=conf.max_ptcl_per_slice, y=jnp.sum(in_slice_mask)
            ), lambda _: None, operand=None
        )

        if vel is None:
            vel = jnp.zeros_like(disp)
        if acc is None:
            acc = jnp.zeros_like(disp)

        valid_count = jnp.minimum(jnp.sum(in_slice_mask), conf.max_ptcl_per_slice)

        def slice_particles(indices):
            """Return a particle pytree sliced by the supplied indices.

            Parameters
            ----------
            indices
                Particle indices selected from each particle field.
            """
            pmid_sliced = jax.lax.gather(
                pmid, indices[:, None], dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, ), collapsed_slice_dims=(0, ), start_index_map=(0, )
                ), slice_sizes=(1, pmid.shape[1])
            )  # Output shape: (indices.shape[0], pmid.shape[1])

            disp_sliced = jax.lax.gather(
                disp, indices[:, None], dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, ), collapsed_slice_dims=(0, ), start_index_map=(0, )
                ), slice_sizes=(1, disp.shape[1])
            )  # Output shape: (indices.shape[0], disp.shape[1])

            vel_sliced = jax.lax.gather(
                vel, indices[:, None], dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, ), collapsed_slice_dims=(0, ), start_index_map=(0, )
                ), slice_sizes=(1, vel.shape[1])
            )  # Output shape: (indices.shape[0], vel.shape[1])

            acc_sliced = jax.lax.gather(
                acc, indices[:, None], dimension_numbers=jax.lax.GatherDimensionNumbers(
                    offset_dims=(1, ), collapsed_slice_dims=(0, ), start_index_map=(0, )
                ), slice_sizes=(1, acc.shape[1])
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
            """Zero slices.

            Parameters
            ----------
            None
                This helper does not accept parameters.
            """
            empty_pmid = jnp.zeros((conf.max_ptcl_per_slice, pmid.shape[1]), dtype=pmid.dtype)
            empty_disp = jnp.zeros((conf.max_ptcl_per_slice, disp.shape[1]), dtype=disp.dtype)
            empty_vel = jnp.zeros((conf.max_ptcl_per_slice, disp.shape[1]), dtype=vel.dtype)
            empty_acc = jnp.zeros((conf.max_ptcl_per_slice, disp.shape[1]), dtype=acc.dtype)

            # JAX 0.10 exposes this variance annotation through pcast. The
            # helper retains compatibility with JAX 0.6 on Python 3.10.
            empty_pmid = _mark_varying(empty_pmid)
            empty_disp = _mark_varying(empty_disp)
            empty_vel = _mark_varying(empty_vel)
            empty_acc = _mark_varying(empty_acc)

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
            halo_mask = Particles.compute_halo_mask(
                x_mod, conf.halo_start[slice_idx], conf.halo_end[slice_idx], unused_index
            )

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
        vel : ArrayLike or None, optional
            Particle velocity vectors.
        acc : ArrayLike or None, optional
            Particle acceleration vectors.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries."""
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
        """Construct particle state of ``pmid`` and ``disp`` from positions.

        Parameters
        ----------
        conf
            Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
        vel
            Particle velocity vectors.
        acc
            Particle acceleration vectors.
        wrap
            Whether positions should be wrapped into the periodic box before anchoring."""
        return cls.from_pos(conf, pos, vel=vel, acc=acc, wrap=wrap)

    @classmethod
    def from_ordered_pos(cls, conf, pos, vel=None, acc=None, wrap=True):
        """Construct particle state from positions stored in particle-grid order.

        This path keeps the particle-grid anchor unique by using the canonical
        `Particles.gen_grid(...)` ordering rather than re-rounding the input
        Eulerian positions. It is intended for ordered particle sets such as LPT
        outputs or CAMELS snapshots that preserve particle-grid order.

        Parameters
        ----------
        conf
            Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
        vel
            Particle velocity vectors.
        acc
            Particle acceleration vectors.
        wrap
            Whether positions should be wrapped into the periodic box before anchoring."""
        del wrap
        grid_axes = []
        for sp, sm in zip(conf.ptcl_grid_shape, conf.mesh_shape):
            axis = np.linspace(0, sm, num=sp, endpoint=False)
            axis = np.rint(axis).astype(np.dtype(conf.pmid_dtype))
            grid_axes.append(axis)
        pmid_host = np.meshgrid(*grid_axes, indexing='ij')
        pmid_host = np.stack(pmid_host, axis=-1).reshape(-1, conf.dim)

        pos_host = np.asarray(jax.device_get(pos), dtype=np.dtype(conf.float_dtype))
        if pos_host.shape[0] != pmid_host.shape[0]:
            raise ValueError(
                "from_ordered_pos requires a full particle-grid ordered position array: "
                f"expected {pmid_host.shape[0]} particles, got {pos_host.shape[0]}."
            )

        anchor_host = pmid_host.astype(pos_host.dtype, copy=False) * np.asarray(conf.cell_size, dtype=pos_host.dtype)
        box_host = np.asarray(conf.box_size, dtype=pos_host.dtype)
        disp_host = (pos_host - anchor_host + 0.5 * box_host) % box_host - 0.5 * box_host

        if conf.use_mGPU:
            pmid, disp, vel, acc, unused_index, halo_mask = cls._partition_and_shard_particle_fields(
                conf, pmid_host, disp_host, vel, acc
            )
            return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=unused_index, halo_mask=halo_mask)

        pmid = jnp.asarray(pmid_host, dtype=conf.pmid_dtype)
        disp = jnp.asarray(disp_host, dtype=conf.float_dtype)
        return cls(conf, pmid, disp, vel=vel, acc=acc, unused_index=None, halo_mask=None)

    @classmethod
    def from_pmid(cls, conf, pmid, disp, vel=None, acc=None):
        """Construct particle state of ``pmid`` and ``disp`` from positions.

        There may be collisions in particle ``pmid``.

        Parameters
        ----------
        conf : Configuration
        pos : ArrayLike
            Particle positions in [L].
        vel : ArrayLike or None, optional
            Particle velocity vectors.
        acc : ArrayLike or None, optional
            Particle acceleration vectors.
        wrap : bool, optional
            Whether to wrap around the periodic boundaries."""
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

        Parameters
        ----------
        ptcl
            Particle state passed through the solver."""
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
            """Construct the local ordered particle grid for one GPU slab.

            Parameters
            ----------
            gpu_id
                Index of the local GPU/device in the slab decomposition.
            """
            pmid, disp = [], []

            sp = conf.ptcl_grid_shape[0]
            runtime = conf.multigpu
            store_particle_halos = runtime is not None and runtime.store_particle_halos
            n_devices = conf.num_devices or 1
            if store_particle_halos:
                sm = conf.local_mesh_shape[0] + conf.ptcl_halo_width
                n_ptcl = sp // n_devices + (1 if n_devices > 1 else 0)
                pmid_shift = -conf.ptcl_halo_width
            else:
                sm = conf.local_mesh_shape[0]
                n_ptcl = sp // n_devices
                pmid_shift = 0
            pmid_x = jnp.linspace(0, sm, num=n_ptcl, endpoint=False)
            offset = 0 if conf.offsets is None else conf.offsets[gpu_id]
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
            capacity = conf.ptcl_num if conf.max_ptcl_per_slice is None else conf.max_ptcl_per_slice
            pmid = jnp.pad(pmid, ((0, capacity - pmid.shape[0]), (0, 0)), mode='constant')
            disp = jnp.meshgrid(*disp, indexing='ij')
            disp = jnp.stack(disp, axis=-1).reshape(-1, conf.dim)
            disp = jnp.pad(disp, ((0, capacity - disp.shape[0]), (0, 0)), mode='constant')

            if conf.compute_mesh is None:
                unused_index = jnp.zeros_like(disp[:, 0], dtype=jnp.bool_)
            else:
                unused_index = jnp.zeros_like(
                    disp[:, 0], dtype=jnp.bool_, device=NamedSharding(conf.compute_mesh, P(AXIS_NAME))
                )
            unused_index = unused_index.at[n_ptcl * sp * sp:].set(True)

            if runtime is None or not store_particle_halos:
                halo_mask = jnp.zeros_like(unused_index)
            else:
                x_mod = (pmid[:, 0] + disp[:, 0] * conf.disp_size) % conf.nMesh
                halo_mask = Particles.compute_halo_mask(
                    x_mod, conf.halo_start[gpu_id], conf.halo_end[gpu_id], unused_index
                )

            return pmid, disp, unused_index, halo_mask

        @partial(
            shard_map, mesh=conf.compute_mesh, in_specs=(P()),
            out_specs=(P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME), P(AXIS_NAME))
        )
        def build_all():
            """Construct ordered particle storage for all configured GPU slabs.

            Parameters
            ----------
            None
                This helper does not accept parameters.
            """
            axis = jax.lax.axis_index(AXIS_NAME)
            return build_local(axis)

        if conf.compute_mesh is None:
            pmid, disp, unused_index, halo_mask = build_local(0)
        else:
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

        Raises
        ------
        OverflowError
            If JAX canonicalizes ``dtype`` to an integer type that cannot
            represent every global mesh index.  Distributed routing uses its
            own two-limb key representation and does not call this helper.

        """
        conf = self.conf

        canonical_dtype = jax.dtypes.canonicalize_dtype(dtype)
        if not jnp.issubdtype(canonical_dtype, jnp.integer):
            raise TypeError("raveled particle IDs require an integer dtype")
        if conf.mesh_size - 1 > jnp.iinfo(canonical_dtype).max:
            raise OverflowError(
                f"mesh size {conf.mesh_size} does not fit canonical JAX dtype {canonical_dtype}; "
                "enable x64 or use the distributed two-limb routing-key helpers"
            )

        pmid = self.pmid
        if wrap:
            pmid = pmid % jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)

        strides = tuple(accumulate((1, ) + conf.mesh_shape[:0:-1], mul))[::-1]

        raveled_id = sum(i.astype(canonical_dtype) * s for i, s in zip(pmid.T, strides))

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
        """Convert mesh-cell IDs and displacements back to positions.

        Parameters
        ----------
        pmid
            Integer particle mesh cell identifiers.
        disp
            Particle displacement vectors relative to ``pmid`` cells.
        conf
            Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
        """
        pos = pmid * conf.cell_size + disp
        pos %= jnp.array(conf.box_size)
        return pos

    @staticmethod
    def pos_to_pmid(pos, conf):
        """Convert positions into nearest mesh-cell IDs and displacements.

        Parameters
        ----------
        pos
            Particle positions in box units.
        conf
            Configuration object that defines mesh sizes, dtypes, units, and multi-GPU runtime helpers.
        """
        pmid = jnp.rint(pos / conf.cell_size)
        disp = pos - pmid * conf.cell_size

        pmid = pmid.astype(conf.pmid_dtype)
        disp = disp.astype(conf.float_dtype)

        pmid %= jnp.array(conf.mesh_shape, dtype=conf.pmid_dtype)
        return pmid, disp

    def values_on_device(self, device_id):
        """Select values that belong to one local device.

        Parameters
        ----------
        device_id
            Index of the local device whose values are selected.
        """
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
        pmid, disp, vel, acc, unused_indexes, new_pmid, new_disp, new_vel, new_acc, new_valid, max_values_to_add,
    ):
        """Append incoming particles into the unused slots of a padded particle buffer.

        Parameters
        ----------
        pmid
            Integer particle mesh cell identifiers.
        disp
            Particle displacement vectors relative to ``pmid`` cells.
        vel
            Particle velocity vectors.
        acc
            Particle acceleration vectors.
        unused_indexes
            Boolean mask marking padded or inactive particle slots.
        new_pmid
            Mesh cell identifiers for particles being inserted into a padded buffer.
        new_disp
            Displacements for particles being inserted into a padded buffer.
        new_vel
            Velocities for particles being inserted into a padded buffer.
        new_acc
            Accelerations for particles being inserted into a padded buffer.
        new_valid
            Boolean validity mask for particles being inserted into a padded buffer.
        max_values_to_add
            Static capacity limit used to size padded multi-GPU communication buffers.
        """
        max_values_to_add = min(max_values_to_add, pmid.shape[0])
        num_values_to_add = jnp.sum(new_valid)

        _ = jax.lax.cond(
            jnp.sum(unused_indexes) < num_values_to_add, lambda _: raise_error(
                "[ERROR] Exceeded max_amount_particles_per_slice. Available slots: {x}, values_to_add: {y}. Consider making 'max_amount_particles_per_slice' bigger.",
                x=jnp.sum(unused_indexes), y=num_values_to_add
            ), lambda _: None, operand=None
        )

        # Pad with a stable sentinel slot instead of 0. Using 0 here can produce repeated
        # destination indices after the real unused slots, and those padded `.at[...].set`
        # writes can overwrite an earlier successful insertion on slot 0.
        real_indices = jnp.nonzero(unused_indexes, size=max_values_to_add, fill_value=pmid.shape[0] - 1, )[0]

        valid_new_indices = jnp.nonzero(new_valid, size=max_values_to_add, fill_value=0)[0]

        chosen_new_pmid = jax.lax.dynamic_slice_in_dim(
            new_pmid[valid_new_indices], start_index=0, slice_size=max_values_to_add, axis=0
        )
        chosen_new_disp = jax.lax.dynamic_slice_in_dim(
            new_disp[valid_new_indices], start_index=0, slice_size=max_values_to_add, axis=0
        )
        chosen_new_vel = jax.lax.dynamic_slice_in_dim(
            new_vel[valid_new_indices], start_index=0, slice_size=max_values_to_add, axis=0
        )
        chosen_new_acc = jax.lax.dynamic_slice_in_dim(
            new_acc[valid_new_indices], start_index=0, slice_size=max_values_to_add, axis=0
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
