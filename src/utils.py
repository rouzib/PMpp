import dataclasses
from dataclasses import field
from functools import partial
from pprint import pformat
import numpy as np

import jax
import jax.numpy as jnp
from jax import Array, float0
from jax.tree_util import register_pytree_node, tree_leaves, tree_map
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


def build_particle_nyquist_filter(kvec, conf):
    """Return per-axis broadcastable masks for particle-grid-resolvable modes.

    Parameters
    ----------
    kvec : sequence of jax.Array
        Sparse broadcastable wavevector components on the active mesh layout.
    conf : Configuration
        Active simulation configuration.

    Returns
    -------
    tuple of jax.Array
        Broadcastable masks, one per axis, that keep only modes resolvable on
        the particle grid.
    """
    if conf.mesh_shape == conf.ptcl_grid_shape:
        return ()

    k_nyquist = jnp.asarray(jnp.pi / conf.ptcl_spacing, dtype=conf.float_dtype)
    eps = k_nyquist * jnp.asarray(8 * jnp.finfo(conf.float_dtype).eps, dtype=conf.float_dtype)
    limit = k_nyquist + eps
    return tuple((jnp.abs(k) <= limit).astype(conf.float_dtype) for k in kvec)


def pytree_dataclass(cls, aux_fields=None, aux_invert=False, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    Also added are methods that return children and aux_data iterators, and pretty
    string representation, and a method that replace fields with changes.

    Parameters
    ----------
    cls : type
        Class to be registered, not a python dataclass yet.
    aux_fields : str, sequence of str, or Ellipsis, optional
        Pytree aux_data fields. Default is none; unrecognized ones are ignored;
        ``Ellipsis`` uses all.
    aux_invert : bool, optional
        Whether to invert ``aux_fields`` selections, convenient when most but not all
        fields are aux_data.
    **kwargs
        Keyword arguments to be passed to python dataclass decorator.

    Returns
    -------
    cls : type
        Registered dataclass.

    Raises
    ------
    TypeError
        If cls is already a python dataclass.

    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d

    .. _flax.struct package — Flax documentation:
        https://flax.readthedocs.io/en/latest/flax.struct.html

    .. _JAX Issue #2371:
        https://github.com/google/jax/issues/2371

    """
    if dataclasses.is_dataclass(cls):
        raise TypeError('cls cannot already be a dataclass')
    cls = dataclasses.dataclass(cls, **kwargs)

    if aux_fields is None:
        aux_fields = ()
    elif isinstance(aux_fields, str):
        aux_fields = (aux_fields,)
    elif aux_fields is Ellipsis:
        aux_fields = [field.name for field in dataclasses.fields(cls)]
    aux_data_names = [field.name for field in dataclasses.fields(cls)
                      if field.name in aux_fields]
    children_names = [field.name for field in dataclasses.fields(cls)
                      if field.name not in aux_fields]

    if aux_invert:
        aux_data_names, children_names = children_names, aux_data_names

    def children(self):
        """Return an iterator over pytree children values."""
        for name, value in self.named_children():
            yield value

    def named_children(self):
        """Return an iterator over pytree children names and values."""
        for name in children_names:
            value = getattr(self, name)
            yield name, value

    def aux_data(self):
        """Return an iterator over pytree aux_data values."""
        for name, value in self.named_aux_data():
            yield value

    def named_aux_data(self):
        """Return an iterator over pytree aux_data names and values."""
        for name in aux_data_names:
            value = getattr(self, name)
            yield name, value

    cls.children = children
    cls.named_children = named_children
    cls.aux_data = aux_data
    cls.named_aux_data = named_aux_data

    def tree_flatten(obj):
        # FIXME JAX doesn't like the flatten function to return iterators, and somehow
        # triggered AssertionError by _closure_convert_for_avals in custom_derivatives.py
        return tuple(obj.children()), tuple(obj.aux_data())

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(children_names, children)),
                   **dict(zip(aux_data_names, aux_data)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    def _is_transforming(self):
        """Whether dataclass fields are pytrees initialized by JAX transformations.

        .. _Pytrees — JAX documentation:
            https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization

        .. _JAX Issue #10238:
            https://github.com/google/jax/issues/10238

        """

        def leaves_all(is_placeholder, tree):
            # similar to tree_all(tree_map(is_placeholder, tree))
            return all(is_placeholder(x) for x in tree_leaves(tree))

        # unnecessary to test for None's since they are empty pytree nodes
        return tree_leaves(self) and leaves_all(
            lambda x: type(x) is object or isinstance(x, str),
            self,
        )

    cls._is_transforming = _is_transforming

    def __str__(self):
        """Pretty string representation for python >= 3.10."""
        return pformat(self)

    cls.__str__ = __str__

    def replace(self, **changes):
        """Create a new object of the same type, replacing fields with changes."""
        return dataclasses.replace(self, **changes)

    cls.replace = replace

    return cls


import timeit

AXIS_NAME = "gpus"


def raise_error(err_msg, **error_dict):
    """
    Raises an error by printing the error message and the associated details.

    This function utilizes the `jax.debug.print` mechanism to output error
    messages, along with any additional contextual information provided as a
    dictionary. It can be used for debugging purposes, ensuring that errors
    are logged with their accompanying details in a structured manner.

    :param err_msg: The message describing the error.
    :type err_msg: str
    :param error_dict: A dictionary containing additional context
        or details about the error.
    :type error_dict: dict
    :return: None
    """
    jax.debug.print(err_msg, **error_dict)


def create_compute_mesh(devices):
    """Create the one-dimensional device mesh used by PM++ multi-GPU paths.

    Parameters
    ----------
    devices : sequence of jax.Device
        Devices to arrange along the slab-decomposition axis.

    Returns
    -------
    jax.sharding.Mesh
        One-dimensional mesh named by ``AXIS_NAME``.
    """
    device_mesh = mesh_utils.create_device_mesh((len(devices),), devices=devices)
    return Mesh(device_mesh, axis_names=(AXIS_NAME,))  # "gpus" is necessary for all other


def distribute_array_on_gpus(array: Array, compute_mesh: Mesh, partition: P) -> Array:
    """Place an array onto a compute mesh with explicit sharding.

    Parameters
    ----------
    array : jax.Array
        Input array already shaped consistently with ``partition``.
    compute_mesh : Mesh
        Device mesh defining the target sharding.
    partition : PartitionSpec
        Partition specification for the output array.

    Returns
    -------
    jax.Array
        Array materialized on ``compute_mesh`` with the requested sharding.
    """
    sharding = NamedSharding(compute_mesh, partition)
    array_parts_device = [jax.device_put(array[i], device=d) for d, i in
                          sharding.addressable_devices_indices_map(array.shape).items()]
    array_distributed = jax.make_array_from_single_device_arrays(array.shape, sharding, array_parts_device)
    return array_distributed


def is_float0_array(x):
    """Return whether ``x`` is JAX's ``float0`` cotangent sentinel.

    Parameters
    ----------
    x : Any
        Candidate object to test.

    Returns
    -------
    bool
        True when ``x`` is a JAX ``float0`` array.
    """
    return hasattr(x, 'dtype') and x.dtype == float0


def pmid_to_idx(pmid, conf, unused_index=None, dtype=jnp.int32):
    """Pack mesh-index triplets into the legacy flat particle key when required.

    Parameters
    ----------
    pmid : ArrayLike
        Mesh-index triplets for each particle slot.
    conf : Configuration
        Active simulation configuration.
    unused_index : ArrayLike or None, optional
        Optional boolean padding mask. Masked entries are set to ``-1``.
    dtype : DTypeLike, optional
        Integer dtype for the packed key.

    Returns
    -------
    jax.Array
        Flat particle keys matching the removed legacy ``Particles.idx``
        convention.

    Notes
    -----
    The default ``int32`` matches the removed ``Particles.idx`` field exactly.
    """
    mesh_shape = jnp.array(conf.mesh_shape, dtype=dtype)
    ix = (pmid[:, 0].astype(dtype)) % mesh_shape[0]
    iy = (pmid[:, 1].astype(dtype)) % mesh_shape[1]
    iz = (pmid[:, 2].astype(dtype)) % mesh_shape[2]

    idx = (ix * mesh_shape[1] + iy) * mesh_shape[2] + iz

    if unused_index is not None:
        idx = jnp.where(unused_index, dtype(-1), idx)

    return idx


def build_ring_permutations(num_devices):
    """Build left/right ring permutations for ``lax.ppermute``.

    Parameters
    ----------
    num_devices : int
        Number of devices in the one-dimensional slab decomposition.

    Returns
    -------
    tuple
        ``(left_perm, right_perm)`` permutations for neighbor exchange.
    """
    left_perm = tuple((i, (i - 1) % num_devices) for i in range(num_devices))
    right_perm = tuple((i, (i + 1) % num_devices) for i in range(num_devices))
    return left_perm, right_perm


def measure_execution_time(func, repetitions=5, number: int = 5):
    """Measure wall-clock execution time for a callable.

    Parameters
    ----------
    func : callable
        Callable to execute and block until ready.
    repetitions : int, optional
        Number of repeated timing groups.
    number : int, optional
        Calls per timing group.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of the per-call execution time in seconds.
    """
    # Wrap the JAX function to ensure it blocks until computations are done
    timer = timeit.Timer(lambda: jax.block_until_ready(func()))

    # Measure execution times for the given repetitions and number of iterations
    total_times = timer.repeat(repeat=repetitions, number=number)

    # Compute statistics in seconds
    average_time = np.mean(total_times) / number  # Average execution time per call
    std_dev_time = np.std(total_times) / number  # Standard deviation per call

    return average_time, std_dev_time


def get_a_schedule(target_z, conf):
    """Build a scale-factor schedule that includes requested output redshifts.

    Parameters
    ----------
    target_z : array-like
        Redshifts that must appear in the schedule.
    conf : Configuration
        Configuration providing the default N-body schedule and step-size limit.

    Returns
    -------
    jax.Array
        Scale-factor schedule containing the requested outputs plus any
        interpolated intermediate steps needed to respect
        ``conf.a_nbody_maxstep``.
    """
    spacing_threshold = conf.a_nbody_maxstep  # in scale_factor

    def get_intermediary_z(target_z):
        """
        Generate a schedule of intermediary redshift (z) values based on the provided target redshift
        and configuration. This function ensures the schedule is appropriately spaced to meet a
        spacing threshold.

        :param target_z: A sequence of target redshift values to include in the schedule.
            Must be unique and sorted in ascending order.
        :type target_z: jnp.ndarray
        :param conf: Configuration dictionary or object providing settings for generating
            the schedule, including parameters such as spacing threshold.
        :type conf: dict
        :return: Array of intermediary redshift (z) values not originally in the input
            `target_z`. The function respects spacing constraints and ensures unique
            redshift entries.
        :rtype: jnp.ndarray
        """
        # Ensure unique, sorted redshifts (lowest z last, so a ascending)
        target_z = jnp.unique(target_z)
        target_a = 1 / (1 + target_z)
        target_a = jnp.sort(target_a)

        diffs = jnp.diff(target_a)
        n_steps = jnp.ceil(diffs / spacing_threshold).astype(int)

        all_a = [target_a[0]]

        for i, steps in enumerate(n_steps):
            steps = int(steps)
            if steps < 1:
                steps = 1  # In case of floating fp error or very small interval

            if steps == 1:
                # No interpolation needed, just add next point
                all_a.append(target_a[i + 1])
            else:
                # Create intermediate values, excluding the first since it's already in all_a
                sub_a = jnp.linspace(target_a[i], target_a[i + 1], steps + 1)[1:]
                all_a.extend(sub_a)

        all_a = jnp.array(all_a)
        all_z = 1 / all_a - 1

        # Only return those not in the original target_z (to match previous expectations)
        mask = ~jnp.isin(jnp.round(all_z, 8), jnp.round(target_z, 8))
        intermediary_z = all_z[mask]

        return intermediary_z

    # Find any necessary intermediary redshifts
    intermediary_z = get_intermediary_z(target_z)

    # Prepare the usual_pm_z array, filtered as before
    usual_pm_z = (1 / conf.a_nbody - 1)
    usual_pm_z = jnp.array([usual_pm_z[i] for i in range(len(usual_pm_z)) if jnp.max(target_z) < usual_pm_z[i]])

    # Concatenate all sources, sort in descending order (high z to low z)
    new_pm_z = jnp.concatenate((target_z, usual_pm_z, intermediary_z))
    new_pm_z = jnp.sort(new_pm_z)[::-1]  # descending order

    return 1 / (1 + new_pm_z)


@partial(jax.jit, static_argnames=['max_slice_len', 'axis'])
def wraparound_slice(array, start, stop, max_slice_len, axis=0):
    """Take a periodic slice with fixed output shape for JIT compatibility.

    Parameters
    ----------
    array : jax.Array
        Input array.
    start, stop : int
        Slice bounds in periodic index space.
    max_slice_len : int
        Static maximum output length.
    axis : int, optional
        Axis along which to slice.

    Returns
    -------
    jax.Array
        Wrapped slice padded with zeros to ``max_slice_len``.
    """
    # Determine the size of the dimension being sliced
    n = array.shape[axis]

    # Calculate the actual length of this specific slice
    true_len = stop - start

    # Generate a full-size sequence of potential indices starting from `start`
    # This works because `max_slice_len` is static.
    indices = jnp.arange(max_slice_len) + start

    # Apply the modulo operator to wrap the indices around the dimension size
    wrapped_indices = jnp.mod(indices, n)

    # Gather the elements from the input array using the wrapped indices.
    # This result will always have a shape determined by `max_slice_len`.
    padded_result = jnp.take(array, wrapped_indices, axis=axis)

    # Create a mask to zero out the elements that are just padding
    mask = jnp.arange(max_slice_len) < true_len
    rank = array.ndim
    new_shape = [1] * rank
    new_shape[axis] = max_slice_len
    reshaped_mask = mask.reshape(new_shape)

    return padded_result * reshaped_mask
