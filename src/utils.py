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
    """Return per-axis broadcastable masks for particle-grid-resolvable modes."""
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
    """Creates a compute mesh from the specified devices.

    This function takes a list of devices and utilizes the
    `mesh_utils.create_device_mesh` function to create a device
    mesh. The resulting mesh is then wrapped in a `Mesh` object
    with specified `axis_names`.

    Args:
        devices (list): A list of device identifiers that will be
            used to create the device mesh.

    Returns:
        Mesh: A `Mesh` object representing the created compute
        mesh using the provided devices.
    """
    device_mesh = mesh_utils.create_device_mesh((len(devices),), devices=devices)
    return Mesh(device_mesh, axis_names=(AXIS_NAME,))  # "gpus" is necessary for all other


def distribute_array_on_gpus(array: Array, compute_mesh: Mesh, partition: P) -> Array:
    """
    Distributes the given array across multiple GPUs for computation. The distribution follows a partition configuration
    and an axis along which the array should be split for distribution.

    Args:
        array (np.ndarray): The input array to be distributed.
        compute_mesh (Mesh): The compute mesh that defines the layout of GPUs.
        partition (P): The partition configuration for distributing the array.

    Returns:
        jnp.ndarray: A Jax array distributed across multiple GPUs.
    """
    sharding = NamedSharding(compute_mesh, partition)
    array_parts_device = [jax.device_put(array[i], device=d) for d, i in
                          sharding.addressable_devices_indices_map(array.shape).items()]
    array_distributed = jax.make_array_from_single_device_arrays(array.shape, sharding, array_parts_device)
    return array_distributed


def is_float0_array(x):
    return hasattr(x, 'dtype') and x.dtype == float0


def pmid_to_idx(pmid, conf, unused_index=None, dtype=jnp.int32):
    """Pack mesh-index triplets into the legacy flat particle key when required.

    This intentionally defaults to ``int32`` to match the removed ``Particles.idx``
    field exactly. Larger dtypes can still be requested explicitly by callers.
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
    """
    Build two permutation lists for ppermute in a 1D ring topology:
      - left_perm:  (i -> i-1 mod N)
      - right_perm: (i -> i+1 mod N)
    """
    left_perm = tuple((i, (i - 1) % num_devices) for i in range(num_devices))
    right_perm = tuple((i, (i + 1) % num_devices) for i in range(num_devices))
    return left_perm, right_perm


def measure_execution_time(func, repetitions=5, number: int = 5):
    """
    Measure the execution time of a function and compute the average and standard deviation.

    Parameters:
        func (callable): The JAX function to execute and block until ready.
        repetitions (int): Number of times to repeat the measurement (default 5).
        number (int): Number of iterations per measurement (default 5).

    Returns:
        tuple: (average_time, std_dev_time) in seconds.
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
    """
    Calculates a schedule of ascending scale factors based on given target redshifts and configuration parameters.

    This function generates an optimized schedule of scale factors through interpolation
    and inclusion of necessary intermediary values to ensure spacing thresholds are respected.
    It also incorporates additional predefined configurations into the schedule. The resulting
    schedule satisfies specific numerical needs for an astrophysical simulation.

    :param target_z: The array of target redshift values to use in the calculation.
    :type target_z: jnp.ndarray
    :param conf: Configuration object containing attributes that specify simulation parameters,
        such as `a_nbody_maxstep` and `a_nbody`.
    :type conf: Any
    :return: An array of scale factor values in ascending order computed from given redshifts,
        accounting for intermediary redshift values (if necessary).
    :rtype: jnp.ndarray
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
    """
    Performs a jittable, padded wraparound slice on a JAX array.

    This function always returns an array of size `max_slice_len`.

    Args:
      array: The input array.
      start: The starting index of the slice (can be a JAX tracer).
      stop: The stopping index of the slice (can be a JAX tracer).
      max_slice_len: The maximum possible length of any slice.
                     This MUST be a static, compile-time constant.
      axis: The axis along which to slice.

    Returns:
      The sliced portion of the array, padded with zeros to `max_slice_len`.
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
