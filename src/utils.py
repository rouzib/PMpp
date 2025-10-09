import dataclasses
from dataclasses import field
from pprint import pformat
import numpy as np

import jax
from jax import Array, float0
from jax.tree_util import register_pytree_node, tree_leaves, tree_map
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


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
        # F IXME JAX doesn't like the flatten function to return iterators, and somehow
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
        return tree_leaves(self) and leaves_all(lambda x: type(x) is object, self)

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
    device_mesh = mesh_utils.create_device_mesh((len(devices), ), devices=devices)
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
    return isinstance(x, np.ndarray) and x.dtype == float0


def build_ring_permutations(num_devices):
    """
    Build two permutation lists for ppermute in a 1D ring topology:
      - left_perm:  (i -> i-1 mod N)
      - right_perm: (i -> i+1 mod N)
    """
    left_perm = tuple((i, (i - 1) % num_devices) for i in range(num_devices))
    right_perm = tuple((i, (i + 1) % num_devices) for i in range(num_devices))
    return left_perm, right_perm


def measure_execution_time(func, repetitions=5, number=5):
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
