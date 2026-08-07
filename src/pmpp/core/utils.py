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
        https://github.com/jax-ml/jax/issues/2371

    """
    if dataclasses.is_dataclass(cls):
        raise TypeError('cls cannot already be a dataclass')
    cls = dataclasses.dataclass(cls, **kwargs)

    all_fields_static = aux_fields is Ellipsis
    if aux_fields is None:
        aux_fields = ()
    elif isinstance(aux_fields, str):
        aux_fields = (aux_fields, )
    elif aux_fields is Ellipsis:
        aux_fields = [field.name for field in dataclasses.fields(cls)]
    aux_data_names = [field.name for field in dataclasses.fields(cls) if field.name in aux_fields]
    children_names = [field.name for field in dataclasses.fields(cls) if field.name not in aux_fields]

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
        """Split a dataclass pytree into dynamic children and static metadata.

        Parameters
        ----------
        obj
            Dataclass instance being flattened into JAX pytree children.
        """
        if all_fields_static:
            # Preserve the initialized object itself. Reconstructing an
            # all-static Configuration by calling its constructor inside a
            # shard_map reruns __post_init__ under the manual axis context;
            # JAX 0.10 then assigns manual sharding to cached setup arrays.
            return (), obj
        return tuple(obj.children()), tuple(obj.aux_data())

    def tree_unflatten(aux_data, children):
        """Reconstruct a dataclass pytree from static metadata and dynamic children.

        Parameters
        ----------
        aux_data
            Static dataclass metadata saved during pytree flattening.
        children
            Dynamic pytree children restored into the dataclass instance.
        """
        if all_fields_static:
            return aux_data
        return cls(**dict(zip(children_names, children)), **dict(zip(aux_data_names, aux_data)))

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
            """Test whether all non-placeholder leaves satisfy a predicate.

            Parameters
            ----------
            is_placeholder
                Predicate identifying placeholder leaves that should be ignored.
            tree
                Pytree whose leaves are checked.
            """
            return all(is_placeholder(x) for x in tree_leaves(tree))

        # unnecessary to test for None's since they are empty pytree nodes
        return tree_leaves(self) and leaves_all(lambda x: type(x) is object or isinstance(x, str), self, )

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
    """Abort a compiled run when a static-capacity invariant is violated.

    PM++ historically only printed these messages and then continued with a
    truncated static buffer.  Continuing produces a numerically plausible but
    scientifically invalid trajectory, so capacity failures must be observable
    as real runtime errors by training and acceptance harnesses.  A debug
    callback is used because the predicate is normally evaluated inside a
    jitted/sharded computation.

    :param err_msg: The message describing the error.
    :type err_msg: str
    :param error_dict: A dictionary containing additional context
        or details about the error.
    :type error_dict: dict
    :raises RuntimeError: when the compiled callback executes
    """

    def _raise_on_host(**values):
        formatted = {}
        for name, value in values.items():
            array = np.asarray(value)
            formatted[name] = array.item() if array.ndim == 0 else array.tolist()
        raise RuntimeError(err_msg.format(**formatted))

    jax.debug.callback(_raise_on_host, **error_dict)


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
    device_mesh = mesh_utils.create_device_mesh((len(devices), ), devices=devices)
    return Mesh(device_mesh, axis_names=(AXIS_NAME, ))  # "gpus" is necessary for all other


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
    array_parts_device = [
        jax.device_put(array[i], device=d) for d, i in sharding.addressable_devices_indices_map(array.shape).items()
    ]
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
    spacing_threshold = float(conf.a_nbody_maxstep)
    if spacing_threshold <= 0:
        raise ValueError("conf.a_nbody_maxstep must be positive")

    target_z = jnp.unique(jnp.asarray(target_z))
    if target_z.size == 0:
        raise ValueError("target_z must contain at least one redshift")

    target_a = 1 / (1 + target_z)
    usual_a = jnp.asarray(conf.a_nbody)
    usual_z = 1 / usual_a - 1
    usual_a = usual_a[usual_z > jnp.max(target_z)]
    required_a = jnp.sort(jnp.unique(jnp.concatenate((target_a, usual_a))))

    all_a = [required_a[0]]
    for start, stop in zip(required_a[:-1], required_a[1:]):
        steps = max(1, int(jnp.ceil((stop - start) / spacing_threshold)))
        if steps == 1:
            all_a.append(stop)
        else:
            all_a.extend(jnp.linspace(start, stop, steps + 1)[1:])

    return jnp.asarray(all_a)


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
