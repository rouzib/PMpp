"""
Author: Nicolas Payot
Date: 06/18/2024

This Python script handles large scale Fourier Transforms utilizing multiple GPUs for computation. In particular, it
provides the functionality to distribute an ndarray across multiple GPUs for efficient computation of the Fourier
Transform.

The primary functions are split_array_for_gpus, distribute_array_on_gpus, create_sharded_fft, and create_ffts which
facilitate array division, allocation of array segments to GPUs, creating sharded versions of an FFT function, and
creating the main FFT functions for complex and real input that work in a distributed fashion across GPUs. Also, there
is a test function to compare the output and performance of multi-GPU FFT against a reference FFT function.

These functions are highly beneficial when dealing with extremely large arrays where parallelization of computation can
lead to significant performance improvements. The code leverages both numpy and jax libraries for computation and
parallelization.

Note: The script expects a computing environment with multiple GPUs available.

"""

import numpy as np
from typing import Callable, Tuple
from functools import partial

import jax
from jax import lax
from jax import custom_vjp
from jax.experimental import mesh_utils
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.tree_util as tree
import jax.numpy as jnp


def split_array_for_gpus(array: np.ndarray, num_gpus: int, axis: int = 1) -> np.ndarray:
    """
    Splits the given array into equal parts along the second axis for distribution across multiple GPUs.

    Args:
    array (np.ndarray): The input array to be split.
    num_gpus (int): The number of GPUs across which the array needs to be split.
    axis (int): The axis along which the array needs to be split

    Returns:
    np.ndarray: A NumPy array containing sub-arrays, each representing a portion of the original array intended for one GPU.

    Note:
    This function assumes that the division of the array size by the number of GPUs results in an integer number of elements.
    The caller must ensure that `num_gpus` evenly divides the second dimension of `array`.
    """
    # Split the array equally along the second dimension for each GPU
    return jnp.array(jnp.array_split(array, num_gpus, axis=axis))


def distribute_array_on_gpus(array: np.ndarray, compute_mesh: Mesh, partition: P,
                             axis_name: str = "gpus") -> jnp.ndarray:
    """
    Distributes the given array across multiple GPUs for computation. The distribution follows a partition configuration
    and an axis along which the array should be split for distribution.

    Args:
    array (np.ndarray): The input array to be distributed.
    compute_mesh (Mesh): The compute mesh that defines the layout of GPUs.
    partition (P): The partition configuration for distributing the array.
    axis_name (str): The axis along which the array needs to be split for distribution. Defaults to "gpus".

    Returns:
    jnp.ndarray: A Jax array distributed across multiple GPUs.

    Note:
    The function 'split_array_for_gpus' is used to split the array equally for each GPU before distributing. Ensure that
    the number of GPUs evenly divides the dimension corresponding to `axis_name` of `array` which is governed by the 'partition' configuration.
    """
    # Get the number of GPUs
    num_gpus = len(compute_mesh.devices)
    # Split the array equally for each GPU
    array_parts = split_array_for_gpus(array, num_gpus, partition.index(axis_name))
    with compute_mesh:
        # Distribute the array parts across different GPUs
        array_parts_device = [jax.device_put(part, device) for part, device in zip(array_parts, compute_mesh.devices)]
        array_distributed = jax.make_array_from_single_device_arrays(array.shape,
                                                                     NamedSharding(compute_mesh, partition),
                                                                     array_parts_device)
        return array_distributed


def create_sharded_fft(basic_fft: Callable, partition_spec: P):
    """
    Creates a sharded version of an FFT function that operates across devices according
    to a specified partitioning scheme.

    Args:
    basic_fft: A function that performs an FFT operation.
    partition_spec: A PartitionSpec specifying how data should be sharded across devices.

    Returns:
    A function that applies the FFT operation with the specified sharding.
    """

    @custom_partitioning
    def sharded_fft(x):
        """Applies the FFT function to the input array with sharding applied."""
        return basic_fft(x)

    def supported_sharding(sharding, shape):
        """Returns a NamedSharding based on the input sharding and the partition specification."""
        return NamedSharding(sharding.mesh, partition_spec)

    def partition(mesh, arg_shapes, result_shape):
        """Defines how to partition the FFT computation across the devices."""
        arg_shardings = tree.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, basic_fft, supported_sharding(arg_shardings[0], arg_shapes[0]), (
            supported_sharding(arg_shardings[0], arg_shapes[0]),)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        """Infers the sharding of the output based on the input sharding."""
        arg_shardings = tree.tree_map(lambda x: x.sharding, arg_shapes)
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    sharded_fft.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition
    )
    return sharded_fft


def _fftn_first_pass(x):
    """ Perform the FFT along the first two axes. """
    return jnp.fft.fftn(x, axes=[0, 1])


def _fftn_second_pass(x):
    """ Perform the FFT along the last axis using FFT due to symmetry. """
    return jnp.fft.fft(x, axis=2)


def _rfftn_second_pass(x):
    """ Perform the real FFT along the last axis using real FFT due to symmetry. """
    return jnp.fft.rfft(x, axis=2)


def _ifftn_first_pass(x):
    """ Perform the inverse FFT along the first two axes. """
    return jnp.fft.ifftn(x, axes=[0, 1])


def _ifftn_second_pass(x):
    """ Perform the inverse FFT along the last axis using FFT due to symmetry. """
    return jnp.fft.ifft(x, axis=2)


def _irfftn_second_pass(x):
    """ Perform the real inverse FFT along the last axis using real FFT due to symmetry. """
    return jnp.fft.irfft(x, axis=2)


def create_ffts(compute_mesh: Mesh, max_n:int) -> Tuple[Callable, Callable, Callable, Callable]:
    """
    Create a set of Fourrier Transform functions that distribute computation across a provided compute mesh (a logical
    grouping of devices for parallel computation).  It returns a tuple of functions: `rfftn_jit`, `irfftn_jit`, `fftn_jit`,
    and `ifftn_jit` for performing real forward, real inverse, complex forward and complex inverse FFT respectively.

    Args:
        compute_mesh (Mesh): The compute mesh defining the layout of devices for parallel computation.
        max_n (int): The maximum size of elements along the largest dimensions of the input arrays.

    Returns:
        tuple: The forward and inverse FFT functions including real and complex variants. These functions are JIT
        compiled and are set up for sharding the computation across multiple devices defined by the `compute_mesh`.
        The FFT functions utilize a two-pass method in which the FFT operation is performed on subsets of the array's
        dimensions.

    Note:
        The returned FFT and IFFT functions are 'jit-ted' (Just-In-Time compiled) for improved performance. They are also
        set up to perform their computations with input data that are distributed across devices depending on the
        `compute_mesh`. This is performed by making use of the earlier defined sharded FFT functions.

        Among the returned functions, `rfftn` and `irfftn` functions are differentiable while `fftn` and `ifftn` are not.

    """
    # Creating sharded versions of FFT and IFFT functions for specific GPU sharding
    fftn_first_pass = create_sharded_fft(_fftn_first_pass, P(None, None, "gpus"))
    fftn_second_pass = create_sharded_fft(_fftn_second_pass, P(None, "gpus"))
    rfftn_second_pass = create_sharded_fft(_rfftn_second_pass, P(None, "gpus"))
    ifftn_first_pass = create_sharded_fft(_ifftn_first_pass, P(None, None, "gpus"))
    ifftn_second_pass = create_sharded_fft(_ifftn_second_pass, P(None, "gpus"))
    irfftn_second_pass = create_sharded_fft(_irfftn_second_pass, P(None, "gpus"))

    def _fftn(x):
        """ Perform a forward FFT. """
        x = fftn_second_pass(x)
        x = fftn_first_pass(x)
        return x

    def _ifftn(x):
        """ Perform an inverse FFT. """
        x = ifftn_first_pass(x)
        x = ifftn_second_pass(x)
        return x

    def _rfftn(x):
        """ Perform a real-valued forward FFT. """
        x = rfftn_second_pass(x)
        x = fftn_first_pass(x)
        return x

    def _irfftn(x):
        """ Perform a real-valued inverse FFT. """
        x = ifftn_first_pass(x)
        x = irfftn_second_pass(x)
        return x

    # Creating jitted versions of FFT and IFFT functions for specific GPU sharding
    _rfftn_jit = jax.jit(_rfftn, in_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))),
                         out_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))))
    _irfftn_jit = jax.jit(_irfftn, in_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))),
                          out_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))))
    _fftn_jit = jax.jit(_fftn, in_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))),
                        out_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))))
    _ifftn_jit = jax.jit(_ifftn, in_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))),
                         out_shardings=(NamedSharding(compute_mesh, P(None, "gpus"))))

    @custom_vjp
    def rfftn(x):
        """
        Perform a real-valued forward FFT with custom VJP (vector-Jacobian product) defined.

        Note:
            This function is differentiable and the derivative is defined using VJP.
        """
        return _rfftn_jit(x)

    def rfftn_fwd(x):
        """ Forward pass for custom VJP of real-valued forward FFT """
        return _rfftn_jit(x), x.shape

    def rfftn_bwd(x_shape, g):
        """ Backward pass for custom VJP of real-valued forward FFT """
        # TODO: Weird behaviour when x_shape is not a power of 2
        g = jnp.pad(g, [(0, si - xi) for xi, si in zip(g.shape, x_shape)])
        g = _ifftn_jit(g.conj()).real
        # the previous code is equivalent to jnp.fft.ifftn(g.conj(), s=x_shape).real
        g *= jnp.prod(jnp.array(x_shape))
        return (g,)

    rfftn.defvjp(rfftn_fwd, rfftn_bwd)

    @custom_vjp
    def irfftn(x):
        """
        Perform a real-valued inverse FFT with custom VJP (vector-Jacobian product) defined.

        Note:
            This function is differentiable and the derivative is defined using VJP.
        """
        return _irfftn_jit(x)

    def irfftn_fwd(x):
        """ Forward pass for custom VJP of real-valued inverse FFT """
        return _irfftn_jit(x), x.shape

    def create_mask(n, is_odd, dtype):
        # Create a large enough static mask
        assert n <= max_n, f"max_n provided ({max_n}) is not big enough for array size of {n}."
        mask = jnp.ones(max_n, dtype=dtype)
        mask = mask.at[1:max_n - 1].set(2.0)  # Set the middle values to 2.0
        mask = mask.at[max_n - 1].set(1.0 - is_odd)  # Adjust the last value based on odd/even
        return mask[:n]  # Slice the mask to the desired size dynamically

    def irfftn_bwd(res, g):
        """ Backward pass for custom VJP of real-valued inverse FFT """
        fft_lengths = jnp.array(g.shape)
        # Compute the RFFT of the gradient
        x = _rfftn_jit(g)
        # Apply the scaling factor and mask
        n = x.shape[-1]
        is_odd = fft_lengths[-1] % 2

        # for jax.jit computation, the mask needs to be static. The commented code is more intuitive, but dynamic
        """full = partial(lax.full_like, g, dtype=x.dtype)
        mask = lax.concatenate(
            [full(1.0, shape=(1,)),
             full(2.0, shape=(n - 2 + is_odd,)),
             full(1.0, shape=(1 - is_odd,))],
            dimension=0)"""
        mask = create_mask(n, is_odd, g.dtype)

        scale = 1 / jnp.prod(fft_lengths)
        out = scale * lax.expand_dims(mask, range(x.ndim - 1)) * x
        return (out,)

    irfftn.defvjp(irfftn_fwd, irfftn_bwd)

    return rfftn, irfftn, _fftn_jit, _ifftn_jit


def test_functions(distributed_func_jit: Callable, reference_func: Callable, array: np.ndarray, mesh: Mesh) -> Tuple[
    bool, float, float]:
    """
    Tests the accuracy and computational performance of two functions. It does this by comparing the output of
    a distributed function (that's been Just-in-Time compiled) provided as "distributed_func_jit" and a reference function.
    It reports whether the outputs of both functions are close (within a certain tolerance), the maximum absolute difference
    in their outputs, and the value at which the maximum absolute difference occurs in the reference function output.

    Args:
        distributed_func_jit (Callable): A function that distributes its computations across multiple devices and has
            been Just-In-Time compiled. This is the function being tested.
        reference_func (Callable): A reference function that is used to compare the accuracy and performance of the
            distributed function.
        array (np.ndarray): The input numpy array that both functions will use to compute their outputs.
        mesh (Mesh) : The compute mesh that defines the layout of devices for distributed computation.

    Returns:
        tuple: A tuple containing the following elements:
            - all_close (bool): A boolean indicating if the outputs of both functions are close within a relative tolerance of
                1.e-2 and absolute tolerance of 1.e-4.
            - max_diff (float): The maximum absolute difference in the outputs of both functions.
            - max_diff_value_ref_func (float): The value in the output of the reference function at which the maximum absolute
                difference occurs.

    Note:
        If the maximum absolute difference is non-zero, also prints the maximum relative difference and the corresponding value
        in the output of the reference function.
    """
    print(f"\nTests for function: {reference_func.__name__}.\n")
    array_distributed = distribute_array_on_gpus(array, mesh, P(None, "gpus", None))
    reference_func_jit = jax.jit(reference_func)
    with mesh:
        distributed_func_jit_out = distributed_func_jit(array_distributed)
    reference_func_jit_out = reference_func_jit(array)

    all_close = np.allclose(distributed_func_jit_out, reference_func_jit_out, rtol=1.e-2, atol=1.e-4)
    diff = reference_func_jit_out - distributed_func_jit_out
    max_diff = np.max(np.abs(diff))
    max_diff_index = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    max_diff_value_ref = reference_func_jit_out[max_diff_index]

    print(f"\n{'Output close to reference' if all_close else 'WARNING: Output not close to reference'}")
    print(f"{max_diff = }")
    print(f"{max_diff_value_ref = }")

    if max_diff != 0:
        relative_diff = np.abs(diff) / np.maximum(np.abs(distributed_func_jit_out), np.abs(reference_func_jit_out))
        max_relative_diff = np.max(relative_diff)
        max_relative_diff_index = np.unravel_index(np.argmax(relative_diff), relative_diff.shape)
        max_relative_diff_value_ref = reference_func_jit_out[max_relative_diff_index]

        print(f"{max_relative_diff = }")
        print(f"{max_relative_diff_value_ref = }")


if __name__ == '__main__':
    import os

    num_gpus = jax.device_count()
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("devices:", jax.device_count(), jax.devices())
    print("local_devices:", jax.local_device_count(), jax.local_devices())
    print("process_index", jax.process_index())
    print("total number of GPUs:", num_gpus)

    devices = mesh_utils.create_device_mesh((num_gpus,), devices=jax.devices()[:num_gpus])
    mesh = Mesh(devices, axis_names=("gpus",))

    rfftn_jit, irfftn_jit, fftn_jit, ifftn_jit = create_ffts(mesh)

    jax.config.update("jax_enable_x64", True)

    size = 512
    global_shape = (size, size, size)
    x = np.random.default_rng(2).random(global_shape, dtype=np.float64)

    x_distributed = distribute_array_on_gpus(x, mesh, P(None, "gpus", None))

    with mesh:
        rfftn_out = rfftn_jit(x_distributed)

    test_functions(rfftn_jit, jnp.fft.rfftn, x, mesh)
