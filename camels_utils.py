"""
Date: 03/10/2024

This Python script makes use of the Jax, NumPy, and ReadGadget libraries to assist in the reading and processing
of CAMELS cosmological data snapshots. It provides utility functions that allow users to interact with the CAMELS
data at various levels of granularity, from individual snapshot reading to the extraction of entire cross-validation
sets.

The 'read_camels' function processes individual snapshots, allowing for particle position and velocity extraction
and optional downsampled data generation.

The 'read_camels_snapshots' function further expands on this by allowing for multiple snapshot files to be processed
and pooled together.

The 'read_camels_cv_set' function builds on the previous functions and allows for the easy generation of
cross-validation sets, which are commonplace in machine learning and statistical analysis contexts.

Finally, the 'normalize_by_mesh' function is provided as a utility for normalizing particle positions and velocities
according to a given mesh size, useful for certain types of cosmological simulations.

All functions return processed data in the form of JAX arrays.
"""
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import readgadget
from tqdm import tqdm

# The default directory path for the CAMELS data set
DEFAULT_CAMELS_DATA_DIR = Path("../../data/CAMELS/Sims/IllustrisTNG_DM")


def read_camels(snapshot: int, cv_index: int = 0, downsampling_factor: int = 32,
                data_dir: Path = DEFAULT_CAMELS_DATA_DIR, cv_or_lh: bool = False, seed: int = 0) \
        -> Tuple[jnp.ndarray, jnp.ndarray, float, float, float]:
    """
    Reads and processes CAMELS data for a given snapshot.

    :param snapshot: The snapshot number.
    :param cv_index: The cross-validation index (default is 0).
    :param downsampling_factor: The downsampling factor for the data (default is 32).
    :param data_dir: The directory where the CAMELS data is stored (default is DEFAULT_CAMELS_DATA_DIR).
    :param cv_or_lh: Whether to use cross-validation (CV) or large halo (LH) data (default is False).
    :param seed: Seed used for downsampling (default is 0)
    :return: Tuple containing the position array, velocity array, redshift value, Omega_m value, and Omega_l value.
        The position array contains positions in Mpc/h units, velocity array contains velocities in km/s,
        and the redshift, Omega_m, and Omega_l values correspond to the header information from the snapshot file.
    """
    # Determine the snapshot filename based on cv_index and type (CV or LH)
    if not cv_or_lh:
        snapshot_filename = str(f"{data_dir}/CV_{cv_index}/snap_{str(snapshot).zfill(3)}.hdf5")
    else:
        snapshot_filename = str(f"{data_dir}/LH_{cv_index}/snap_{str(snapshot).zfill(3)}.hdf5")

    # Read the header file
    header = readgadget.header(snapshot_filename)
    BoxSize = header.boxsize / 1e3  # Convert box size from kiloparsecs to megaparsecs
    Omega_m = header.omega_m  # Omega_m value
    Omega_l = header.omega_l  # Omega_lambda value
    redshift = header.redshift  # Redshift value at the snapshot

    # Load dark matter properties (particle type 1)
    ptype = [1]
    ids = np.argsort(readgadget.read_block(snapshot_filename, "ID  ", ptype) - 1)
    # Get positions in Mpc/h units and velocities in km/s
    pos = readgadget.read_block(snapshot_filename, "POS ", ptype)[ids] / 1e3
    vel = readgadget.read_block(snapshot_filename, "VEL ", ptype)[ids]
    pos = jnp.array(pos)
    # Recalculate velocities considering cosmological redshift
    vel = jnp.array(vel / 100 * (1.0 / (1 + redshift)))

    # Do downsampling if downsampling_factor provided
    if downsampling_factor is not None:
        downsampling_factor = len(pos) // downsampling_factor ** 3
        # Generate random indices for downsampling
        key = jax.random.PRNGKey(seed)
        permuted_indices = jax.random.permutation(key, len(pos))
        selected_indices = permuted_indices[: len(pos) // downsampling_factor]
        pos = jnp.take(pos, selected_indices, axis=0)
        vel = jnp.take(vel, selected_indices, axis=0)
    return pos, vel, redshift, Omega_m, Omega_l


def read_camels_snapshots(snapshot_list: List[str], cv_index: int = 0, downsampling_factor: int = 500,
                          data_dir: str = DEFAULT_CAMELS_DATA_DIR) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Reads snapshots of camels data and returns the positions, velocities, and redshifts.

    :param snapshot_list: A list of snapshot names to be loaded.
    :param cv_index: An optional integer representing the cross-validation index. Default value is 0.
    :param downsampling_factor: An optional integer representing the downsampling factor. Default value is 500.
    :param data_dir: An optional string representing the directory where the data is located. Default value is the
     constant DEFAULT_CAMELS_DATA_DIR.
    :return: A tuple containing JAX numpy arrays of positions, velocities, and redshifts.

    """
    # Placeholder lists for storing respective data for all the snapshots
    pos, vel, redshift = [], [], []
    # Load all the snapshots
    for s in tqdm(snapshot_list):
        p, v, z, _, _ = read_camels(snapshot=s, cv_index=cv_index, downsampling_factor=downsampling_factor,
                                    data_dir=data_dir)
        pos.append(p)
        vel.append(v)
        redshift.append(z)
    # Convert list to JAX numpy array
    return jnp.array(pos), jnp.array(vel), jnp.array(redshift)


def read_camels_cv_set(cv_index_list: Optional[List[int]] = None, snapshot_list: List[int] = range(34),
                       downsampling_factor: int = 500, data_dir=DEFAULT_CAMELS_DATA_DIR) \
        -> Tuple[jnp.ndarray, jnp.ndarray, List[float]]:
    """
    Read and process CamelS CV set.

    :param cv_index_list: List of cross-validation set indices. Default is [0, 1] if None.
    :param snapshot_list: List of snapshot indices. Default is range(34).
    :param downsampling_factor: Downsampling factor for snapshots. Default is 500.
    :param data_dir: Directory path for CamelS data. Default is DEFAULT_CAMELS_DATA_DIR.
    :return: A tuple containing numpy arrays of positions, velocities, and redshifts.
    """
    # mutable argument
    if cv_index_list is None:
        cv_index_list = [0, 1]
    pos, vel = [], []
    # Loop over all cross-validation sets
    for cv_index in cv_index_list:
        p, v, redshift = read_camels_snapshots(snapshot_list=snapshot_list, cv_index=cv_index,
                                               downsampling_factor=downsampling_factor, data_dir=data_dir, )
        pos.append(p)
        vel.append(v)
    # Return positions and velocities for all CV sets and redshifts
    return jnp.array(pos), jnp.array(vel), redshift


def normalize_by_mesh(positions: jnp.ndarray, velocities: jnp.ndarray, box_size: float, n_mesh: int) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalize positions and velocities according to the mesh size.

    :param positions: The positions of particles.
    :param velocities: The velocities of particles.
    :param box_size: The size of the simulation box.
    :param n_mesh: The size of the mesh.

    :return: The normalized positions and velocities.
    """
    # Normalize positions and velocities according to the mesh size
    positions = positions / box_size * n_mesh
    velocities = velocities / box_size * n_mesh
    return positions, velocities
