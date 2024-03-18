"""
Date: 03/18/2024

This code provides functionality for reading files from the CAMELS simulation. The primary functions
allow for the reading of snapshot files, each providing the position, velocity and redshift for a
given set of data. The data can be downsampled upon reading to reduce the size and complexity of
the returned data sets. The main functions included:

- 'read_camels' function reads the dark matter properties from a given snapshot file, with an
option to downsample the data.

- The 'read_camels_snapshots_lh' function iterates over a list of snapshot files, reading and
returning the position, velocity, and redshift for each snapshot but this time allowing to use
large-scale simulations.
"""

import os
import traceback
from typing import List

import jax.numpy as jnp
import jax.profiler
from tqdm import tqdm

from camels_utils import read_camels

# disable memory preallocation to conserve memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# get the available cpu and gpu devices
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")


def read_camels_snapshots_lh(snapshot_list, lh_index: int = 0, downsampling_factor=32, data_dir="", seed=0):
    """
    Read CAMELS snapshots for a specific LH index.

    :param snapshot_list: A list of snapshot names.
    :param lh_index: The index of the Luminosity Halo to read. Default is 0.
    :param downsampling_factor: The downsampling factor for the data. Default is 32.
    :param data_dir: The directory where the data is stored. Default is an empty string.
    :param seed: The seed value for random number generation. Default is 0.
    :return: A tuple containing three NumPy arrays: pos, vel, and redshift.
    """
    pos, vel, redshift = [], [], []

    # iterate over each snapshot
    for s in tqdm(snapshot_list):
        # read the snapshot data
        p, v, z, _, _ = read_camels(
            snapshot=s, cv_index=lh_index, downsampling_factor=downsampling_factor, data_dir=data_dir, cv_or_lh=True,
            seed=seed)
        # append positions, velocities, and redshifts to respective lists
        pos.append(p)
        vel.append(v)
        redshift.append(z)

    # returns the position, velocity, and redshift numpy arrays
    return jnp.array(pos, copy=False), jnp.array(vel, copy=False), jnp.array(redshift, copy=False)


def read_camels_lh_set(lh_index_list: List[int] = [0, 1], snapshot_list=range(34), downsampling_factor: int = 32,
                       data_dir="", save_dir="/home/rouzib/scratch/", seed=0):
    """
    Read a set of CAMELS snapshot data for multiple LH indices.

    :param lh_index_list: A list of LH indices to read. Default is [0,1].
    :param snapshot_list: A list of snapshot names. Default is range(34).
    :param downsampling_factor: The downsampling factor for the data. Default is 32.
    :param data_dir: The directory where the data is stored. Default is an empty string.
    :param save_dir: The directory in which to save the data. Default is "/home/rouzib/scratch/"
    :param seed: The seed value for random number generation. Default is 0.
    :return: A tuple containing four members: a position array, a velocity array, the redshift, and a cosmology factor.
    """

    pos, vel, cosmo = [], [], []

    # iterate over each LH index
    for lh_index in lh_index_list:
        try:
            # construct the file path for the cosmoastro parameters of the current LH index
            file_path = str(f"{data_dir}/LH_{lh_index}/CosmoAstro_params.txt")

            with open(file_path, 'r') as file:
                line = file.readline()
                values = line.split(' ')
            # append cosmology factors to the list
            cosmo.append([float(values[0]), float(values[1])])

            # read and save the snapshot data for this LH index
            p, v, redshift = read_camels_snapshots_lh(snapshot_list=snapshot_list, lh_index=lh_index,
                                                      downsampling_factor=downsampling_factor, data_dir=data_dir,
                                                      seed=seed)
            pos.append(p)
            vel.append(v)

            # save the read values to numpy files in the save directory
            jnp.save(f"{save_dir}/LH_{lh_index}_pos_{downsampling_factor}_{seed}.npy", p)
            jnp.save(f"{save_dir}/LH_{lh_index}_vel_{downsampling_factor}_{seed}.npy", v)
            jnp.save(f"{save_dir}/LH_{lh_index}_z_{downsampling_factor}.npy", redshift)
            jnp.save(f"{save_dir}/LH_{lh_index}_cosmo.npy", jnp.array([float(values[0]), float(values[1])]))

        except Exception as e:
            # print traceback and message if an error occurs, then carry on with next iteration
            traceback.print_exc()
            print(f"Passed LH_{lh_index} because of {e}")
            pass

    return jnp.array(pos), jnp.array(vel), redshift, cosmo


if __name__ == '__main__':
    n_mesh = 64
    box_size = [25.0, 25.0, 25.0]
    sim_ids = [0]

    # read and save the CAMELS snapshot data for the test indices
    read_camels_lh_set(lh_index_list=list(sim_ids), downsampling_factor=n_mesh, data_dir="../Camels",
                       save_dir="../CamelsSims", seed=0)
