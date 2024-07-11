"""
Author: Nicolas Payot
Date: 10/07/2024

The following script handles the extraction, manipulation, and storing of data generated
from simulations related to cosmology. It defines the 'Header' class that stores important
data from the simulations such as the current simulation time, redshift value,
box size of the simulation, file number of the simulation, density parameters,
particle counts, and other properties.

The script contains functions 'write_hdf5', 'copy_camels_hdf5', and others meant
for writing this data to HDF5 files, copying the simulation data to a new file and
making the requisite modifications to match the simulation's intrinsic format.

The main function is used for reading the header file, loading dark matter
properties, recalculating velocities considering cosmological redshift for a selected simulation,
then copying the data to a new file with the same properties and finally verifying the copied data.
It uses specific functions from the 'readgadget' module. After validation, the created file is removed.
"""
import numpy as np
import glob
import os
from tqdm import tqdm
import hdf5plugin
import tables
import h5py
import readgadget


class Header:
    """
    Header class

    This class represents the header information of a simulation.

    Attributes:
        time (float): The current simulation time.
        redshift (float): The redshift value of the simulation.
        boxsize (float): The size of the simulation box.
        filenum (int): The file number of the simulation.
        omega_m (float): The value of the matter density parameter of the simulation.
        omega_l (float): The value of the dark energy density parameter of the simulation.
        hubble (float): The value of the Hubble constant of the simulation.
        massarr (list): A list of mass values for different particle types in the simulation.
        npart (list): A list of particle counts for different particle types in the simulation.
        nall (int): The total number of particles in the simulation.

    """

    def __init__(self, time, redshift, boxsize, filenum, omega_m, omega_l, hubble, massarr, npart, nall):
        self.time = time
        self.redshift = redshift
        self.boxsize = boxsize
        self.filenum = filenum
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.hubble = hubble
        self.massarr = massarr
        self.npart = npart
        self.nall = nall


def write_hdf5(filename, header_data, ptype, array_pos, array_vel, array_id, compression_json):
    """
    :param filename: The name of the HDF5 file to be created. (str)
    :param header_data: A dictionary containing the header data to be assigned as attributes to the 'Header' group. (dict)
    :param ptype: The particle type. (int)
    :param array_pos: An array of coordinates for the particle type. (array)
    :param array_vel: An array of velocities for the particle type. (array)
    :param array_id: An array of particle IDs for the particle type. (array)
    :param compression_json: The json giving the compression information. (str)
    :return: None

    This method creates a new HDF5 file, creates a 'Header' group, and assigns attributes to the 'Header' group from
    the provided header_data dictionary. It then creates groups for various blocks based on the particle type and
    creates datasets for the coordinates, velocities, and particle IDs if they do not already exist. Finally, it closes
    the HDF5 file.
    """
    # Create a new HDF5 file
    file = h5py.File(filename, 'w')

    # Create a group named 'Header' (and 'Cosmology'/'Parameters' as needed)
    hdr_group = file.create_group("Header")

    # Assign attributes to the 'Header' group
    for attr, value in header_data.items():
        hdr_group.attrs[attr] = value

    file.create_group("CompressionInfo").attrs["json"] = compression_json

    # Prefix for the particle type
    prefix = 'PartType%d/' % ptype

    # Create groups for various blocks
    if prefix + "Coordinates" not in file:
        file.create_dataset(prefix + "Coordinates", data=array_pos)  # replace array_pos with the actual data

    if prefix + "ParticleIDs" not in file:
        file.create_dataset(prefix + "ParticleIDs", data=array_id)  # replace array_id with the actual data

    if prefix + "Velocities" not in file:
        file.create_dataset(prefix + "Velocities", data=array_vel)  # replace array_vel with the actual data

    # Always close your HDF5 files when you're done
    file.close()


def copy_camels_hdf5(filename, file_ref, pos, vel, ids):
    """
    Copy data from the specified HDF5 file to a new file.

    :param filename: The name of the output HDF5 file.
    :param file_ref: The reference to the input HDF5 file.
    :param pos: The position data.
    :param vel: The velocity data.
    :param ids: The ID data.
    :return: The name of the output HDF5 file.
    """
    with h5py.File(file_ref, "r") as f:
        header_attrs = f["Header"].attrs
        header_dict = {attr: header_attrs[attr] for attr in header_attrs}
        compression_info = f["CompressionInfo"].attrs["json"]

    resorted_ids = np.argsort(ids)
    write_hdf5(filename, header_dict, ptype=1,
               array_pos=pos[resorted_ids] * 1e3,
               array_vel=vel[resorted_ids] * 100 * (1.0 + header_dict["Redshift"]) / np.sqrt(header_dict["Time"]),
               array_id=resorted_ids + 1,
               compression_json=compression_info)
    return filename


if __name__ == '__main__':
    id = 10

    snapshot_filename = f"CAMELS/LH_{id}/snapshot_014.hdf5"

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
    pos = np.array(pos)
    # Recalculate velocities considering cosmological redshift
    vel = np.array(vel / 100 * (1.0 / (1 + redshift)))

    mass = (header.massarr[ptype] * 1e10)[0]

    snapshot_filename = "test.hdf5"
    # Copy the data to a new file
    copy_camels_hdf5(snapshot_filename, header=header, pos=pos, vel=vel, ids=ids)

    # Read the header file
    new_header = readgadget.header(snapshot_filename)
    new_BoxSize = new_header.boxsize / 1e3  # Convert box size from kiloparsecs to megaparsecs
    new_Omega_m = new_header.omega_m  # Omega_m value
    new_Omega_l = new_header.omega_l  # Omega_lambda value
    new_redshift = new_header.redshift  # Redshift value at the snapshot

    # Load dark matter properties (particle type 1)
    new_ptype = [1]
    new_ids = np.argsort(readgadget.read_block(snapshot_filename, "ID  ", new_ptype) - 1)
    # Get positions in Mpc/h units and velocities in km/s
    new_pos = readgadget.read_block(snapshot_filename, "POS ", new_ptype)[new_ids] / 1e3
    new_vel = readgadget.read_block(snapshot_filename, "VEL ", new_ptype)[new_ids]
    new_pos = np.array(new_pos)
    # Recalculate velocities considering cosmological redshift
    new_vel = np.array(new_vel / 100 * (1.0 / (1 + new_redshift)))

    new_mass = (new_header.massarr[new_ptype] * 1e10)[0]

    print(np.allclose(pos, new_pos))
    print(np.allclose(vel, new_vel))
    print(np.allclose(ids, new_ids))

    print(f"Removing file: {snapshot_filename}")
    os.remove(snapshot_filename)
