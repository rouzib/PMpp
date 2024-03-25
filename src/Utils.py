"""
Date: 03/21/2024

This code defines a class named 'SimInfo' that is used for handling simulation information. The class contains an initializer
that sets up simulation parameters such as seed, index, number of meshes and the base path. It also specifies the
downsampling method for the simulation.

There are four methods in this class that generate paths for different types of data associated with a simulation. These
methods returns paths for positional data ('get_pos_path'), velocity data ('get_vel_path'), cosmic data ('get_cosmo_path'),
and redshift data ('get_redshift_path'). The paths are based on the provided base path, and include identification details
like index, number of meshes, downsampling method, and seed. This class and its methods provide a structured way to organize
simulation information and the associated data paths.
"""
from pathlib import Path

from src.CamelsUtils import DownsamplingMethod


class SimInfo:
    """
    Initializes a new instance of the SimInfo class.

    :param idx: The index of the simulation.
    :type idx: int
    :param seed: The seed used for random number generation.
    :type seed: int
    :param n_mesh: The number of mesh points.
    :type n_mesh: int
    :param downsampling_method: The downsampling method. Default is "mesh".
    :type downsampling_method: str
    :param base_path: The base path for the simulation files. Default is "CamelsSims".
    :type base_path: str | Path
    """

    def __init__(self, idx: int, seed: int, n_mesh: int, downsampling_method: str = DownsamplingMethod.MESH,
                 base_path: str | Path = "CamelsSims"):
        """
        Initializes an instance of the class.

        :param idx: The index of the instance.
        :param seed: The seed value used for randomization.
        :param n_mesh: The size of the mesh.
        :param downsampling_method: The downsampling method to be used. Default is "mesh".
        :param base_path: The base path to the simulation directory. Default is "CamelsSims".
        """
        self.seed = seed
        self.idx = idx
        self.n_mesh = n_mesh
        self.downsampling_method = downsampling_method
        self.base_path = base_path

    def format_downsampling_method(self):
        """
        Format the downsampling method.

        :return: The downsampling method, with a leading underscore if it is not empty, otherwise an empty string.
        :rtype: str
        """
        return ("_" + self.downsampling_method) if self.downsampling_method not in ["",
                                                                                    DownsamplingMethod.RANDOM] else ""

    def get_pos_path(self):
        """
        Returns the file path for the position data based on the specified parameters.

        :return: The file path for the position data.
        :rtype: str
        """
        return f"{self.base_path}/LH_{self.idx}_pos_{self.n_mesh}{self.format_downsampling_method()}_{self.seed}.npy"

    def get_vel_path(self):
        """
        Returns the velocity path based on the given parameters.

        :return: A string representing the velocity path.
        :rtype: str
        """
        return f"{self.base_path}/LH_{self.idx}_vel_{self.n_mesh}{self.format_downsampling_method()}_{self.seed}.npy"

    def get_cosmo_path(self):
        """
        Returns the path to the Cosmo file for the given instance.

        :return: The path to the Cosmo file.
        :rtype: str
        """
        return f"{self.base_path}/LH_{self.idx}_cosmo.npy"

    def get_redshift_path(self):
        """Returns the path to the Redshift file based on the provided parameters.

        :return: The path to the Redshift file.
        :rtype: str
        """
        return f"{self.base_path}/LH_{self.idx}_z_{self.n_mesh}.npy"

    def get_k_path(self):
        """
        Get the file path for the k_LH file for a given index and number of meshes.

        :return: The file path for the k_LH file.
        :rtype: str
        """
        return f"{self.base_path}/K_LH_{self.idx}_{self.n_mesh}{self.format_downsampling_method()}_{self.seed}.npy"

    def get_pk_path(self):
        """
        Return the path for the PK file.

        :return: The path for the PK file.
        :rtype: str
        """
        return f"{self.base_path}/PK_LH_{self.idx}_{self.n_mesh}{self.format_downsampling_method()}_{self.seed}.npy"

    def __str__(self):
        """
        Returns a string representation of the object.

        :return: (str) String representation of the object.
        """
        return (f"SimInfo(idx={self.idx}, seed={self.seed}, n_mesh={self.n_mesh}, "
                f"downsampling_method={self.downsampling_method}, base_path={self.base_path})")

    def __repr__(self):
        """
        Return a string representation of the object.

        :return: A string representation of the object.
        """
        return self.__str__()
