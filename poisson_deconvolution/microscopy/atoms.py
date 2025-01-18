from enum import Enum
from matplotlib import pyplot as plt
import numpy as np

from poisson_deconvolution.microscopy.points_initializers import circle_points


def _grid_gen(dx: int, dy: int, num: int) -> np.ndarray:
    """
    Generate a grid of points.

    Parameters:
        dx (int): The step size in the x-direction.
        dy (int): The step size in the y-direction.
        num (int): The number of points in each direction.

    Returns:
        np.ndarray: The generated grid of points.
    """
    a = np.array([[i, k] for i in range(0, num, dx) for k in range(0, num, dy)])
    a = a / (num - 1) * 0.6 + 0.2

    return a


class AtomsType(Enum):
    Grid = "grid"
    Line1 = "1 line"
    Line2 = "2 lines"
    UShape = "u-shape"
    Corners = "corners"
    Clusters = "clusters"
    Clusters6 = "clusters (6 points)"


BASE_ATOMS = [
    AtomsType.Grid,
    AtomsType.Line1,
    AtomsType.Line2,
    AtomsType.UShape,
]


class AtomsData:
    @staticmethod
    def grid(num: int = 4) -> "AtomsData":
        """
        Generate a grid of atoms.

        Parameters:
            num (int): The number of points in each direction.

        Returns:
            AtomsData: An instance of AtomsData representing the generated grid of atoms.
        """
        return AtomsData(_grid_gen(1, 1, num), f"${num}^2$ grid", AtomsType.Grid, num)

    @staticmethod
    def line1(num: int = 4) -> "AtomsData":
        """
        Generate a line of atoms.

        Parameters:
            num (int): The number of points in the line.

        Returns:
            AtomsData: An instance of AtomsData representing the generated line of atoms.
        """
        return AtomsData(
            _grid_gen(1, num, num), f"1 line ({num} points)", AtomsType.Line1, num
        )

    @staticmethod
    def lines2(num: int = 4) -> "AtomsData":
        """
        Generate two lines of atoms.

        Parameters:
            num (int): The number of points in each line.

        Returns:
            AtomsData: An instance of AtomsData representing the generated two lines of atoms.
        """
        return AtomsData(
            _grid_gen(1, num - 1, num), f"2 lines ({num} points)", AtomsType.Line2, num
        )

    @staticmethod
    def corners(num: int = 2) -> "AtomsData":
        """
        Generate corner atoms.

        Parameters:
            num (int): The number of corner atoms.

        Returns:
            AtomsData: An instance of AtomsData representing the generated corner atoms.
        """
        return AtomsData(_grid_gen(1, 1, 2), "corners", AtomsType.Corners, 2)

    @staticmethod
    def uShape(num: int = 4) -> "AtomsData":
        """
        Generate a U-shaped pattern of atoms.

        Parameters:
            num (int): The number of points in each direction.

        Returns:
            AtomsData: An instance of AtomsData representing the generated U-shaped pattern of atoms.
        """
        lower = _grid_gen(1, num, num)
        atoms = np.concatenate([_grid_gen(num - 1, 1, num), lower[1:-1]])
        return AtomsData(atoms, f"u-shape ({num} points)", AtomsType.UShape, num)

    @staticmethod
    def clusters6(cluster_points: int, rad=0.05) -> "AtomsData":
        """
        Generate a 2 clusters pattern of 6 atoms.

        Parameters:
            cluster_points (int): The number of points in second cluster.
            rad (float): The radius of the clusters.

        Returns:
            AtomsData: An instance of AtomsData representing the generated 2 clusters pattern of atoms.
        """
        num = 6
        atoms = np.concatenate(
            [
                circle_points(num - cluster_points) * rad + 0.3,
                circle_points(cluster_points) * rad + 0.7,
            ]
        )
        return AtomsData(
            atoms,
            f"clusters ({num-cluster_points}:{cluster_points} points) radius:{rad}",
            AtomsType.Clusters6,
            cluster_points,
        )

    @staticmethod
    def from_json(data: dict) -> "AtomsData":
        """
        Create an instance of AtomsData from a JSON-like dictionary.

        Parameters:
            data (dict): The dictionary containing the atoms data.

        Returns:
            AtomsData: An instance of AtomsData created from the provided dictionary.
        """
        data["atoms"] = np.array(data["atoms"])
        data["type"] = AtomsType(data["type"])
        return AtomsData(**data)

    @staticmethod
    def from_type(type: AtomsType, num_points: int = 4) -> "AtomsData":
        """
        Create an instance of AtomsData based on the specified atoms type.

        Parameters:
            type (AtomsType): The type of atoms to generate.
            num_points (int): The number of points to generate.

        Returns:
            AtomsData: An instance of AtomsData representing the generated atoms.
        """
        return _atoms_data_from_type[type](num_points)

    def __init__(
        self,
        atoms: np.ndarray,
        name: str = "custom",
        type: AtomsType = None,
        n_points: int = None,
    ):
        """
        Initialize an instance of AtomsData.

        Parameters:
            atoms (np.ndarray): The array of atoms.
            name (str): The name of the atoms data.
            type (AtomsType): The type of atoms.
            n_points (int): The number of points in the atoms data.
        """
        self.atoms = atoms
        self.name = name
        self.type = type
        self.n_points = n_points

    def to_json(self) -> dict:
        """
        Convert the AtomsData instance to a JSON-like dictionary.

        Returns:
            dict: The JSON-like dictionary representing the AtomsData instance.
        """
        return {
            "atoms": self.atoms.tolist(),
            "name": self.name,
            "type": self.type.value,
            "n_points": self.n_points,
        }

    def plot(self) -> None:
        """
        Plot the atoms data.
        """
        atoms = self.atoms
        plt.scatter(atoms[:, 0], atoms[:, 1])
        plt.title(self.name)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    @staticmethod
    def plot_all(
        num_points: int = 4, figsize: int = 2, types: list[AtomsType] = BASE_ATOMS
    ) -> None:
        """
        Plot all types of atoms.

        Parameters:
            num_points (int): The number of points in each direction.
            figsize (int): The size of the figure.
        """
        num = len(types)
        plt.figure(figsize=(num * figsize, figsize))
        for i, atoms_type in enumerate(types):
            plt.subplot(1, num, i + 1)
            atoms_data = AtomsData.from_type(atoms_type, num_points)
            atoms = atoms_data.atoms
            plt.scatter(atoms[:, 0], atoms[:, 1])
            plt.title(atoms_data.name)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.yticks(np.arange(0, 1.2, 0.2) if i == 0 else [])
            plt.xticks(np.arange(0, 1.2, 0.2))

        plt.tight_layout()


_atoms_data_from_type = {
    AtomsType.Grid: AtomsData.grid,
    AtomsType.Line1: AtomsData.line1,
    AtomsType.Line2: AtomsData.lines2,
    AtomsType.Corners: AtomsData.corners,
    AtomsType.UShape: AtomsData.uShape,
    AtomsType.Clusters6: lambda n: AtomsData.clusters6(n, 0.05),
}
