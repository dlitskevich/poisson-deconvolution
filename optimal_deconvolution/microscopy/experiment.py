from matplotlib import pyplot as plt
import numpy as np


class MicroscopyExperiment:
    @staticmethod
    def from_data(data: np.ndarray):
        return MicroscopyExperiment(data, float(data.sum()))

    def __init__(
        self,
        data: np.ndarray,
        t: float,
        atoms: np.ndarray = None,
        bins_loc: np.ndarray = None,
    ):
        """
        Represents a microscopy experiment.

        Parameters:
            data (np.ndarray): The experimental data.
            t (float): The illumination time.
            atoms (np.ndarray, optional): The true atoms in the experiment. Defaults to None.
            bins_loc (np.ndarray, optional): The locations of the bins. Defaults to None.
        """
        self.data = data
        self.atoms = atoms
        self.t = t
        self.n_bins = data.shape
        self.bins_loc = bins_loc
        if not bins_loc:
            self.bins_loc = generate_bins_loc(self.n_bins)

    def plot_data(self, cmap=None, **kwargs):
        """
        Plots the experimental data.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        return plt.imshow(
            self.data.T,
            origin="lower",
            extent=[0, n_x / n, 0, n_y / n],
            vmin=0,
            cmap=cmap,
            **kwargs
        )

    def plot(self, c: str = "black"):
        """
        Plots the experimental data and atoms.

        Parameters:
            c (str, optional): The color of the atoms. Defaults to "black".
        """
        self.plot_data()
        if self.atoms is not None:
            plt.scatter(self.atoms[:, 0], self.atoms[:, 1], marker=".", c=c, s=5)


def generate_bins_loc(n_bins: tuple[int] | int) -> np.ndarray:
    """
    Generates the locations of the n_bins**2 bins in the complex unit square [0,1+1j].

    Parameters:
        n_bins (tuple[int] | int): Number of bins.

    Returns:
        np.ndarray: The locations of the bins.
    """
    n_x, n_y = (n_bins,) * 2 if isinstance(n_bins, int) else n_bins
    bins = np.array([[i, j] for i in range(n_x) for j in range(n_y)])

    return (bins + 0.5) / max(n_x, n_y)
