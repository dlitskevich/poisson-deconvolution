import numpy as np
from scipy import stats

from optimal_deconvolution.microscopy.sampler import MicroscopySampler


class NormMicroscopySampler(MicroscopySampler):
    """
    Class for sampling microscopy experiments with halved scale for horizontal axis.
    """

    def __init__(
        self, atoms: np.ndarray, n_bins: tuple[int] | int, scale: np.ndarray, t=1e8
    ):
        """
        Initialize the NormMicroscopySampler.

        Parameters:
            atoms (np.ndarray): Array of atom positions.
            n_bins (tuple[int] | int): Number of bins.
            scale (np.ndarray): Covariance matrix for the convolution.
            t (float, optional): Illumination time. Defaults to 1e8.
        """
        super().__init__(atoms, n_bins, scale, t)

        # slow to integrate
        self.convolution = lambda x: np.mean(
            [stats.multivariate_normal.pdf(x, loc, scale) for loc in atoms]
        )

    def convolution_measure_bin(self, bin_id: int) -> float:
        """
        Compute the convolution measure for a given bin.

        Parameters:
            bin_id (int): The bin ID.

        Returns:
            float: The convolution measure for the bin.
        """
        bin = self.bins_loc[bin_id]
        scale = self.scale
        h = self.h
        prob = 0
        for atom in self.atoms:
            atom_prob, _ = stats.mvn.mvnun(bin - h, bin + h, atom, scale)
            prob += atom_prob

        return prob / self.atoms.shape[0] if prob > 0 else 0
