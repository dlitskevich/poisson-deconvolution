import numpy as np
from scipy import stats

from optimal_deconvolution.microscopy.microscopy_estimator import StdMicroscopyCommon


class HalfStdDensityMicroscopyCommon(StdMicroscopyCommon):

    def get_xy_probabilities(self, mu: np.ndarray, scale: float):
        """
        Computes the x and y probabilities.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The x and y probabilities.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        x_bins = np.linspace(0, n_x / n, n_x + 1)
        y_bins = np.linspace(0, n_y / n, n_y + 1)
        x_prob = np.diff(
            [stats.norm.cdf(x, mu[:, 0], scale / 2) for x in x_bins], axis=0
        )
        y_prob = np.diff([stats.norm.cdf(y, mu[:, 1], scale) for y in y_bins], axis=0)

        return (x_prob, y_prob)

    def get_xy_density_diff(self, mu: np.ndarray, scale: float):
        """
        Computes the x and y density differences.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The x and y density differences.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        x_bins = np.linspace(0, n_x / n, n_x + 1)
        y_bins = np.linspace(0, n_y / n, n_y + 1)
        x_loc = mu[:, 0]
        y_loc = mu[:, 1]
        x_dens_diff = np.diff(
            [stats.norm.pdf(x, x_loc, scale / 2) for x in x_bins], axis=0
        )
        y_dens_diff = np.diff([stats.norm.pdf(x, y_loc, scale) for x in y_bins], axis=0)

        return (x_dens_diff, y_dens_diff)
