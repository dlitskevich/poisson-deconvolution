import numpy as np
from scipy import stats

from optimal_deconvolution.microscopy.microscopy_estimator import StdMicroscopyCommon


class StdDensityMicroscopyCommon(StdMicroscopyCommon):
    def get_xy_probabilities(self, mu: np.ndarray, scale: float) -> tuple:
        """
        Get the x and y probabilities approximated by density multiplied by the size of a bin.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.

        Returns:
            tuple: The x and y probabilities.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        x_loc = np.linspace(0, n_x / n, n_x)
        y_loc = np.linspace(0, n_y / n, n_y)
        x_prob = np.array([stats.norm.pdf(x, mu[:, 0], scale) / n_x for x in x_loc])
        y_prob = np.array([stats.norm.pdf(y, mu[:, 1], scale) / n_y for y in y_loc])

        # trapezoidal rule
        # x_loc = np.linspace(0, n_x / n, n_x)
        # y_loc = np.linspace(0, n_y / n, n_y)
        # x_prob = np.array([stats.norm.pdf(x, mu[:, 0], scale) for x in x_loc])
        # y_prob = np.array([stats.norm.pdf(y, mu[:, 1], scale) for y in y_loc])
        # x_prob = (x_prob[1:, :] + x_prob[:-1, :]) / 2 / n_x
        # y_prob = (y_prob[1:, :] + y_prob[:-1, :]) / 2 / n_y

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
        x_bins = np.linspace(0, n_x / n, n_x)
        y_bins = np.linspace(0, n_y / n, n_y)

        x_loc = mu[:, 0]
        y_loc = mu[:, 1]
        x_dens_diff = [
            (x_loc - x) / scale**2 / n_x * stats.norm.pdf(x, x_loc, scale)
            for x in x_bins
        ]
        y_dens_diff = [
            (y_loc - x) / scale**2 / n_y * stats.norm.pdf(x, y_loc, scale)
            for x in y_bins
        ]

        return (x_dens_diff, y_dens_diff)
