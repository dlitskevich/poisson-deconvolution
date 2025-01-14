import numpy as np
from scipy import stats

from optimal_deconvolution.microscopy.experiment import MicroscopyExperiment
from optimal_deconvolution.microscopy.sampler import StdMicroscopySampler
from optimal_deconvolution.wasserstein_distance import wasserstein_distance


class MicroscopyEstimator:
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the MicroscopyEstimator class.

        Parameters:
            experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        self.experiment = experiment
        self.t = experiment.t
        self.data = experiment.data
        self.n_bins = experiment.data.shape

    def error(self, true_mu: np.ndarray, estimated: np.ndarray) -> float:
        """
        Compute the error between the true mu and the estimated mu.

        Parameters:
            true_mu (np.ndarray): The true mu (atoms locations) values.
            estimated (np.ndarray): The estimated mu (atoms locations) values.

        Returns:
            float: The error value.
        """
        return wasserstein_distance(estimated, true_mu)

    def approximate_error(
        self, estimated, scale, samplerCls=StdMicroscopySampler
    ) -> np.ndarray:
        """
        Compute the approximate error.

        Parameters:
            estimated (np.ndarray): The estimated mu (atoms locations) values.
            scale: The scale value.
            samplerCls: The sampler class.

        Returns:
            np.ndarray: The approximate error.
        """
        sampler = samplerCls(estimated, self.data.shape[0], scale, self.t)
        exp = sampler.sample_convolution()

        return self.data / self.t - exp.data


class StdMicroscopyCommon(MicroscopyEstimator):
    def get_xy_probabilities(self, mu: np.ndarray, scale: float) -> tuple:
        """
        Get the x and y probabilities.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.

        Returns:
            tuple: The x and y probabilities.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        x_bins = np.linspace(0, n_x / n, n_x + 1)
        y_bins = np.linspace(0, n_y / n, n_y + 1)
        x_prob = np.diff([stats.norm.cdf(x, mu[:, 0], scale) for x in x_bins], axis=0)
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
        x_dens_diff = np.diff([stats.norm.pdf(x, x_loc, scale) for x in x_bins], axis=0)
        y_dens_diff = np.diff([stats.norm.pdf(x, y_loc, scale) for x in y_bins], axis=0)

        return (x_dens_diff, y_dens_diff)
