from matplotlib import pyplot as plt
import numpy as np

from optimal_deconvolution.wasserstein_distance import wasserstein_distance
from optimal_deconvolution.microscopy import MicroscopyExperiment
from optimal_deconvolution.moment_estimator import (
    StdComplexNormMomentApproximator,
    MeasureMomentApproximator,
)


class StdMicroscopyMomentEstimator:
    """
    Class for estimating moments and atoms in microscopy experiments.
    """

    atoms = None

    def __init__(self, experiment: MicroscopyExperiment, max_atoms: int, scale=None):
        """
        Initialize the StdMicroscopyMomentEstimator.

        Parameters:
        - experiment (MicroscopyExperiment): The microscopy experiment object.
        - max_atoms (int): The maximum number of atoms to consider.
        - scale (float): The scale of Gaussian distribution (kernel).
        """
        self.experiment = experiment
        self.data = experiment.data.ravel()
        self.bins_loc = experiment.bins_loc.view(complex).ravel()
        self.t = experiment.t
        self.max_atoms = max_atoms
        self.moment_estimator = StdComplexNormMomentApproximator(max_atoms)
        self.atoms_estimator = MeasureMomentApproximator()

    def estimate(self, num_atoms: int, shift=0):
        """
        Estimate the atoms in the microscopy experiment.

        Parameters:
        - num_atoms (int): The number of atoms to estimate.
        - shift (float): The shifting factor for the atoms (default: 0).

        Returns:
        np.ndarray: The estimated atoms.
        """
        emp_moments = self.empirical_moments(num_atoms, shift)
        est_moments = self.moment_estimator.approximate(emp_moments)
        # the 0-th moment should be omitted
        # small shift is required to avoid problems with casting to float array (consider [0, 0, 0])
        atoms = self.atoms_estimator.approximate_atoms(est_moments[1:]) + 1e-32j
        return (atoms - shift).view(float).reshape(-1, 2)

    def empirical_moments(self, max_order: int, shift=0):
        """
        Calculate the empirical moments of the microscopy experiment.

        Parameters:
        - max_order (int): The maximum order of moments to calculate.
        - shift (float): The shifting factor for the moments (default: 0).

        Returns:
        np.ndarray: The empirical moments.
        """
        scaled_bins_loc = self.bins_loc + shift
        # data = self.data / self.t # (worse results)
        mass = np.sum(self.data)
        data = self.data / mass if mass > 0 else self.data

        return np.array(
            [np.dot(scaled_bins_loc**k, data) for k in range(max_order + 1)]
        )

    def error(self, true_atoms: np.ndarray, num_atoms: int, shift=0):
        """
        Calculate the error between the estimated atoms and the true atoms.

        Parameters:
        - true_atoms (np.ndarray): The true atoms.
        - num_atoms (int): The number of atoms to estimate.
        - shift (float): The shifting factor for the atoms (default: 0).

        Returns:
        float: The error between the estimated atoms and the true atoms.
        """
        return wasserstein_distance(self.estimate(num_atoms, shift), true_atoms)

    def error_all(self, true_atoms: np.ndarray, shift=0):
        """
        Calculate the error for all possible numbers of atoms.

        Parameters:
        - true_atoms (np.ndarray): The true atoms.
        - shift (float): The shifting factor for the atoms (default: 0).

        Returns:
        np.ndarray: The errors for all possible numbers of atoms.
        """
        return np.array(
            [self.error(true_atoms, i, shift) for i in range(1, self.max_atoms + 1)]
        )

    def plot_estimated(self, num_atoms: int, shift=0):
        """
        Plot the estimated atoms.

        Parameters:
        - num_atoms (int): The number of atoms to estimate.
        - shift (float): The shifting factor for the atoms (default: 0).
        """
        estimated = self.estimate(num_atoms, shift)
        self.experiment.plot()
        plt.scatter(
            estimated[:, 0],
            estimated[:, 1],
            c="w",
            linewidths=0.25,
            edgecolors="k",
            s=10,
        )
        if self.experiment.atoms is not None:
            cost = self.error(self.experiment.atoms, num_atoms, shift)
            plt.title(f"ot cost = {cost:.2e}")
