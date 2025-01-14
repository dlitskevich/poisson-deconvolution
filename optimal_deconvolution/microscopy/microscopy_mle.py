from typing import Callable
import numpy as np
from scipy import stats
import scipy.optimize as optim
from matplotlib import pyplot as plt

from optimal_deconvolution.microscopy.experiment import MicroscopyExperiment
from optimal_deconvolution.microscopy.microscopy_estimator import (
    MicroscopyEstimator,
    StdMicroscopyCommon,
)
from optimal_deconvolution.wasserstein_distance import wasserstein_distance


class MicroscopyMLE(MicroscopyEstimator):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initializes the MicroscopyMLE class.

        Parameters:
            experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)

    def compute_lambdas(self):
        """
        Computes the lambdas.
        """
        pass

    def log_likelihood(
        self, mu: np.ndarray, sample: np.ndarray, scale, t, lambdas=None
    ) -> float:
        """
        Computes the log-likelihood.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            sample (np.ndarray): The sample array.
            scale (float): The scale value.
            t (float): The illumination time.
            lambdas (Optional[np.ndarray]): The lambdas array. Defaults to None.

        Returns:
            float: The log-likelihood value.
        """
        if lambdas is None:
            lambdas = self.compute_lambdas(mu, scale, t)

        return np.sum(sample.ravel() * np.log(lambdas + 1e-20) - lambdas)

    def get_objective_function(self, mu_0: np.ndarray, scale, jac=True):
        """
        Get the objective function.

        Parameters:
            mu_0 (np.ndarray): The initial mu array.
            scale (float): The scale value.
            jac (bool, optional): Whether to compute the Jacobian. Defaults to True.

        Returns:
            Callable: The objective function.
        """
        objective: Callable[[np.ndarray], float] = lambda mu: -self.log_likelihood(
            mu.reshape(mu_0.shape), self.data, scale, self.t
        )

        return objective

    def estimate(self, mu_0: np.ndarray, scale, jac=None):
        """
        Estimates the mu (atoms locations) values.

        Parameters:
            mu_0 (np.ndarray): The initial mu array.
            scale (float): The scale value.
            jac (bool, optional): Whether to compute the Jacobian. Defaults to None.

        Returns:
            np.ndarray: The estimated mu array.
        """
        self.mu_0 = mu_0

        # one may want to try penalties
        objective = self.get_objective_function(mu_0, scale, jac)

        result = optim.minimize(
            objective,
            mu_0.ravel(),
            bounds=[(0, 1)] * mu_0.size,
            jac=jac,
        )
        self.result = result
        self.atoms = result.x.reshape(mu_0.shape)

        return self.atoms


class StdMicroscopyMLE(MicroscopyMLE, StdMicroscopyCommon):

    def compute_lambdas(self, mu: np.ndarray, scale, t, xy_prob=None):
        """
        Computes the lambdas.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.
            t (float): The illumination time.
            xy_prob (Optional[Tuple[np.ndarray, np.ndarray]]): The x and y probabilities. Defaults to None.

        Returns:
            np.ndarray: The lambdas array.
        """
        num_atoms = mu.shape[0]
        if xy_prob is None:
            xy_prob = self.get_xy_probabilities(mu, scale)

        bin_prob = np.array([xy_prob[1] @ x for x in xy_prob[0]])
        self.lambdas = t / num_atoms * bin_prob.ravel()

        return self.lambdas

    def compute_grad_lambdas(self, mu: np.ndarray, scale, t, xy_prob=None):
        """
        Computes the gradient of lambdas.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.
            t (float): The illumination time.
            xy_prob (Optional[Tuple[np.ndarray, np.ndarray]]): The x and y probabilities. Defaults to None.

        Returns:
            np.ndarray: The gradient of lambdas array.
        """
        num_atoms = mu.shape[0]
        dens_diff = self.get_xy_density_diff(mu, scale)
        if xy_prob is None:
            xy_prob = self.get_xy_probabilities(mu, scale)
        bin_grad = np.zeros((*self.n_bins, 2 * num_atoms))
        bin_grad[:, :, ::2] = np.array([xy_prob[1] * x for x in dens_diff[0]])
        bin_grad[:, :, 1::2] = np.array([dens_diff[1] * x for x in xy_prob[0]])
        bin_grad = np.row_stack(bin_grad)
        bin_grad = -t / num_atoms * bin_grad

        return bin_grad

    def grad_log_likelihood(
        self,
        mu: np.ndarray,
        sample: np.ndarray,
        scale,
        t,
        lambdas=None,
        xy_prob=None,
    ):
        """
        Computes the gradient of log-likelihood.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            sample (np.ndarray): The sample array.
            scale (float): The scale value.
            t (float): The illumination time.
            lambdas (Optional[np.ndarray]): The lambdas array. Defaults to None.
            xy_prob (Optional[Tuple[np.ndarray, np.ndarray]]): The x and y probabilities. Defaults to None.

        Returns:
            np.ndarray: The gradient of log-likelihood array.
        """
        if xy_prob is None:
            xy_prob = self.get_xy_probabilities(mu, scale)
        if lambdas is None:
            lambdas = self.compute_lambdas(mu, scale, t, xy_prob)
        bin_grad = self.compute_grad_lambdas(mu, scale, t, xy_prob)
        grad_lambdas = (sample.ravel() / (lambdas + 1e-20) - 1) @ bin_grad

        return grad_lambdas

    def get_objective_function(self, mu_0: np.ndarray, scale, jac=True):
        """
        Get the objective function.

        Parameters:
            mu_0 (np.ndarray): The initial mu array.
            scale (float): The scale value.
            jac (bool, optional): Whether to compute the Jacobian. Defaults to True.

        Returns:
            Callable: The objective function.
        """
        if jac == True:

            def objective(mu: np.ndarray):
                mu = mu.reshape(mu_0.shape)

                xy_prob = self.get_xy_probabilities(mu, scale)
                lambdas = self.compute_lambdas(mu, scale, self.t, xy_prob)
                return (
                    -self.log_likelihood(mu, self.data, scale, self.t, lambdas),
                    -self.grad_log_likelihood(
                        mu, self.data, scale, self.t, lambdas, xy_prob
                    ),
                )

        else:
            objective: Callable[[np.ndarray], float] = lambda mu: -self.log_likelihood(
                mu.reshape(mu_0.shape), self.data, scale, self.t
            )

        return objective

    def estimate(self, mu_0: np.ndarray, scale, jac=True):
        """
        Estimates the mu (atoms locations) values.

        Parameters:
            mu_0 (np.ndarray): The initial mu array.
            scale (float): The scale value.
            jac (bool, optional): Whether to compute the Jacobian. Defaults to True.

        Returns:
            np.ndarray: The estimated mu array.
        """
        return super().estimate(mu_0, scale, jac)

    def plot_estimated(self, mu_0: np.ndarray, scale, jac=True, estimated=None):
        """
        Plots the estimated mu array.

        Parameters:
            mu_0 (np.ndarray): The initial mu array.
            scale (float): The scale value.
            jac (bool, optional): Whether to compute the Jacobian. Defaults to True.
            estimated (Optional[np.ndarray]): The estimated mu array. Defaults to None.
        """
        if estimated is None:
            estimated = self.estimate(mu_0, scale, jac)
        grad = self.grad_log_likelihood(estimated, self.data, scale, self.t)

        if mu_0 is not None:
            plt.scatter(
                mu_0[:, 0], mu_0[:, 1], c="b", edgecolors="w", linewidths=0.5, s=10
            )

        self.experiment.plot()
        plt.scatter(
            estimated[:, 0],
            estimated[:, 1],
            c="w",
            linewidths=0.5,
            edgecolors="k",
            s=10,
        )
        # plt.quiver(estimated[:, 0], estimated[:, 1], grad[::2], grad[1::2])
        plt.xticks([])
        plt.yticks([])
        if self.experiment.atoms is not None:
            cost = self.error(self.experiment.atoms, estimated)
            plt.title(f"ot: {cost:.2e}\ngrad:{np.abs(grad).max():.2e}")
