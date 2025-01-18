import numpy as np
from scipy import stats
import scipy.optimize as optim
from matplotlib import pyplot as plt

from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.microscopy.microscopy_estimator import (
    MicroscopyEstimator,
    StdMicroscopyCommon,
)


class MicroscopyEM(MicroscopyEstimator):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the MicroscopyEM class.

        Parameters:
            experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)

    def compute_atom_lambdas(self, mu: np.ndarray, scale, t) -> np.ndarray:
        """
        Compute the lambdas for each atom.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            xy_prob: The x and y probabilities.

        Returns:
            np.ndarray: The atom lambdas.
        """
        pass

    def get_q_func(self, multinomial_prob: np.ndarray, scale, t, jac=None):
        """
        Get the q function.

        Parameters:
            multinomial_prob (np.ndarray): The multinomial probabilities.
            scale: The scale value.
            t: The illumination time.
            jac (bool): Whether to compute the Jacobian.

        Returns:
            function: The q function.
        """
        sample = self.data.ravel()

        def q_func(mu):
            atom_lambdas = self.compute_atom_lambdas(mu, scale, t)
            lambdas = atom_lambdas.sum(axis=-1)
            weighted_lambdas = np.sum(
                multinomial_prob * np.log(atom_lambdas + 1e-20), axis=-1
            )
            value = np.sum(sample * weighted_lambdas - lambdas)

            return -value

        return q_func

    def expectation_step(self, mu_tilde: np.ndarray, scale, t, jac=None):
        """
        Perform the expectation step.

        Parameters:
            mu_tilde (np.ndarray): The given mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            jac (bool): Whether to compute the Jacobian.

        Returns:
            function: The q function.
        """
        atom_lambdas_tilde = self.compute_atom_lambdas(mu_tilde, scale, t)
        lambdas_tilde = atom_lambdas_tilde.sum(axis=-1)
        multinomial_prob = atom_lambdas_tilde / (lambdas_tilde[:, None] + 1e-20)

        return self.get_q_func(multinomial_prob, scale, t, jac)

    def maximization_step(self, mu_tilde: np.ndarray, scale, t, jac=None) -> np.ndarray:
        """
        Perform the maximization step.

        Parameters:
            mu_tilde (np.ndarray): The given mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            jac (bool): Whether to compute the Jacobian.

        Returns:
            np.ndarray: The updated mu (atoms locations) values.
        """
        q_func = self.expectation_step(mu_tilde, scale, t, jac)
        bounds = [(0, 1)] * mu_tilde.size

        def objective(mu: np.ndarray):
            return q_func(mu.reshape(mu_tilde.shape))

        result = optim.minimize(objective, mu_tilde.ravel(), bounds=bounds, jac=jac)
        atoms = result.x.reshape(mu_tilde.shape)

        return atoms

    def estimate(
        self, mu_0: np.ndarray, scale, jac=None, max_iter=5, tol=1e-6
    ) -> np.ndarray:
        """
        Estimate The mu (atoms locations) values.

        Parameters:
            mu_0 (np.ndarray): The initial mu (atoms locations) values.
            scale: The scale value.
            jac (bool): Whether to compute the Jacobian.
            max_iter (int): The maximum number of iterations.
            tol (float): The tolerance value.

        Returns:
            np.ndarray: The estimated mu (atoms locations) values.
        """
        self.mu_0 = mu_0
        mu_tilde = mu_0
        tol *= mu_0.size
        for i in range(max_iter):
            mu_new = self.maximization_step(mu_tilde, scale, self.t, jac)
            if np.linalg.norm(mu_tilde - mu_new) < tol:
                break
            mu_tilde = mu_new

        self.total_iter = i
        self.atoms = mu_new

        return self.atoms


class StdMicroscopyEM(MicroscopyEM, StdMicroscopyCommon):

    def compute_atom_lambdas(
        self, mu: np.ndarray, scale, t, xy_prob=None
    ) -> np.ndarray:
        """
        Compute the lambdas for each atom.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            xy_prob: The x and y probabilities.

        Returns:
            np.ndarray: The atom lambdas.
        """
        num_atoms = mu.shape[0]
        if xy_prob is None:
            xy_prob = self.get_xy_probabilities(mu, scale)

        atom_bin_prob = np.array([xy_prob[1] * x for x in xy_prob[0]])
        atom_lambdas = t / num_atoms * atom_bin_prob

        return np.reshape(atom_lambdas, (-1, num_atoms))

    def get_q_func(self, multinomial_prob: np.ndarray, scale, t, jac=True):
        """
        Get the q function.

        Parameters:
            multinomial_prob (np.ndarray): The multinomial probabilities.
            scale: The scale value.
            t: The illumination time.
            jac (bool): Whether to compute the Jacobian.

        Returns:
            function: The q function.
        """
        sample = self.data.ravel()

        def q_func(mu):
            xy_prob = self.get_xy_probabilities(mu, scale)
            atom_lambdas = self.compute_atom_lambdas(mu, scale, t, xy_prob)
            lambdas = atom_lambdas.sum(axis=-1)
            weighted_lambdas = np.sum(
                multinomial_prob * np.log(atom_lambdas + 1e-20), axis=-1
            )
            value = np.sum(sample * weighted_lambdas - lambdas)
            if jac is not True:
                return -value

            bin_grad = self.compute_grad_lambdas(mu, scale, t, xy_prob)

            pre_grad = sample[:, None] * multinomial_prob / (atom_lambdas + 1e-20) - 1

            grad = np.array(
                [
                    np.sum(pre_grad * bin_grad[:, ::2], 0),
                    np.sum(pre_grad * bin_grad[:, 1::2], 0),
                ]
            ).T
            return -value, -grad.ravel()

        return q_func

    def expectation_step(self, mu_tilde: np.ndarray, scale, t, jac=True):
        """
        Perform the expectation step.

        Parameters:
            mu_tilde (np.ndarray): The given mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            jac (bool): Whether to compute the Jacobian.

        Returns:
            function: The q function.
        """
        return super().expectation_step(mu_tilde, scale, t, jac)

    def maximization_step(self, mu_tilde: np.ndarray, scale, t, jac=True) -> np.ndarray:
        """
        Perform the maximization step.

        Parameters:
            mu_tilde (np.ndarray): The given mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            jac (bool): Whether to compute the Jacobian.

        Returns:
            np.ndarray: The updated mu (atoms locations) values.
        """
        return super().maximization_step(mu_tilde, scale, t, jac)

    def compute_grad_lambdas(
        self, mu: np.ndarray, scale, t, xy_prob=None
    ) -> np.ndarray:
        """
        Compute the gradient of lambdas.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.
            xy_prob: The x and y probabilities.

        Returns:
            np.ndarray: The gradient of lambdas.
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

    def estimate(
        self, mu_0: np.ndarray, scale, jac=True, max_iter=50, tol=1e-6
    ) -> np.ndarray:
        """
        Estimate The mu (atoms locations) values.

        Parameters:
            mu_0 (np.ndarray): The initial mu (atoms locations) values.
            scale: The scale value.
            jac (bool): Whether to compute the Jacobian.
            max_iter (int): The maximum number of iterations.
            tol (float): The tolerance value.

        Returns:
            np.ndarray: The estimated mu (atoms locations) values.
        """
        return super().estimate(mu_0, scale, jac, max_iter, tol)

    def plot_estimated(self, mu_0: np.ndarray, scale, jac=True, estimated=None):
        """
        Plot the estimated mu (atoms locations) values.

        Parameters:
            mu_0 (np.ndarray): The initial mu (atoms locations) values.
            scale: The scale value.
            jac (bool): Whether to compute the Jacobian.
            estimated (np.ndarray): The estimated mu (atoms locations) values.
        """
        if estimated is None:
            estimated = self.estimate(mu_0, scale, jac)
        grad = self.expectation_step(estimated, scale, self.t, True)(estimated)[1]

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

        plt.xticks([])
        plt.yticks([])
        if self.experiment.atoms is not None:
            cost = self.error(self.experiment.atoms, estimated)
            plt.title(f"ot: {cost:.2e}\ngrad:{np.abs(grad).max():.2e}")
