from scipy.special import comb
import scipy.stats as stats
import numpy as np

from poisson_deconvolution.normal_distribution import std_norm_moment_exact


class MomentApproximator:
    def __init__(self, kernel_moments: np.ndarray):
        """
        Initializes a MomentApproximator object.

        Parameters:
            kernel_moments (np.ndarray): The moments of the kernel distribution.
        """
        self.order = len(kernel_moments) - 1
        self.kernel_moments = kernel_moments
        self.coeffs = self.poly_coeffs_matrix(self.order, kernel_moments)

    def approximate(self, moments: np.ndarray) -> np.ndarray:
        """
        Approximates the moments of underlying distribution, given moments of mixture, using the kernel moments.

        Parameters:
            moments (np.ndarray): The moments of the mixture distribution.

        Returns:
            np.ndarray:  The approximated moments of the underlying distribution.
        """
        return np.dot(self.coeffs, moments)

    def poly_coeffs(self, k: int, kernel_moments: np.ndarray) -> np.ndarray:
        """
        Calculates the polynomial coefficients for a given order and kernel moments.

        Parameters:
            k (int): The order of the polynomial.
            kernel_moments (np.ndarray): The moments of the kernel distribution.

        Returns:
            np.ndarray: The polynomial coefficients.
        """
        row = lambda i: np.array(
            [int(comb(i, j, exact=True)) * kernel_moments[i - j] for j in range(i)]
        )
        coeffs = row(k)

        for i in range(1, k):
            coeffs[: k - i] = coeffs[: k - i] - row(k - i) * coeffs[k - i]

        return np.append(-coeffs, [1])

    def poly_coeffs_matrix(self, k: int, kernel_moments: np.ndarray) -> np.ndarray:
        """
        Calculates the matrix of polynomial coefficients for a given order and kernel moments.

        Parameters:
            k (int): The order of the polynomial.
            kernel_moments (np.ndarray): The moments of the kernel distribution.

        Returns:
            np.ndarray: The matrix of polynomial coefficients.
        """
        coeffs = np.eye(k + 1, k + 1)
        for i in range(1, k):
            for j in range(i + 1):
                coeffs[i + 1] -= (
                    coeffs[j]
                    * int(comb(i + 1, j, exact=True))
                    * kernel_moments[i - j + 1]
                )

        return coeffs


class GaussianMomentApproximator(MomentApproximator):
    def __init__(self, order: int, sigma=1):
        """
        Initializes a GaussianMomentApproximator object.

        Parameters:
            order (int): The order of the moments to be approximated.
            sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1.
        """
        kernel_moments = np.array(
            [stats.norm.moment(k, scale=sigma) for k in range(order + 1)]
        )

        super().__init__(kernel_moments)


class StdGaussianMomentApproximator(MomentApproximator):
    def __init__(self, order: int):
        """
        Initializes a StdGaussianMomentApproximator object.

        Parameters:
            order (int): The order of the moments to be approximated.
        """
        kernel_moments = np.array(
            [int(std_norm_moment_exact(i)) for i in range(order + 1)]
        )

        super().__init__(kernel_moments)
