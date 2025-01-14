from scipy.special import comb
import numpy as np

from optimal_deconvolution.normal_distribution import (
    complex_centered_normal_moment,
    complex_half_std_normal_moment,
)


class ComplexMomentApproximator:
    def __init__(self, kernel_moments: np.ndarray):
        """
        Initializes the ComplexMomentApproximator class.

        Parameters:
            kernel_moments (np.ndarray): The moments of the kernel distribution.
        """
        self.order = len(kernel_moments) - 1
        self.kernel_moments = kernel_moments
        self.coeffs = self.poly_coeffs_matrix(self.order, kernel_moments)

    def approximate(self, moments: np.ndarray) -> np.ndarray:
        """
        Approximates the moments of the underlying distribution, given the moments of the mixture, using the kernel moments.

        Parameters:
            moments (np.ndarray): The moments of the mixture distribution.

        Returns:
            np.ndarray: The approximated moments of the underlying distribution.
        """
        size = moments.size
        return self.coeffs[:size, :size] @ moments

    def poly_coeffs_matrix(self, k: int, kernel_moments: np.ndarray) -> np.ndarray:
        """
        Calculates the polynomial coefficients matrix.

        Parameters:
            k (int): The order of the polynomial.
            kernel_moments (np.ndarray): The moments of the kernel distribution.

        Returns:
            np.ndarray: The polynomial coefficients matrix.
        """
        coeffs = np.eye(k + 1, k + 1, dtype=np.complex128)
        for i in range(1, k):
            for j in range(i + 1):
                coeffs[i + 1] = coeffs[i + 1] - (
                    coeffs[j]
                    * int(comb(i + 1, j, exact=True))
                    * kernel_moments[i - j + 1]
                )

        return coeffs


class StdComplexNormMomentApproximator(ComplexMomentApproximator):
    def __init__(self, order: int):
        """
        Initializes the StdComplexNormMomentApproximator class.

        Parameters:
            order (int): The order of the moments.
        """
        kernel_moments = np.array([1] + [0 for _ in range(order)])

        super().__init__(kernel_moments)

    def poly_coeffs_matrix(self, k: int, _: np.ndarray) -> np.ndarray:
        """
        Calculates the polynomial coefficients matrix.

        Parameters:
            k (int): The order of the polynomial.
            _: No need for the moments of the kernel distribution.

        Returns:
            np.ndarray: The polynomial coefficients matrix.
        """
        return np.eye(k + 1, k + 1)


class HalfStdComplexNormMomentApproximator(ComplexMomentApproximator):
    def __init__(self, order: int, scale: float):
        """
        Initializes the StdComplexNormMomentApproximator class.

        Parameters:
            order (int): The order of the moments.
            scale (float): The scale of Gaussian distribution (kernel).
        """
        kernel_moments = np.array(
            [complex_half_std_normal_moment(i, scale) for i in range(order + 1)]
        )

        super().__init__(kernel_moments)


class ComplexNormMomentApproximator(ComplexMomentApproximator):
    def __init__(self, order: int, scale: float):
        """
        Initializes the StdComplexNormMomentApproximator class.

        Parameters:
            order (int): The order of the moments.
            scale (float): The scale of Gaussian distribution (kernel).
        """
        kernel_moments = np.array(
            [complex_centered_normal_moment(i, scale) for i in range(order + 1)]
        )

        super().__init__(kernel_moments)
