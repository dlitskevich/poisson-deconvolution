import numpy as np


class MeasureMomentApproximator:
    moments, size, polynomial, locations = None, None, None, None

    def approximate_atoms(self, moments: np.ndarray) -> np.ndarray:
        """
        Approximates the atoms (locations) of the underlying measure based on the given moments.

        Parameters:
            moments (np.ndarray): The moments of the measure.

        Returns:
            np.ndarray: The locations of the atoms.
        """
        self.moments = moments
        self.size = moments.size
        self.polynomial = self.atoms_polynomial(moments)
        self.locations = np.roots(self.polynomial)

        return self.locations

    def epsilon_coeffs(self, moments: np.ndarray) -> np.ndarray:
        """
        Calculates the epsilon coefficients based on the given moments.

        Parameters:
            moments (np.ndarray): The moments of the measure.

        Returns:
            np.ndarray: The epsilon coefficients.
        """
        size = self.size
        coeffs = np.ones(size + 1, dtype=complex)
        ones = [1 - 2 * (j % 2) for j in range(size)]
        signed_moments = ones * moments
        for l in range(1, size + 1):
            coeffs[l] = size / l * np.dot(coeffs[:l][::-1], signed_moments[:l])

        return coeffs

    def atoms_polynomial(self, moments: np.ndarray) -> np.ndarray:
        """
        Calculates the polynomial for the atoms based on the given moments.

        Parameters:
            moments (np.ndarray): The moments of the measure.

        Returns:
            np.ndarray: The polynomial for the atoms.
        """
        eps_coeffs = self.epsilon_coeffs(moments)
        ones = np.array([1 - 2 * (i % 2) for i in range(eps_coeffs.size)])

        return eps_coeffs * ones

    def calculate_moments(self) -> np.ndarray:
        """
        Calculates the moments based on the locations of the atoms.

        Returns:
            np.ndarray: The calculated moments.
        """
        return np.array([(self.locations**i).mean() for i in range(1, self.size + 1)])

    def error(self) -> float:
        """
        Calculates the error between the given moments and the calculated moments.

        Returns:
            float: The error.
        """
        return np.linalg.norm(self.moments - self.calculate_moments())
