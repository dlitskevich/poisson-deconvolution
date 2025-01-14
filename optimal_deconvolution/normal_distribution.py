from math import comb
import numpy as np
from scipy import stats

from optimal_deconvolution.utils import count_partitions_binomial


def double_factorial(n: int) -> int:
    """
    Calculate the double factorial of a given number.

    Parameters:
    n (int): The number for which the double factorial is calculated.

    Returns:
    int: The double factorial of the given number.

    Examples:
    >>> double_factorial(5)
    15
    >>> double_factorial(6)
    48
    """
    return 1 if n in (0, 1) else n * double_factorial(n - 2)


def std_norm_moment_exact(n: int) -> int:
    """
    Calculate the exact standard normal moment of order n.

    Parameters:
    n (int): The order of the moment to calculate.

    Returns:
    int: The value of the exact standard normal moment of order n.
    """
    if n == 0:
        return 1

    return 0 if n % 2 else double_factorial(n - 1)


def complex_centered_normal_mixed_moment(n: int, k: int, scale: np.ndarray) -> complex:
    """
    Calculate the complex centered normal mixed moment of order n.

    Parameters:
    n (int): The order of the moment to calculate.
    k (int): The order of the second component of the moment.
    scale (np.array): The scale matrix of the Gaussian distribution.

    Returns:
    complex: The value of the complex normal moment of order n.
    """
    # Isserlis' theorem: https://de.wikipedia.org/wiki/Satz_von_Isserlis
    if n <= 0:
        raise ValueError("The order of the moment must be greater than 0.")
    if n % 2 == 1:
        return 0
    if k == 0:
        return std_norm_moment_exact(n) * scale[0, 0] ** (n / 2)
    if k == n:
        return std_norm_moment_exact(n) * scale[1, 1] ** (n / 2)

    scales_ordered = [scale[0, 0], scale[0, 1], scale[1, 1]]
    partitions = count_partitions_binomial(n, k)
    moment = 0
    for partition, count in partitions.items():
        product = 1
        for group in partition:
            product *= scales_ordered[group]
        moment += count * product

    return moment


def complex_centered_normal_moment(n: int, scale: np.ndarray) -> complex:
    """
    Calculate the complex centered normal moment of order n.

    Parameters:
    n (int): The order of the moment to calculate.
    scale (np.array): The scale matrix of the Gaussian distribution.

    Returns:
    complex: The value of the complex normal moment of order n.
    """
    if n == 0:
        return 1

    moment = 0
    for k in range(n + 1):
        moment += (
            comb(n, k) * (1j) ** k * complex_centered_normal_mixed_moment(n, k, scale)
        )

    return moment


def complex_std_normal_moment(n: int) -> complex:
    """
    Calculate the complex standard normal moment of order n.

    Parameters:
    n (int): The order of the moment to calculate.

    Returns:
    complex: The value of the complex standard normal moment of order n.
    """
    moment = 0
    for k in range(n + 1):
        moment += (
            comb(n, k)
            * (1j) ** k
            * std_norm_moment_exact(k)
            * std_norm_moment_exact(n - k)
        )
    return moment


def complex_half_std_normal_moment(n: int, scale: float) -> complex:
    """
    Calculate the complex standard normal moment of order n with halved scale on horizontal axis.

    Parameters:
    n (int): The order of the moment to calculate.
    scale (float): The scale of Gaussian distribution (kernel).

    Returns:
    complex: The value of the complex standard normal moment of order n.
    """
    moment = 0
    for k in range(n + 1):
        moment += (
            comb(n, k)
            * (1j) ** k
            * std_norm_moment_exact(k)
            * std_norm_moment_exact(n - k)
            / 2 ** (n - k)
        )
    return scale**n * moment


def complex_normal_pdf(x, y, loc: complex, scale: int = 1) -> float:
    """
    Calculate the probability density function (PDF) of a complex normal distribution.

    Parameters:
    x: The x-coordinate of the point at which to evaluate the PDF.
    y: The y-coordinate of the point at which to evaluate the PDF.
    loc (complex): The location parameter of the complex normal distribution.
    scale (int, optional): The scale parameter of the complex normal distribution. Defaults to 1.

    Returns:
    float: The value of the PDF at the given point (x, y).
    """
    x_pdf = stats.norm.pdf(x, np.real(loc), scale)
    y_pdf = stats.norm.pdf(y, np.imag(loc), scale)
    return x_pdf * y_pdf
