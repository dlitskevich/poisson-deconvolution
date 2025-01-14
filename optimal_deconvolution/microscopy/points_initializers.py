import numpy as np

from optimal_deconvolution.microscopy import MicroscopyExperiment
from optimal_deconvolution.microscopy.moment_estimator import (
    StdMicroscopyMomentEstimator,
)
from optimal_deconvolution.microscopy.sampler import MicroscopySampler


def mode_from_std_data(
    experiment: MicroscopyExperiment,
    num_atoms: int,
    scale: float,
    samplerCls: type[MicroscopySampler],
):
    """
    Initializes points based on iterative approach mode selection.

    Parameters:
        experiment (MicroscopyExperiment): The microscopy experiment object.
        num_atoms (int): The number of atoms to initialize.
        scale (float): The scale factor for the sampler.
        samplerCls (MicroscopySampler): The sampler class.

    Returns:
        numpy.ndarray: The initialized points as a numpy array of shape (num_atoms, 2).
    """
    data = experiment.data / experiment.data.sum()
    t = experiment.t
    n_bins = experiment.data.shape
    best_loc = -np.ones((num_atoms, 2))
    diff = data
    center = (np.array([n_bins]) // 2 + 0.5) / max(n_bins)
    sampler = samplerCls(center, n_bins, scale, t)
    exp = sampler.sample_convolution()

    for i in range(num_atoms):
        max_id = np.argmax(diff)
        best_loc[i, :] = experiment.bins_loc[max_id]

        max_id_x, max_id_y = np.unravel_index(max_id, diff.shape)
        left = max(0, max_id_x - n_bins[0] // 2)
        right = min(diff.shape[0], max_id_x + (n_bins[0] + 1) // 2)
        bottom = max(0, max_id_y - n_bins[1] // 2)
        top = min(diff.shape[1], max_id_y + (n_bins[1] + 1) // 2)

        sample = exp.data[
            n_bins[0] // 2 - (max_id_x - left) : n_bins[0] // 2 + (right - max_id_x),
            n_bins[1] // 2 - (max_id_y - bottom) : n_bins[1] // 2 + (top - max_id_y),
        ]
        diff[left:right, bottom:top] -= sample / num_atoms

    return best_loc


def from_moment_estimator(
    experiment: MicroscopyExperiment,
    num_atoms: int,
    scale: float,
    shift=0,
    momCls=StdMicroscopyMomentEstimator,
):
    """
    Initializes points based on the moment estimator.

    Parameters:
        experiment (MicroscopyExperiment): The microscopy experiment object.
        num_atoms (int): The number of atoms to initialize.
        shift (int, optional): The shift value. Defaults to 0.
        scale (float): The scale value.
        momCls (StdMicroscopyMomentEstimator): The moment estimator class.

    Returns:
        numpy.ndarray: The initialized points as a numpy array of shape (num_atoms, 2).
    """
    estimator = momCls(experiment, num_atoms, scale)

    return estimator.estimate(num_atoms, shift)


def circle_points(number: int):
    """
    Generates points on a circle.

    Parameters:
        number (int): The number of points to generate.

    Returns:
        numpy.ndarray: The generated points as a numpy array of shape (number, 2).
    """
    rad = np.linspace(0, 2 * np.pi, number, endpoint=False)

    return np.stack([np.cos(rad), np.sin(rad)], axis=-1)
