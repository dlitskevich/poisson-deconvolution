import numpy as np
from optimal_deconvolution.microscopy.experiment import MicroscopyExperiment


def mode_from_data(
    experiment: MicroscopyExperiment, num_atoms: int, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes points based on iterative approach mode selection.

    Parameters:
        experiment (MicroscopyExperiment): The microscopy experiment object.
        num_atoms (int): The number of atoms to initialize.
        kernel (np.ndarray): Kernel, that centered at (n_bins // 2, n_bins // 2).

    Returns:
        tuple[np.ndarray, np.ndarray]: The best locations and the denoised data.
    """
    data = experiment.data / experiment.data.sum()
    n_bins = max(experiment.data.shape)
    best_loc = -np.ones((num_atoms, 2))
    diff = data

    for i in range(num_atoms):
        max_id = np.argmax(diff)
        mode_loc = experiment.bins_loc[max_id]
        best_loc[i, :] = mode_loc
        max_id_x, max_id_y = np.unravel_index(max_id, diff.shape)
        left = max(0, max_id_x - n_bins // 2)
        right = min(diff.shape[0], max_id_x + (n_bins + 1) // 2)
        bottom = max(0, max_id_y - n_bins // 2)
        top = min(diff.shape[1], max_id_y + (n_bins + 1) // 2)

        diff[left:right, bottom:top] -= (
            kernel[
                n_bins // 2 - (max_id_x - left) : n_bins // 2 + (right - max_id_x),
                n_bins // 2 - (max_id_y - bottom) : n_bins // 2 + (top - max_id_y),
            ]
            / kernel.max()
            * diff[max_id_x, max_id_y]
        )
    diff = np.ma.array(diff, mask=diff < 0).filled(0)

    return best_loc, data - diff * experiment.t
