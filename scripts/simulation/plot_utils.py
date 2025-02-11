import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np

from poisson_deconvolution.microscopy.atoms import AtomsData, AtomsType
from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy.estimators import (
    EstimatorType,
    MicroscopyEstimators,
)


VIS_SCALES = [0.05, 0.1, 0.5]
VIS_N_BINS = [10, 20, 40, 80]
VIS_N_POINTS = [4, 5, 7, 10]

COLORS_ESTIMATORS = {
    EstimatorType.Mode: "darkorange",
    EstimatorType.Moment: "darkorange",
    EstimatorType.EMMode: "darkblue",
    EstimatorType.EMMoment: "blue",
    # grouped estimators
    "Mode": "darkorange",
    "MoM": "darkorange",
    "EM": "blue",
}

LINESTYLE_ESTIMATORS = {
    EstimatorType.Mode: ":",
    EstimatorType.Moment: "-",
    EstimatorType.EMMode: ":",
    EstimatorType.EMMoment: "-",
    # grouped estimators
    "Mode": "darkorange",
    "MoM": "darkorange",
    "EM": "blue",
}


def set_box_color(bp, color):
    plt.setp(bp["boxes"], color=color)
    plt.setp(bp["whiskers"], color=color)
    plt.setp(bp["caps"], color=color)
    plt.setp(bp["medians"], color=color)


def plot_estimations(
    config: Config,
    scale: float,
    t_list: List[float],
    n_bins_list: List[int],
    atoms_type: AtomsType,
    figsize: float = 5.5,
    savepath=None,
):
    """
    Plot a row of estimations for different amount of bins.

    Parameters:
        config (Config): The configuration for the experiment.
        scale (float): The scale parameter for the estimation.
        t (float): The illumination time.
        n_bins_list (List[int]): The list of number of bins.
        atoms_type (AtomsType): The type of atoms.
        figsize (float, optional): The figure size. Defaults to 5.5.
    """
    t_list = [np.inf] + t_list
    n_col = len(n_bins_list)
    n_row = len(t_list)
    plt.figure(figsize=(n_col * figsize, n_row * figsize))
    for i, t in enumerate(t_list):
        for j, n_bins in enumerate(n_bins_list):
            atoms = AtomsData.from_type(atoms_type).atoms
            if t == np.inf:
                exp = config.sampler(atoms, n_bins, scale, 1).sample_convolution()
            else:
                exp = config.sampler(atoms, n_bins, scale, t).sample()
            estim = MicroscopyEstimators(exp, scale, config)
            plt.subplot(n_row, n_col, n_col * i + j + 1)
            estim.plot()
            if i == 0:
                plt.title(f"$m={n_bins}^2$")
            if j == 0:
                label = f"$t=10^{int(np.log10(t))}$" if t != np.inf else "$t=\infty$"
                plt.ylabel(label)

    plt.tight_layout()
    if savepath is not None:
        path = os.path.join(savepath, f"ind_{atoms_type.name}.pdf")
        plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
