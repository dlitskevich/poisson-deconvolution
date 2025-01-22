from typing import List
from matplotlib import pyplot as plt

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


def plot_estimations_row(
    config: Config,
    scale: float = 0.1,
    t: float = 1e8,
    n_bins_list: List[int] = [10, 20, 40],
    atoms_type: AtomsType = AtomsType.Corners,
    n_points: int = 4,
    figsize: float = 5.5,
    savepath=None,
):
    """
    Plot a row of estimations for different amount of bins.

    Parameters:
        config (Config): The configuration for the experiment.
        scale (float, optional): The scale parameter for the estimation. Defaults to 0.1.
        t (float, optional): The illumination time. Defaults to 1e8.
        n_bins_list (List[int], optional): The list of number of bins. Defaults to [10, 20, 40].
        atoms_type (AtomsType, optional): The type of atoms. Defaults to AtomsType.Corners.
        n_points (int, optional): The number of points for atoms. Defaults to 4.
        figsize (float, optional): The figure size. Defaults to 5.5.
    """
    num = len(n_bins_list)
    plt.figure(figsize=(num * figsize, figsize))
    for i, n_bins in enumerate(n_bins_list):
        atoms = AtomsData.from_type(atoms_type, n_points).atoms
        exp = config.sampler(atoms, n_bins, scale, t).sample()
        estim = MicroscopyEstimators(exp, scale, config)
        plt.subplot(1, num, i + 1)
        estim.plot()
        plt.title(f"bins:{n_bins}")

    plt.suptitle(
        f"{atoms_type.name}({n_points} points) scale:{scale} t:{t:.0e}",
        fontsize=16,
        horizontalalignment="left",
        x=0,
    )

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(f"{savepath}ind_{atoms_type.name}.png", bbox_inches="tight")


def plot_estimations_atoms(
    config: Config,
    scale: float = 0.1,
    t: float = 1e8,
    n_bins_list: List[int] = [10, 20, 40],
    atom_types: List[AtomsType] = [
        AtomsType.Grid,
        AtomsType.Line1,
        AtomsType.Line2,
        AtomsType.UShape,
    ],
    n_points: int = 4,
    figsize: float = 5.5,
    savepath=None,
):
    """
    Plot estimations for different atom types.

    Parameters:
        config (Config): The configuration for the experiment.
        scale (float, optional): The scale parameter for the estimation. Defaults to 0.1.
        t (float, optional): The illumination time. Defaults to 1e8.
        n_bins_list (List[int], optional): The list of number of bins. Defaults to [10, 20, 40].
        atom_types (List[AtomsType], optional): The list of atom types. Defaults to [AtomsType.Grid, AtomsType.Line1, AtomsType.Line2, AtomsType.UShape].
        n_points (int, optional): The number of points for atoms. Defaults to 4.
        figsize (float, optional): The figure size. Defaults to 5.5.
    """
    for atom_type in atom_types:
        plot_estimations_row(
            config, scale, t, n_bins_list, atom_type, n_points, figsize, savepath
        )
