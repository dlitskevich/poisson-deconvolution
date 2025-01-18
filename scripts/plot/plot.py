import os
from matplotlib import pyplot as plt

from poisson_deconvolution.microscopy.estimators import EstimatorType
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from scripts.estimation.estimation_results import SplitEstimationsResults
from scripts.plot.types import COLORS, PREFIX


def plot_estimated(
    exp: MicroscopyExperiment,
    split_res: SplitEstimationsResults,
    num_atoms_list: list[int],
    estimator: EstimatorType,
    savepath: str = None,
):
    n_col = len(num_atoms_list)
    estimations = [split_res.estimations, split_res.denoised]
    for l, estims in enumerate(estimations):
        plt.figure(figsize=(3 * n_col, 3))
        for k, num_atoms in enumerate(num_atoms_list):
            plt.subplot(1, n_col, k + 1)
            exp.plot_data("binary")
            for s, e in enumerate(estims[num_atoms]):
                e.plot(estimator, color=COLORS[l], alpha=0.5, s=9)

        if savepath is not None:
            name = "estims_" + PREFIX[l]
            name += "_".join([str(n) for n in num_atoms_list])
            path = os.path.join(savepath, name + ".pdf")
            plt.savefig(path, bbox_inches="tight", dpi=300)
