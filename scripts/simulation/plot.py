import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from poisson_deconvolution.microscopy.atoms import AtomsData, AtomsType
from poisson_deconvolution.microscopy.estimators import EstimatorType
from scripts.simulation.data_utils import StatsDataArray
from scripts.simulation.plot_utils import COLORS_ESTIMATORS, set_box_color


def plot_data(atoms_data_list: list[AtomsData], savepath: str):
    plt.figure(figsize=(15, 5))
    for i, atoms_data in enumerate(atoms_data_list):
        plt.subplot(1, 3, i + 1)
        atoms_data.plot()
        plt.title("")
        plt.xticks([])
        plt.yticks([])
    path = os.path.join(savepath, "points_arrangements.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_errors_bin_scale(
    data: StatsDataArray,
    atoms_types: list[AtomsType],
    n_bins_list: list[int],
    scales: list[float],
    savepath: str,
):
    estimators = [EstimatorType.Moment, EstimatorType.EMMoment]
    names = ["Moment", "EM"]
    num = len(atoms_types)
    plt.figure(figsize=(num * 5, len(scales) * 5))
    for j, scale in enumerate(scales):
        for i, atoms_type in enumerate(atoms_types):
            ax = plt.subplot(len(scales), num, j * len(scales) + i + 1)
            for l, estimator in enumerate(estimators):
                ax.add_line(Line2D([], [], color="none", label=names[l]))
                data.plot_errors_estimator(
                    estimator,
                    [scale],
                    n_bins_list,
                    atoms_type,
                    4,
                    legend=False,
                    colors=[COLORS_ESTIMATORS[estimator]],
                    scale_label=False,
                )
            plt.title("")
    plt.legend(loc="upper right", ncol=10, bbox_to_anchor=(0.9, -0.15))

    # plt.tight_layout()
    path = os.path.join(savepath, f"error_{"_".join(names)}.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_time(
    data: StatsDataArray,
    atom_type: AtomsType,
    n_bins_list: list[int],
    scales: list[float],
    n_points=4,
):
    keys = [EstimatorType.Moment, EstimatorType.EMMoment]
    names = ["Moment", "EM"]
    res = {k: [] for k in keys}
    for n_bins in n_bins_list:
        for scale in scales:
            time_data = data.get_time_data(scale, n_bins, atom_type, n_points)
            res_bin = {k: [] for k in keys}
            for name, times in time_data.items():
                for i in range(len(keys)):
                    if name == keys[i].name:
                        res_bin[keys[i]] += times
                        break

        for n, v in res_bin.items():
            res[n].append(v)

    x_pos = np.array(range(len(n_bins_list))) * 2.0

    legends = []

    for i in range(len(keys)):
        xs = x_pos  # + shifts[i]

        # box plot
        bpl = plt.boxplot(res[keys[i]], positions=xs, sym="", widths=0.5)
        set_box_color(bpl, COLORS_ESTIMATORS[keys[i]])
        legends.append(bpl["boxes"][0])

    plt.xticks(x_pos, n_bins_list)
    plt.ylim(1e-4, 1e3)
    plt.yscale("log")
    plt.legend(legends, names, loc="upper left")
    plt.xlabel("Number of bins")
    plt.ylabel("Elapsed time, s")


def plot_time_types(
    data: StatsDataArray,
    atoms_types: list[AtomsType],
    n_bins_list: list[int],
    scales: list[float],
    savepath: str,
):
    n_col = len(atoms_types)
    plt.figure(figsize=(5 * n_col, 5))
    for i, atoms_type in enumerate(atoms_types):
        plt.subplot(1, n_col, i + 1)
        plot_time(data, atoms_type, n_bins_list, scales)
        if i > 0:
            plt.ylabel("")

    path = os.path.join(savepath, "time_complexity.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close
