import json
import os
from os import listdir
from os.path import isfile, join

from matplotlib import pyplot as plt

from poisson_deconvolution.microscopy import AtomsType
from poisson_deconvolution.microscopy.estimators import EstimatorType

from .error_data import StatsData, ErrorType


ALL_SCALES = [0.01, 0.05, 0.1, 0.2, 0.5]
ALL_N_BINS = [10, 20, 30, 40, 80]
ALL_N_POINTS = [4, 5, 7, 10]

ALL_ATOM_TYPES = [
    AtomsType.Grid,
    AtomsType.Line1,
    AtomsType.Line2,
    AtomsType.UShape,
    AtomsType.Corners,
]
NUM_REPS = 20


def get_filenames(rep=NUM_REPS, folder="data"):
    filenames = [f for f in listdir(folder) if isfile(join(folder, f))]
    filenames = [f for f in filenames if f.startswith(f"err_") and f"_rep{rep}_" in f]
    return filenames


def read_data_files(filenames: list[str], folder="data"):
    stats: list[StatsData] = []
    for f in filenames:
        with open(os.path.join(folder, f), "rb") as file:
            data = json.load(file)
            stats.append(StatsData.from_json(data))
    return stats


class StatsDataArray:
    def read_data_files(rep=NUM_REPS, folder="data"):
        filenames = get_filenames(rep, folder)
        data = StatsDataArray(read_data_files(filenames, folder))
        print("read data files:", len(filenames))

        return data

    def __init__(self, stats: list[StatsData]):
        self.stats = stats

    def find(self, scale: float, n_bins: int, n_points: int, atom_type: AtomsType):
        stats = self.filter([scale], [n_bins], [n_points], [atom_type]).stats
        if len(stats) != 1:
            raise LookupError(
                f"Should have found 1 element. Found: {len(stats)}. n_points:{n_points} scale:{scale} n_bins:{n_bins} atom_type:{atom_type.name}"
            )
        return stats[0]

    def filter(
        self,
        scale: list[float] = None,
        n_bins: list[int] = None,
        n_points: list[int] = None,
        atom_type: list[AtomsType] = None,
    ):
        stats = []
        for stat in self.stats:
            if (
                (stat.atoms_data.type == AtomsType.Corners)
                and (atom_type is None or AtomsType.Corners in atom_type)
                and (scale is None or stat.scale in scale)
                and (n_bins is None or stat.n_bins in n_bins)
            ):
                stats.append(stat)
                continue
            if (
                (scale is None or stat.scale in scale)
                and (n_bins is None or stat.n_bins in n_bins)
                and (n_points is None or stat.atoms_data.n_points in n_points)
                and (atom_type is None or stat.atoms_data.type in atom_type)
            ):
                stats.append(stat)

        return StatsDataArray(stats)

    #  Error plots for estimator
    def plot_errors_estimator(
        self,
        estimator=EstimatorType.EMMoment,
        scales=[0.05, 0.1, 0.5],
        n_bins_list=[10, 20, 40, 80],
        atom_type=AtomsType.Grid,
        n_points=4,
        legend=False,
        error_type=ErrorType.OT,
        colors=["blue", "green", "red"],
        scale_label=True,
    ):
        saturations = [0.5, 0.7, 0.9, 1]
        for i, scale in enumerate(scales):
            for j, n_bins in enumerate(n_bins_list):
                stat = self.find(scale, n_bins, n_points, atom_type)
                mean_errors = [v[0][error_type.value] for v in stat.stats[estimator]]
                label = f"scale: {scale} " if scale_label else ""
                plt.loglog(
                    stat.t_values,
                    mean_errors,
                    label=label + f"bins: {n_bins}",
                    alpha=saturations[j],
                    linestyle="-",
                    lw=2,
                    c=colors[i],
                )
        if legend:
            plt.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
        plt.xlabel("$t$")
        plt.ylim(1e-5, 5)
        plt.title(f"{estimator.value}")

    # data for time plots
    def get_time_data(
        self, scale=0.05, n_bins=40, atom_type=AtomsType.Grid, n_points=4
    ):
        stats = self.find(scale, n_bins, n_points, atom_type)
        res: dict[str, list] = {}
        for estim, data in stats.stats.items():
            res[estim.name] = [v[2] / stats.num_exp for v in data]

        return res
