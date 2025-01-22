import json
import os
from os import listdir
from os.path import isfile, join

from matplotlib import pyplot as plt
import numpy as np

from poisson_deconvolution.microscopy import AtomsType
from poisson_deconvolution.microscopy.estimators import EstimatorType
from scripts.simulation.plot_utils import (
    COLORS_ESTIMATORS,
    VIS_N_BINS,
    VIS_SCALES,
    set_box_color,
)

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


def get_filenames(rep=NUM_REPS, folder="data/"):
    path = os.path.join(os.path.dirname(__file__), folder)
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    filenames = [f for f in filenames if f.startswith(f"err_") and f"_rep{rep}_" in f]
    return filenames


def read_data_files(filenames: list[str], folder="data/"):
    path = os.path.join(os.path.dirname(__file__), folder)
    stats: list[StatsData] = []
    for f in filenames:
        with open(path + f, "rb") as file:
            data = json.load(file)
            stats.append(StatsData.from_json(data))
    return stats


class StatsDataArray:
    def read_data_files(
        rep=NUM_REPS, folder="data/", check_missing=True, n_points=ALL_N_POINTS
    ):
        filenames = get_filenames(rep, folder)
        data = StatsDataArray(read_data_files(filenames, folder))
        if n_points:
            print(
                "read data files (num points):",
                [len([f for f in filenames if f"_np{i}" in f]) for i in [2, *n_points]],
            )
        else:
            print("read data files:", len(filenames))
        if check_missing:
            print(
                "missing data (atom type):",
                [len(data.missing_data(t, n_points)) for t in AtomsType],
            )

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

    def missing_data(self, atom_type=AtomsType.Corners, n_points=ALL_N_POINTS):
        n_points_list = [2] if atom_type == AtomsType.Corners else n_points
        missing_params = []
        for scale in ALL_SCALES:
            for n_bins in ALL_N_BINS:
                for n_points in n_points_list:
                    if not self.filter(
                        [scale], [n_bins], [n_points], [atom_type]
                    ).stats:
                        missing_params.append(
                            {
                                "scale": scale,
                                "n_bins": n_bins,
                                "n_points": n_points,
                                "atom_type": atom_type.value,
                            }
                        )
        return missing_params

    #  Error plots along atom types
    def plot_errors_atoms_row(
        self,
        scale=0.05,
        n_bins=40,
        atom_types=ALL_ATOM_TYPES,
        n_points=4,
        figsize=4,
        error_type=ErrorType.OT,
        savepath=None,
    ):
        num = len(atom_types)
        plt.figure(figsize=(num * figsize, figsize))
        for i, atom_type in enumerate(atom_types):
            plt.subplot(1, num, i + 1)
            stat = self.find(scale, n_bins, n_points, atom_type)
            stat.plot(
                legend=(i == num - 1),
                title=f"{stat.atoms_data.name}",
                error_type=error_type,
            )
        plt.suptitle(
            f"scale:{stat.scale}, bins:{stat.n_bins}, rep:{stat.num_exp}, time:{stat.time_elapsed:.0f}",
            fontsize=16,
            x=0.85,
        )

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(
                f"{savepath}scale_{str(scale).replace('.','_')}.png",
                bbox_inches="tight",
            )

    def plot_errors_atoms_scale(
        self,
        scale_list=VIS_SCALES,
        n_bins=40,
        atom_types=ALL_ATOM_TYPES[:-1],
        n_points=4,
        figsize=4,
        error_type=ErrorType.OT,
        savepath=None,
    ):
        for scale in scale_list:
            self.plot_errors_atoms_row(
                scale, n_bins, atom_types, n_points, figsize, error_type, savepath
            )

    #  Error plots along number of points
    def plot_errors_points_row(
        self,
        scale=0.05,
        n_bins=40,
        atom_type=AtomsType.Grid,
        n_points_list=ALL_N_POINTS,
        figsize=4,
        error_type=ErrorType.OT,
        savepath=None,
    ):
        num = len(n_points_list)
        plt.figure(figsize=(num * figsize, figsize))
        for i, n_points in enumerate(n_points_list):
            plt.subplot(1, num, i + 1)
            stat = self.find(scale, n_bins, n_points, atom_type)
            stat.plot(
                legend=(i == num - 1),
                title=f"{stat.atoms_data.name}",
                error_type=error_type,
            )
        plt.suptitle(
            f"scale:{stat.scale}, bins:{stat.n_bins}, rep:{stat.num_exp}, time:{stat.time_elapsed:.0f}",
            fontsize=16,
            x=0.85,
        )

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(f"{savepath}points_{atom_type.name}.png", bbox_inches="tight")

    def plot_errors_points_atoms(
        self,
        scale=0.05,
        n_bins=40,
        atom_types=ALL_ATOM_TYPES[:-1],
        n_points_list=ALL_N_POINTS,
        figsize=4,
        error_type=ErrorType.OT,
        savepath=None,
    ):
        for atom_type in atom_types:
            self.plot_errors_points_row(
                scale, n_bins, atom_type, n_points_list, figsize, error_type, savepath
            )

    #  Error plots along estimators type
    def plot_errors_estimators(
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
        plt.xlabel("t")
        plt.ylim(1e-5, 5)
        plt.title(f"{estimator.value}")

    def plot_errors_estimators_row(
        self,
        scales=[0.05, 0.1, 0.5],
        n_bins_list=[10, 20, 40, 80],
        atom_type=AtomsType.Grid,
        n_points=4,
        figsize=4,
        error_type=ErrorType.OT,
        savepath=None,
        estimators=[
            EstimatorType.Moment,
            EstimatorType.EMMoment,
        ],
    ):
        num = len(estimators)
        plt.figure(figsize=(num * figsize, figsize))
        for i, estim in enumerate(estimators):
            plt.subplot(1, num, i + 1)
            self.plot_errors_estimators(
                estim,
                scales,
                n_bins_list,
                atom_type,
                n_points,
                legend=(i == num - 1),
                error_type=error_type,
            )

        plt.suptitle(
            f"{atom_type.name}, {n_points} points",
            fontsize=16,
            horizontalalignment="left",
            x=0,
        )

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(f"{savepath}estim_{atom_type.name}.png", bbox_inches="tight")

    def plot_errors_estimators_atoms(
        self,
        scales=[0.05, 0.1, 0.5],
        n_bins_list=[10, 20, 40, 80],
        atom_types=ALL_ATOM_TYPES[:-1],
        n_points=4,
        figsize=4,
        error_type=ErrorType.OT,
        savepath=None,
    ):
        for atom_type in atom_types:
            self.plot_errors_estimators_row(
                scales, n_bins_list, atom_type, n_points, figsize, error_type, savepath
            )

    # Time plots
    def get_time_data(
        self, scale=0.05, n_bins=40, atom_type=AtomsType.Grid, n_points=4
    ):
        stats = self.find(scale, n_bins, n_points, atom_type)
        res: dict[str, list] = {}
        for estim, data in stats.stats.items():
            res[estim.name] = [v[2] / stats.num_exp for v in data]

        return res

    def plot_time(
        self, scale=0.05, atom_type=AtomsType.Grid, n_points=4, n_bins_list=VIS_N_BINS
    ):
        keys = ["MoM", "EM"]
        res = {k: [] for k in keys}
        for n_bins in n_bins_list:
            data = self.get_time_data(scale, n_bins, atom_type, n_points)
            res_bin = {k: [] for k in keys}
            for name, times in data.items():
                if name.startswith("Mom"):
                    res_bin[keys[0]] += times
                if name.startswith("EM"):
                    res_bin[keys[2]] += times
            for n, v in res_bin.items():
                res[n].append(v)

        x_pos = np.array(range(len(n_bins_list))) * 2.0

        shifts = [-0.6, 0, 0.6]
        legends = []

        for i in range(len(keys)):
            xs = x_pos + shifts[i]
            # reference line
            plt.plot(
                xs,
                np.median(res[keys[i]][0]) * np.array(n_bins_list) / n_bins_list[0],
                ":",
                c=COLORS_ESTIMATORS[keys[i]],
                lw=0.5,
            )
            plt.plot(
                xs,
                np.median(res[keys[i]][-1])
                * (np.array(n_bins_list) / n_bins_list[-1]) ** 2,
                "-",
                c=COLORS_ESTIMATORS[keys[i]],
                lw=0.5,
            )
            # box plot
            bpl = plt.boxplot(res[keys[i]], positions=xs, sym="", widths=0.5)
            set_box_color(bpl, COLORS_ESTIMATORS[keys[i]])
            legends.append(bpl["boxes"][0])

        plt.xticks(x_pos, n_bins_list)
        plt.ylim(1e-4, 1e3)
        plt.yscale("log")
        plt.legend(legends, keys, loc="upper left")
        plt.xlabel("Number of bins")
        plt.ylabel("Elapsed time, s")
        plt.title(f"Scale:{scale}")

    def plot_time_row(
        self,
        atom_type=AtomsType.Grid,
        n_points=4,
        scale_list=VIS_SCALES,
        n_bins_list=VIS_N_BINS,
        figsize=4,
        savepath=None,
    ):
        n_points = 2 if atom_type == AtomsType.Corners else n_points
        x_num = len(scale_list)
        plt.figure(figsize=(x_num * figsize, figsize))
        for i in range(x_num):
            plt.subplot(1, x_num, i + 1)
            scale = scale_list[i]
            self.plot_time(scale, atom_type, n_points, n_bins_list)
        plt.suptitle(
            f" {atom_type.name}, {n_points} points",
            fontsize=16,
            x=0.05,
        )

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(f"{savepath}time_{atom_type.name}.png", bbox_inches="tight")

    def plot_time_all(
        self,
        atom_types=ALL_ATOM_TYPES[:-1],
        n_points=4,
        scale_list=VIS_SCALES,
        n_bins_list=VIS_N_BINS,
        figsize=4,
        savepath=None,
    ):
        for atom_type in atom_types:
            self.plot_time_row(
                atom_type, n_points, scale_list, n_bins_list, figsize, savepath
            )

    # time data for each parameter and estimator
    def _time_data(self):
        res_array = []
        for stat in self.stats:

            res = {}
            # res["filename"]=filename
            res["scale"] = stat.scale
            res["bins"] = stat.n_bins
            res["time"] = stat.time_elapsed
            res["atoms"] = stat.atoms_data.type.name
            res["points"] = stat.atoms_data.n_points
            for key in stat.stats:
                times = [v[2] for v in stat.stats[key]]
                res[key.value] = np.sum(times)
            res_array.append(res)
        return res_array
