import os
import pathlib

from matplotlib import pyplot as plt

from poisson_deconvolution.microscopy.estimators import EstimatorType
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from scripts.estimation.estimation_results import SplitEstimationsResults
from scripts.plotting.plot_config import PlotConfig
from scripts.plotting.types import COLORS, PREFIX
from scripts.plotting.utils import plot_box
from scripts.dataset.read_dataset import read_dataset
from scripts.dataset.path_constants import DATASET_DIR, OUTPUT_DIR


class PlotResults:
    def __init__(self, dataset: str):
        self.out_path = os.path.join(OUTPUT_DIR, dataset)
        self.img_out_path = os.path.join(self.out_path, "img-zoom")
        # also creates 'out_path'
        pathlib.Path(self.img_out_path).mkdir(parents=True, exist_ok=True)

        self.data, _, self.kernel = read_dataset(self.out_path, no_config=True)
        print(f"Successfully read dataset from {self.out_path}")
        self.plt_config = PlotConfig.from_path(
            os.path.join(self.out_path, "plot_config.json")
        )

        self.x_lim = self.plt_config.x_lim
        self.y_lim = self.plt_config.y_lim
        self.exp = MicroscopyExperiment.from_data(self.data)
        self.best_delta = self.plt_config.best_delta
        self.deltas = self.plt_config.deltas
        best_path = os.path.join(self.out_path, f"estimations_d{self.best_delta}.json")
        self.res_best = SplitEstimationsResults.from_path(best_path)
        self.results = [
            SplitEstimationsResults.from_path(
                os.path.join(self.out_path, f"estimations_d{delta}.json")
            )
            for delta in self.deltas
        ]
        self.estimators = self.plt_config.estimators

    def plot_data(self):
        savepath = os.path.join(self.img_out_path, "data")
        pathlib.Path(savepath).mkdir(parents=True, exist_ok=True)
        for init_points in [None, self.res_best.split.nodes]:
            for is_zoom in [False, True]:
                self._plot_data(is_zoom, init_points)
                name = (
                    "data"
                    + ("_init" if init_points is not None else "")
                    + ("_zoom" if is_zoom else "")
                )
                path = os.path.join(savepath, name + ".pdf")
                plt.savefig(path, bbox_inches="tight", dpi=300)
                plt.close()

    def _plot_data(self, is_zoom, init_points=None):
        exp = self.exp
        x_lim = self.x_lim
        y_lim = self.y_lim

        exp.plot_data("binary")
        plt.xticks([])
        plt.yticks([])
        if is_zoom:
            plt.xlim(x_lim)
            plt.ylim(y_lim)
        else:
            plot_box(x_lim, y_lim)

        if init_points is not None:
            plt.scatter(init_points[:, 0], init_points[:, 1], c="r", s=1)

    def plot_best_estimations(self):
        self._plot_estimations(self.res_best)

    def _plot_estimations(self, split_res: SplitEstimationsResults):
        exp = self.exp
        x_lim = self.x_lim
        y_lim = self.y_lim
        estimations = split_res.denoised
        color = COLORS[1]
        n_col = len(self.plt_config.num_atoms)
        estimators = self.estimators
        for estimator in estimators:
            plt.figure(figsize=(5 * n_col, 2 * 5))
            for j, num_atoms in enumerate(self.plt_config.num_atoms):
                res = estimations[num_atoms]
                plt.subplot(2, n_col, j + 1)
                exp.plot_data("binary")
                plt.xticks([])
                plt.yticks([])
                for e in res:
                    e.plot(estimator, color=color, s=6)
                plot_box(x_lim, y_lim)
                plt.subplot(2, n_col, n_col + j + 1)
                exp.plot_data("binary")
                for e in res:
                    e.plot(estimator, color=color, s=20)
                plt.xlim(x_lim)
                plt.ylim(y_lim)
                plt.xticks([])
                plt.yticks([])

            plt.savefig(
                os.path.join(
                    self.img_out_path,
                    f"estimations_d{split_res.split.delta}_{estimator.name}.pdf",
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

    def plot_splits(self, cmap="gist_ncar"):
        exp = self.exp
        splits = self.results
        plt.figure(figsize=(5 * len(splits), 5))
        for i in range(len(splits)):
            plt.subplot(1, len(splits), i + 1)
            exp.plot_data("binary")
            splits[i].split.plot_split(cmap, components=False, alpha=0.7)
            plot_box(self.x_lim, self.y_lim)
            plt.xticks([])
            plt.yticks([])

        savepath = self.img_out_path
        name = "splits_" + "_".join([str(s.split.delta) for s in splits])
        plt.savefig(os.path.join(savepath, name + ".pdf"), bbox_inches="tight", dpi=600)
        plt.close()

    def plot_split_estimations(self):
        config = self.plt_config.split_estim_config
        self._plot_split_estimations(config.num_atoms, config.estimator)

    def _plot_split_estimations(self, num_atoms=70, estim=EstimatorType.EMMoment):
        exp = self.exp
        splits = self.results
        cmap = "gist_ncar"
        n_col = len(splits)
        plt.figure(figsize=(5 * n_col, 5 * 5))
        for i in range(n_col):
            plt.subplot(5, n_col, i + 1)
            exp.plot_data("binary")
            splits[i].split.plot_split(cmap, components=False, alpha=0.7)
            plot_box(self.x_lim, self.y_lim)
            plt.xticks([])
            plt.yticks([])

            for l, estimations in enumerate(
                [splits[i].estimations[num_atoms], splits[i].denoised[num_atoms]]
            ):
                plt.subplot(5, n_col, (1 + l) * n_col + i + 1)
                exp.plot_data("binary")
                for e in estimations:
                    e.plot(estim, s=10, color=COLORS[l])
                plot_box(self.x_lim, self.y_lim)
                plt.xticks([])
                plt.yticks([])
                plt.subplot(5, n_col, (3 + l) * n_col + i + 1)
                exp.plot_data("binary")
                for e in estimations:
                    e.plot(estim, s=20, color=COLORS[l])
                plt.xlim(self.x_lim)
                plt.ylim(self.y_lim)
                plt.xticks([])
                plt.yticks([])

        savepath = self.img_out_path
        name = f"split_estim_n{num_atoms}_{estim.name}_d"
        name += "_".join([str(s.split.delta) for s in splits])
        plt.savefig(os.path.join(savepath, name + ".pdf"), bbox_inches="tight", dpi=600)
        plt.close()

    def plot_estimated_zoomed(self):
        config = self.plt_config.zoom_estim_config
        if not config.valid:
            print("Invalid zoom estimation configuration")
            return
        num_atoms_list = config.num_atoms
        self._plot_estimated_zoomed(num_atoms_list, config.estimators)

    def _plot_estimated_zoomed(
        self, num_atoms_list: list[int], estimators: list[EstimatorType]
    ):
        exp = self.exp
        splits = self.results
        labels = [est.value.split(" ", 1)[0] for est in estimators]
        labels = labels + [f"{l} (denoised)" for l in labels]
        colors = ["teal", "blue", "darkorange", "r"]
        markers = ["s", "o"]
        sizes = [9, 9]
        n_row = len(num_atoms_list)
        n_col = len(splits)
        fig = plt.figure(figsize=(3 * n_col, 3 * n_row))

        for k, num_atoms in enumerate(num_atoms_list):
            for l in range(n_col):
                plt.subplot(n_row, n_col, k * n_col + l + 1)
                exp.plot_data("binary")
                for j, estimations in enumerate(
                    [splits[l].estimations[num_atoms], splits[l].denoised[num_atoms]]
                ):
                    for s, e in enumerate(estimations):
                        for i, estim in enumerate(estimators):
                            label = (
                                labels[2 * j + i]
                                if (s == 0) and (k == n_row - 1)
                                else None
                            )
                            e.plot(
                                estim,
                                colors[2 * j + i],
                                0.5,
                                sizes[i],
                                label=label,
                                marker=markers[i],
                            )
                plt.xlim(self.x_lim)
                plt.ylim(self.y_lim)
                plt.xticks([])
                plt.yticks([])
        plt.legend(
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(
                0.5 - (1 + fig.subplotpars.wspace) * (n_col - 1) / 2,
                -0.1,
            ),
        )

        savepath = self.img_out_path
        name = f"zoom_estim_{estimators[0].name}_{estimators[1].name}"
        name += "_n" + "_".join(str(n) for n in num_atoms_list)
        name += "_d" + "_".join([str(s.split.delta) for s in splits])
        plt.savefig(os.path.join(savepath, name + ".pdf"), bbox_inches="tight", dpi=600)
        plt.close()
