import json
import logging
import os
import pathlib

from matplotlib import pyplot as plt

from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from scripts.estimation.estimation_results import SplitEstimationsResults
from scripts.plotting.types import PREFIX
from scripts.plotting.utils import plot_box
from scripts.read_dataset import read_dataset
from scripts.path_constants import DATASET_DIR, OUTPUT_DIR


class PlotConfig:
    @staticmethod
    def from_path(path: str):
        with open(path, "r") as f:
            return PlotConfig.from_json(json.load(f))

    @staticmethod
    def from_json(data: dict):
        return PlotConfig(
            data["x_lim"],
            data["y_lim"],
            data["best_delta"],
            data["deltas"],
            data["num_atoms"],
        )

    def __init__(
        self, x_lim: list, y_lim: list, best_delta: float, deltas: list, num_atoms: list
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.best_delta = best_delta
        self.deltas = deltas
        self.num_atoms = num_atoms


class PlotResults:
    def __init__(self, dataset: str):
        dataset_path = os.path.join(DATASET_DIR, dataset)
        self.out_path = os.path.join(OUTPUT_DIR, dataset)
        self.img_out_path = os.path.join(self.out_path, "img-zoom")
        # also creates 'out_path'
        pathlib.Path(self.img_out_path).mkdir(parents=True, exist_ok=True)

        self.data, _, self.kernel = read_dataset(dataset_path, no_config=True)
        logging.info(f"Successfully read dataset from {dataset_path}")
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
