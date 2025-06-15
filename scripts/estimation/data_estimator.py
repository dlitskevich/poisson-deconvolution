import json
import os
import pathlib

from matplotlib import pyplot as plt
import numpy as np

from scripts.dataset.kernel import get_uniform_kernel
from scripts.estimation import (
    run_split_estimations,
    SplitEstimationsResults,
    mode_from_data,
)
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.voronoi import VoronoiSplit
from scripts.plotting.plot_config import PlotConfig
from scripts.plotting.plot import plot_all_data, plot_estimated
from scripts.dataset.read_dataset import save_dataset, read_dataset
from scripts.dataset.path_constants import DATASET_DIR, get_output_path


class DataEstimator:
    def __init__(self, dataset: str, dataset_path: str = None, out_path: str = None):
        if dataset_path is None:
            dataset_path = os.path.join(DATASET_DIR, dataset)
        if out_path is None:
            self.out_path = get_output_path(dataset)
        else:
            self.out_path = out_path
        self.img_out_path = os.path.join(self.out_path, "img")
        # also creates 'out_path'
        pathlib.Path(self.img_out_path).mkdir(parents=True, exist_ok=True)

        self.data, self.estim_config, self.kernel = read_dataset(dataset_path)

        self.n_processes = self.estim_config.n_processes

        self.scale = self.estim_config.scale
        self.estimators = self.estim_config.estimators
        self.config = self.estim_config.config
        if self.estim_config.t:
            self.exp = MicroscopyExperiment(self.data, self.estim_config.t)
        else:
            self.exp = MicroscopyExperiment.from_data(self.data)
        print(f"Using t={self.exp.t}")
        print(f"Use t in moment estimation: {self.config.use_t_in_mom}")
        print(
            f"Using split_num_atoms_factor={self.estim_config.split_num_atoms_factor}"
        )

        self.deltas = self.estim_config.deltas
        PlotConfig(
            [0.17, 0.37],
            [0.61, 0.81],
            self.estimators,
            self.deltas[0],
            self.deltas,
            self.estim_config.num_atoms,
        ).dump(self.out_path)

        if self.kernel is None:
            init_guess_scale = self.estim_config.init_scale
            print(f"Using uniform kernel with scale={init_guess_scale}")
            self.kernel = get_uniform_kernel(
                self.data.shape, init_guess_scale, self.config
            )
        else:
            print(f"Using kernel from {dataset_path} with shape {self.kernel.shape}")

        save_dataset(self.data, self.estim_config, self.kernel, self.out_path)
        print(f"Successfully saved dataset to {self.out_path}")

        init_guess_num = self.estim_config.init_guess
        self.init_guess, data_denoised = mode_from_data(
            self.exp, init_guess_num, self.kernel
        )
        print(f"Successfully made init guess with {init_guess_num} points")
        self.exp_denoised = MicroscopyExperiment.from_data(data_denoised)

        self.plot_all_data()
        self.plot_kernel()
        print(f"Successfully plotted data")

    def run_estimations(self):
        out_path = self.out_path
        n_processes = self.n_processes
        num_atoms_list = self.estim_config.num_atoms
        scale = self.scale
        config = self.config
        estimators = self.estimators
        data = self.data
        t = self.exp.t
        data_denoised = self.exp_denoised.data
        t_denoised = self.exp_denoised.t
        deltas = self.deltas
        split_factor = self.estim_config.split_num_atoms_factor

        for delta in deltas:
            print(f"Starting data estimation... {scale} scale {delta} delta")
            split = VoronoiSplit(self.init_guess, delta, data.shape)
            results = SplitEstimationsResults({}, split)

            file_path = os.path.join(out_path, f"estimations_d{delta}.json")
            for num_atoms in num_atoms_list:
                print(f"{num_atoms} number of components")
                print(f"Estimating with original data")
                estimation_res = run_split_estimations(
                    data,
                    split,
                    estimators,
                    num_atoms,
                    scale,
                    t,
                    config,
                    n_processes,
                    split_num_atoms_factor=split_factor,
                )
                print(f"Estimating with denoised data")
                denoised_estimation_res = run_split_estimations(
                    data_denoised,
                    split,
                    estimators,
                    num_atoms,
                    scale,
                    t_denoised,
                    config,
                    n_processes,
                    split_num_atoms_factor=split_factor,
                )

                results.add_result(num_atoms, estimation_res)
                results.add_denoised_result(num_atoms, denoised_estimation_res)

                with open(file_path, "w") as file:
                    json.dump(results.to_json(), file)
                self.dump_estimations(results, num_atoms, scale)

            self.plot_estimated_data(results)

    def dump_estimations(
        self, results: SplitEstimationsResults, num_atoms: int, scale: float
    ):
        out_dir = os.path.join(self.out_path, "estimations")
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        results.dump_estimations(
            out_dir, num_atoms, self.estimators, scale, denoised=False
        )
        results.dump_estimations(
            out_dir, num_atoms, self.estimators, scale, denoised=True
        )

    def plot_all_data(self):
        savepath = self.img_out_path
        plot_all_data([self.exp, self.exp_denoised], savepath)
        plt.close()

    def plot_estimated_data(self, results: SplitEstimationsResults):
        savepath = self.img_out_path
        plot_estimated(
            self.exp,
            results,
            self.estim_config.num_atoms,
            self.estimators,
            savepath=savepath,
        )
        plt.close()

    def plot_kernel(self):
        n_x, n_y = self.kernel.shape
        n = max(n_x, n_y)
        plt.imshow(
            self.kernel.T,
            origin="lower",
            extent=[0, n_x / n, 0, n_y / n],
            cmap="binary",
        )
        plt.xticks([])
        plt.yticks([])
        path = os.path.join(self.img_out_path, "kernel.pdf")
        plt.savefig(path, bbox_inches="tight", dpi=600)
        plt.close()
