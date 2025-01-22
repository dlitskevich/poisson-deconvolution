import json
import logging
import os
import pathlib

from matplotlib import pyplot as plt
import numpy as np

from scripts.estimation import (
    run_split_estimations,
    SplitEstimationsResults,
    mode_from_data,
)
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.voronoi import VoronoiSplit
from scripts.plotting.plot_config import PlotConfig
from scripts.plotting.plot import plot_all_data, plot_estimated
from scripts.dataset.read_dataset import read_dataset
from scripts.dataset.path_constants import DATASET_DIR, OUTPUT_DIR


class DataEstimator:
    def __init__(self, dataset: str):
        dataset_path = os.path.join(DATASET_DIR, dataset)
        self.out_path = os.path.join(OUTPUT_DIR, dataset)
        self.img_out_path = os.path.join(self.out_path, "img")
        # also creates 'out_path'
        pathlib.Path(self.img_out_path).mkdir(parents=True, exist_ok=True)

        self.data, self.estim_config, self.kernel = read_dataset(dataset_path)
        logging.info(f"Successfully read dataset from {dataset_path}")
        self.estim_config.dump(os.path.join(self.out_path, "config.json"))

        self.scale = self.estim_config.scale
        self.estimators = self.estim_config.estimators
        self.config = self.estim_config.config
        self.exp = MicroscopyExperiment.from_data(self.data)
        self.deltas = self.estim_config.deltas
        PlotConfig(
            [0.4, 0.6],
            [0.4, 0.6],
            self.estimators,
            self.deltas[0],
            self.deltas,
            self.estim_config.num_atoms,
        ).dump(self.out_path)

        if self.kernel is None:
            logging.info(
                f"No kernel found in {dataset_path}\nUsing std kernel with scale={self.scale}"
            )
            center = np.array([self.exp.n_bins])
            center = center / 2 / np.max(center)
            self.kernel = (
                self.config.sampler(center, self.data.shape, self.scale, 1)
                .sample_convolution()
                .data
            )
        init_guess_num = self.estim_config.init_guess
        self.init_guess, data_denoised = mode_from_data(
            self.exp, init_guess_num, self.kernel
        )
        logging.info(f"Successfully made init guess")
        self.exp_denoised = MicroscopyExperiment.from_data(data_denoised)

        self.plot_all_data()
        logging.info(f"Successfully plotted data")

    def run_estimations(self):
        out_path = self.out_path
        num_atoms_list = self.estim_config.num_atoms
        scale = self.scale
        config = self.config
        estimators = self.estimators
        data = self.data
        t = self.exp.t
        data_denoised = self.exp_denoised.data
        t_denoised = self.exp_denoised.t
        deltas = self.deltas

        for delta in deltas:
            logging.info(f"Starting data estimation... {scale} scale {delta} delta")
            split = VoronoiSplit(self.init_guess, delta, data.shape)
            results = SplitEstimationsResults({}, split)

            file_path = os.path.join(out_path, f"estimations_d{delta}.json")
            for num_atoms in num_atoms_list:
                logging.info(f"{num_atoms} number of components")
                estimation_res = run_split_estimations(
                    data, split, estimators, num_atoms, scale, t, config
                )
                denoised_estimation_res = run_split_estimations(
                    data_denoised,
                    split,
                    estimators,
                    num_atoms,
                    scale,
                    t_denoised,
                    config,
                )

                results.add_result(num_atoms, estimation_res)
                results.add_denoised_result(num_atoms, denoised_estimation_res)

                with open(file_path, "w") as file:
                    json.dump(results.to_json(), file)
            self.plot_estimated_data(results)

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
