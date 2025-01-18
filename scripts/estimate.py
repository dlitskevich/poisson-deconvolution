import json
import logging
import os
import pathlib
import sys

from matplotlib import pyplot as plt
import numpy as np

from scripts.estimation import (
    run_split_estimations,
    SplitEstimationsResults,
    mode_from_data,
)
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.voronoi import VoronoiSplit
from scripts.plot.plot import plot_estimated
from scripts.read_dataset import read_dataset

ROOT = pathlib.Path(__file__).parent.parent
DATASET_DIR = os.path.join(ROOT, "datasets")
OUTPUT_DIR = os.path.join(ROOT, "results")

if __name__ == "__main__":
    dataset = sys.argv[1]
    dataset_path = os.path.join(DATASET_DIR, dataset)
    out_path = os.path.join(OUTPUT_DIR, dataset)
    img_out_path = os.path.join(out_path, "img")
    pathlib.Path(img_out_path).mkdir(parents=True, exist_ok=True)

    data, estim_config = read_dataset(dataset_path)
    logging.info(f"Successfully read dataset from {dataset_path}")

    scale = estim_config.scale
    t = data.sum()
    estimators = estim_config.estimators
    config = estim_config.config
    exp = MicroscopyExperiment.from_data(data)

    kernel = (
        config.sampler(np.ones((1, 2)) / 2, data.shape, scale, 1)
        .sample_convolution()
        .data
    )
    # TODO: add num of init guess to config
    init_guess, data_denoised = mode_from_data(exp, 4, kernel)
    logging.info(f"Successfully made init guess")
    t_denoised = data_denoised.sum()

    exp_denoised = MicroscopyExperiment.from_data(data_denoised)
    exp_denoised.plot_data("binary")
    plt.savefig(
        os.path.join(img_out_path, "denoised_data.pdf"), bbox_inches="tight", dpi=300
    )

    split = VoronoiSplit.empty(init_guess, data.shape)

    results = SplitEstimationsResults({}, split)

    file_path = os.path.join(out_path, "estimations.json")
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    for num_atoms in estim_config.num_atoms:
        print(
            f"Starting data estimation... {scale} scale {num_atoms} number of components"
        )
        estimation_res = run_split_estimations(
            data, split, estimators, num_atoms, scale, t, config
        )
        denoised_estimation_res = run_split_estimations(
            data_denoised, split, estimators, num_atoms, scale, t_denoised, config
        )

        results.add_result(num_atoms, estimation_res)
        results.add_denoised_result(num_atoms, denoised_estimation_res)

        with open(file_path, "w") as file:
            json.dump(results.to_json(), file)

    plot_estimated(
        exp, results, estim_config.num_atoms, estimators[0], savepath=img_out_path
    )
