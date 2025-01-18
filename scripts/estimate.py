import json
import logging
import os
import sys

import numpy as np

from estimation import run_split_estimations, SplitEstimationsResults, mode_from_data
from optimal_deconvolution.microscopy.experiment import MicroscopyExperiment
from optimal_deconvolution.voronoi import VoronoiSplit
from read_dataset import read_dataset


DATASET_DIR = os.path.abspath("../datasets")
OUTPUT_DIR = os.path.abspath("../results")

if __name__ == "__main__":
    dataset = int(sys.argv[1])
    dataset_path = os.path.join(DATASET_DIR, dataset)
    out_path = os.path.join(OUTPUT_DIR, dataset)
    data, estim_config = read_dataset(dataset_path)
    logging.info(f"Successfully read dataset from {dataset_path}")

    scale = estim_config.scale
    t = data.sum()
    estimators = estim_config.estimators
    config = estim_config.config
    exp = MicroscopyExperiment.from_data(data)

    kernel = config.sampler(
        np.ones((1, 2)) / 2, data.shape, scale, 1
    ).sample_convolution()
    init_guess, data_denoised = mode_from_data(exp, 10, kernel)
    logging.info(f"Successfully made init guess")
    t_denoised = data_denoised.sum()

    split = VoronoiSplit.empty(init_guess, data.shape)

    results = SplitEstimationsResults({}, None)

    file_path = os.path.join(out_path, "estimations.json")
    os.mkdir(out_path, exist_ok=True)
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
