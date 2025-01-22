import os
import json
import pathlib
import time
from typing import List

import numpy as np

from poisson_deconvolution.microscopy.config import Config
from scripts.dataset.path_constants import OUTPUT_DIR

from .error_data import StatsData
from poisson_deconvolution.microscopy import (
    AtomsType,
    AtomsData,
    EstimatorType,
    BASE_ESTIMATORS,
    BASE_ATOMS,
    MicroscopyEstimators,
)
from poisson_deconvolution.microscopy.atoms import AtomsData


def run_experiments_convolution(
    config: Config,
    atoms_data: AtomsData,
    estimators: List[EstimatorType],
    scale: float,
    n_bins: int,
):
    atoms = atoms_data.atoms
    sampler = config.sampler(atoms, n_bins, scale, 1)

    exp = sampler.sample_convolution()
    estim = MicroscopyEstimators(exp, scale, config)
    errors = estim.error(estimators)

    return errors


def run_experiments(
    config: Config,
    atoms_data: AtomsData,
    estimators: List[EstimatorType],
    num_experiments: int,
    scale: float,
    t: float,
    n_bins: int,
):
    atoms = atoms_data.atoms
    sampler = config.sampler(atoms, n_bins, scale, t)
    errors = {estimator: [] for estimator in estimators}
    times = {estimator: 0 for estimator in estimators}

    for _ in range(num_experiments):
        exp = sampler.sample(config.empirical_t)
        estim = MicroscopyEstimators(exp, scale, config)
        for estimator, error_time in estim.error_time(estimators).items():
            errors[estimator].append(error_time[0])
            times[estimator] += error_time[1]

    stat_errors = {}
    # print(f"Time elapsed: {sum(times.values())} {times}")

    for estimator, error in errors.items():
        if len(error) > 0:
            stat_errors[estimator] = (
                np.mean(error, axis=0).tolist(),
                np.std(error, axis=0).tolist(),
                times[estimator],
            )

    return stat_errors


def run_experiments_along_t(
    config: Config,
    atoms_data: AtomsData,
    estimators: List[EstimatorType],
    num_experiments: int,
    scale: float,
    n_bins: int,
    t_values=np.logspace(4, 8, 20),
    real_scale=None,
):
    if real_scale is None:
        real_scale = scale

    stats = {estimator: [] for estimator in estimators}
    for t in t_values:
        stat_errors = run_experiments(
            config, atoms_data, estimators, num_experiments, real_scale, t, n_bins
        )
        print({e.value: v for (e, v) in stat_errors.items()})
        for estimator, stat in stat_errors.items():
            stats[estimator].append(stat)

    return StatsData(
        stats,
        t_values.tolist(),
        atoms_data,
        num_experiments,
        scale,
        n_bins,
        real_scale=(
            real_scale.tolist() if isinstance(real_scale, np.ndarray) else real_scale
        ),
    )


def generate_error_data(
    config: Config,
    num_exp: int,
    scale: float,
    n_bins: int,
    estimators=BASE_ESTIMATORS,
    num_points=4,
    t_values=np.logspace(4, 8, 20),
    atoms_types=BASE_ATOMS,
    save_path="std",
):
    abs_path = os.path.join(OUTPUT_DIR, "simulations", save_path, "data")
    pathlib.Path(abs_path).mkdir(parents=True, exist_ok=True)

    real_scale = (
        scale**2 * config.inner_scale
        if isinstance(config.inner_scale, np.ndarray)
        else scale
    )

    errors = []
    for atoms_type in atoms_types:
        print(f"Starting {atoms_type.name}...")
        time_start = time.time()
        atoms_data = AtomsData.from_type(atoms_type, num_points)
        res = run_experiments_along_t(
            config, atoms_data, estimators, num_exp, scale, n_bins, t_values, real_scale
        )
        conv_errors = run_experiments_convolution(
            config, atoms_data, estimators, real_scale, n_bins
        )
        res.conv_errors = conv_errors
        time_elapsed = time.time() - time_start
        res.time_elapsed = time_elapsed

        errors.append(res)
        print(f"Finished {len(errors)}, elapsed time: {time_elapsed:.0f}s")

        new_file_path = abs_path + res.filename + ".json"
        with open(new_file_path, "w") as file:
            json.dump(res.to_json(), file)

    return errors
