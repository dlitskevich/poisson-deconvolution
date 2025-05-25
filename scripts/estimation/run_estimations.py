from functools import partial
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy.estimators import (
    EstimatorType,
    MicroscopyEstimators,
)
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.voronoi.voronoi_split import DataSplit, VoronoiSplit
from .estimation_results import EstimationResults


def run_estimations(
    exp: MicroscopyExperiment,
    estimators: list[EstimatorType],
    num_atoms: int,
    scale: float,
    config: Config,
):
    res = EstimationResults()
    estim = MicroscopyEstimators(exp, scale, config, num_atoms)
    for estimator, data in estim.estimate(estimators).items():
        res.add_result(estimator, data)

    return res


def split_estimation(
    mass: float,
    sample: np.ndarray,
    split: VoronoiSplit,
    estimators: list[EstimatorType],
    num_atoms: int,
    scale: float,
    t: float,
    config: Config,
    split_id: int,
):
    data_split = split.split_data(sample, split_id)
    split_mass = np.sum(data_split.data)
    split_num_atoms = int(np.round(split_mass / mass * num_atoms))
    split_scale = scale / data_split.split_scale
    exp = MicroscopyExperiment(data_split.data, t * split_mass / mass)
    estimations = run_estimations(exp, estimators, split_num_atoms, split_scale, config)
    for est in estimations.data.keys():
        estimations.data[est] = correct_position(estimations.data[est], data_split)
    return estimations


def run_split_estimations(
    sample: np.ndarray,
    split: VoronoiSplit,
    estimators: list[EstimatorType],
    num_atoms: int,
    scale: float,
    t: float,
    config: Config,
    n_processes: int = 1,
) -> list[EstimationResults]:
    mass = np.sum(sample)
    partial_estimation = partial(
        split_estimation, mass, sample, split, estimators, num_atoms, scale, t, config
    )
    n_components = split.n_components
    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(partial_estimation, range(n_components)),
                total=n_components,
            )
        )

    return results


def correct_position(atoms: np.ndarray, data_split: DataSplit):
    if len(atoms) == 0:
        return atoms
    n = max(data_split.n_bins)
    x1, y1 = data_split.min_pos / n
    x2, y2 = (data_split.max_pos + 1) / n

    return atoms * max(x2 - x1, y2 - y1) + [x1, y1]
