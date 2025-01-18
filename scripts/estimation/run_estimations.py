from matplotlib import pyplot as plt
import numpy as np
from optimal_deconvolution.microscopy.config import Config
from optimal_deconvolution.microscopy.estimators import (
    EstimatorType,
    MicroscopyEstimators,
)
from optimal_deconvolution.microscopy.experiment import MicroscopyExperiment
from optimal_deconvolution.voronoi.voronoi_split import DataSplit, VoronoiSplit
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


def run_split_estimations(
    sample: np.ndarray,
    split: VoronoiSplit,
    estimators: list[EstimatorType],
    num_atoms: int,
    scale: float,
    t: float,
    config: Config,
    ids: list[int] = None,
) -> list[EstimationResults]:
    res = []
    mass = np.sum(sample)
    ids = ids if ids else range(split.n_components)
    for i in ids:
        print(f"Starting data estimation for split {i}...")
        data_split = split.split_data(sample, i)
        split_mass = np.sum(data_split.data)
        split_num_atoms = int(np.round(split_mass / mass * num_atoms))
        split_scale = scale / data_split.split_scale
        exp = MicroscopyExperiment(data_split.data, split_mass)
        estimations = run_estimations(
            exp, estimators, split_num_atoms, split_scale, config
        )
        for est in estimations.data.keys():
            estimations.data[est] = correct_position(estimations.data[est], data_split)
        res.append(estimations)

    return res


def correct_position(atoms: np.ndarray, data_split: DataSplit):
    if len(atoms) == 0:
        return atoms
    n = max(data_split.n_bins)
    x1, y1 = data_split.min_pos / n
    x2, y2 = (data_split.max_pos + 1) / n

    return atoms * max(x2 - x1, y2 - y1) + [x1, y1]
