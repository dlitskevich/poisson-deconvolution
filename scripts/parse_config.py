import json
import numpy as np
from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy.estimators import EstimatorType
from poisson_deconvolution.normal_distribution import is_covariance_matrix


class EstimationConfig:
    @staticmethod
    def from_path(path: str):
        with open(path, "r") as file:
            spec = json.load(file)
        return parse_config(spec)

    def __init__(
        self,
        estimators: list[EstimatorType],
        num_atoms: list[int],
        scale: float,
        config: Config,
    ):
        self.estimators = estimators
        self.num_atoms = num_atoms
        self.scale = scale
        self.config = config


def parse_estimator_type(spec: dict) -> EstimatorType:
    try:
        spec_type = spec["type"].upper()
        if spec_type in [EstimatorType.Moment.value]:
            return EstimatorType(spec_type)

        value = f"{spec_type} ({spec["init"].lower()})"
        type = EstimatorType(value)
    except (KeyError, ValueError):
        raise ValueError(f"Invalid estimator type specification: {spec}")

    return type


def parse_covariance(cov: list[list[float]]) -> np.ndarray:
    try:
        covariance = np.array(cov)
    except ValueError:
        raise ValueError(f"Invalid covariance specification: {cov}")

    if covariance.shape != (2, 2):
        raise ValueError(f"Invalid covariance shape: {covariance.shape}")

    if not is_covariance_matrix(covariance):
        raise ValueError(f"Invalid covariance matrix: {covariance}")

    return covariance


def parse_config(spec: dict) -> dict:
    try:
        estimators = spec["estimators"]
    except KeyError:
        raise ValueError("Missing required key in config: 'estimators'")
    estimators = [parse_estimator_type(estim_spec) for estim_spec in estimators]

    try:
        num_atoms = spec["num_atoms"]
    except KeyError:
        raise ValueError("Missing required key in config: 'num_atoms'")

    try:
        scale = spec["scale"]
    except KeyError:
        raise ValueError("Missing required key in config: 'scale'")

    try:
        covariance = parse_covariance(spec["covariance"])
    except KeyError:
        covariance = None

    config = Config.std() if covariance is None else Config.normal(covariance)

    return EstimationConfig(estimators, num_atoms, scale, config)
