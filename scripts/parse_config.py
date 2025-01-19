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
        deltas: list[float],
        init_guess: int,
    ):
        self.estimators = estimators
        self.num_atoms = num_atoms
        self.scale = scale
        self.config = config
        self.deltas = deltas
        self.init_guess = init_guess

    def to_json(self) -> dict:
        res = {
            "estimators": [estim.name for estim in self.estimators],
            "num_atoms": self.num_atoms,
            "scale": self.scale,
            "init_guess": self.init_guess,
            "deltas": self.deltas,
        }
        if self.config.inner_scale != 1:
            res["covariance"] = self.config.inner_scale.tolist()

        return res

    def dump(self, path: str):
        with open(path, "w") as file:
            json.dump(self.to_json(), file)


def parse_estimator_type(spec: str) -> EstimatorType:
    try:
        type = EstimatorType(spec)
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
        init_guess = spec["init_guess"]
    except KeyError:
        raise ValueError("Missing required key in config: 'init_guess'")
    try:
        deltas = spec["deltas"]
    except KeyError:
        raise ValueError("Missing required key in config: 'deltas'")

    try:
        covariance = parse_covariance(spec["covariance"])
    except KeyError:
        covariance = None

    config = Config.std() if covariance is None else Config.normal(covariance)

    return EstimationConfig(estimators, num_atoms, scale, config, deltas, init_guess)
