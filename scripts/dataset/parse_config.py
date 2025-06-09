import json
import os
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
        init_scale: float,
        n_processes: int,
        split_num_atoms_factor: int = 1,
        t: float | None = None,
        scale_data_by: float | None = None,
        use_t_in_mom: bool | None = None,
    ):
        self.estimators = estimators
        self.num_atoms = num_atoms
        self.scale = scale
        self.config = config
        self.deltas = deltas
        self.init_guess = init_guess
        self.init_scale = init_scale
        self.n_processes = n_processes

        self.split_num_atoms_factor = split_num_atoms_factor

        self.t = t
        self.scale_data_by = scale_data_by
        self.use_t_in_mom = use_t_in_mom

    def to_json(self) -> dict:
        res = {
            "estimators": [estim.value for estim in self.estimators],
            "num_atoms": self.num_atoms,
            "scale": self.scale,
            "init_guess": self.init_guess,
            "init_scale": self.init_scale,
            "deltas": self.deltas,
            "n_processes": self.n_processes,
            "split_num_atoms_factor": self.split_num_atoms_factor,
        }
        if self.config.inner_scale != 1:
            res["covariance"] = self.config.inner_scale.tolist()

        if self.t is not None:
            res["t"] = self.t
        if self.scale_data_by is not None:
            res["scale_data_by"] = self.scale_data_by
        if self.use_t_in_mom is not None:
            res["use_t_in_mom"] = self.use_t_in_mom

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
        init_scale = spec["init_scale"]
    except KeyError:
        print(
            "Missing required key in config: 'init_scale', using 'scale' as 'init_scale'"
        )
        init_scale = scale
    try:
        deltas = spec["deltas"]
    except KeyError:
        raise ValueError("Missing required key in config: 'deltas'")

    try:
        covariance = parse_covariance(spec["covariance"])
    except KeyError:
        covariance = None

    split_num_atoms_factor = spec.get("split_num_atoms_factor", 1)

    t = spec.get("t", None)
    if t is not None:
        t = float(t)
        if t <= 0:
            raise ValueError(f"Invalid value for 't': {t}. It must be positive.")
    scale_data_by = spec.get("scale_data_by", None)
    if scale_data_by is not None:
        scale_data_by = float(scale_data_by)
        if scale_data_by <= 0:
            raise ValueError(
                f"Invalid value for 'scale_data_by': {scale_data_by}. It must be positive."
            )

    n_processes = spec.get("n_processes", 1)
    cpu_nodes = os.cpu_count()
    print(f"Using {n_processes} processes out of {cpu_nodes} available")
    if n_processes > cpu_nodes:
        raise ValueError(
            f"Number of processes ({n_processes}) exceeds available CPU nodes ({cpu_nodes})"
        )

    config = Config.std() if covariance is None else Config.normal(covariance)

    use_t_in_mom = spec.get("use_t_in_mom", None)
    if use_t_in_mom is not None:
        use_t_in_mom = bool(use_t_in_mom)
        config.use_t_in_mom = use_t_in_mom

    return EstimationConfig(
        estimators,
        num_atoms,
        scale,
        config,
        deltas,
        init_guess,
        init_scale,
        n_processes,
        split_num_atoms_factor,
        t,
        scale_data_by,
        use_t_in_mom,
    )
