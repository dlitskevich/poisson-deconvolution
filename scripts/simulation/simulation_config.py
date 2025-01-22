import json
import os

import numpy as np

from poisson_deconvolution.microscopy.atoms import AtomsData, AtomsType


class TValuesSpec:
    @staticmethod
    def from_json(spec: dict):
        return TValuesSpec(spec["start"], spec["stop"], spec["num"], spec["step"])

    def __init__(self, start: int, stop: int, num: int, step: int):
        self.start = start
        self.stop = stop
        self.num = num
        self.step = step

    @property
    def values(self) -> np.ndarray:
        return np.logspace(self.start, self.stop, self.num)[:: self.step]


class SimulationConfig:
    @staticmethod
    def from_dir(dir: str):
        return SimulationConfig.from_path(os.path.join(dir, "config.json"))

    @staticmethod
    def from_path(path: str):
        with open(path, "r") as file:
            spec = json.load(file)
        return SimulationConfig.from_json(spec)

    @staticmethod
    def from_json(spec: dict):
        return SimulationConfig(
            spec["num_experiments"],
            spec["n_bins"],
            TValuesSpec.from_json(spec["t_values"]),
            spec["scales"],
            [
                AtomsData.from_type(AtomsType(s["type"]), s["num_points"])
                for s in spec["atoms_data"]
            ],
        )

    def __init__(
        self,
        num_experiments: int,
        n_bins: list[int],
        t_values_spec: TValuesSpec,
        scales: list[float],
        atoms_data: list[AtomsData],
    ):
        self.num_experiments = num_experiments
        self.n_bins = n_bins
        self.t_values_spec = t_values_spec
        self.scales = scales
        self.atoms_data = atoms_data

    @property
    def t_values(self) -> np.ndarray:
        return self.t_values_spec.values
