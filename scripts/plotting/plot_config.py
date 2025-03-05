import json
import os

from poisson_deconvolution.microscopy.estimators import EstimatorType


class PlotSplitEstimConfig:
    @staticmethod
    def from_json(data: dict):
        return PlotSplitEstimConfig(
            EstimatorType(data["estimator"]),
            data["num_atoms"],
        )

    def __init__(self, estimator: EstimatorType, num_atoms: int):
        self.estimator = estimator
        self.num_atoms = num_atoms

    def to_json(self):
        return {
            "estimator": self.estimator.value,
            "num_atoms": self.num_atoms,
        }


class PlotZoomEstimConfig:
    @staticmethod
    def from_json(data: dict):
        return PlotZoomEstimConfig(
            [EstimatorType(est) for est in data["estimators"]],
            data["num_atoms"],
        )

    def __init__(self, estimators: list[EstimatorType], num_atoms: list):
        self.estimators = estimators
        self.num_atoms = num_atoms

    def to_json(self):
        return {
            "estimators": [est.value for est in self.estimators],
            "num_atoms": self.num_atoms,
        }


class PlotConfig:
    @staticmethod
    def from_path(path: str):
        with open(path, "r") as f:
            return PlotConfig.from_json(json.load(f))

    @staticmethod
    def from_json(data: dict):
        return PlotConfig(
            data["x_lim"],
            data["y_lim"],
            [EstimatorType(est) for est in data["estimators"]],
            data["best_delta"],
            data["deltas"],
            data["num_atoms"],
            PlotSplitEstimConfig.from_json(data["split_estim_config"]),
            PlotZoomEstimConfig.from_json(data["zoom_estim_config"]),
        )

    def __init__(
        self,
        x_lim: list,
        y_lim: list,
        estimators: list[EstimatorType],
        best_delta: float,
        deltas: list,
        num_atoms: list,
        split_estim_config: PlotSplitEstimConfig = None,
        zoom_estim_config: PlotZoomEstimConfig = None,
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.estimators = estimators
        self.best_delta = best_delta
        self.deltas = deltas
        self.num_atoms = num_atoms
        self.split_estim_config = (
            PlotSplitEstimConfig(estimators[0], num_atoms[-1])
            if split_estim_config is None
            else split_estim_config
        )
        self.zoom_estim_config = (
            PlotZoomEstimConfig(estimators[:2], num_atoms)
            if zoom_estim_config is None
            else zoom_estim_config
        )

    def to_json(self):
        return {
            "x_lim": self.x_lim,
            "y_lim": self.y_lim,
            "estimators": [est.value for est in self.estimators],
            "best_delta": self.best_delta,
            "deltas": self.deltas,
            "num_atoms": self.num_atoms,
            "split_estim_config": self.split_estim_config.to_json(),
            "zoom_estim_config": self.zoom_estim_config.to_json(),
        }

    def dump(self, dir: str):
        with open(os.path.join(dir, "plot_config.json"), "w") as f:
            json.dump(self.to_json(), f)
