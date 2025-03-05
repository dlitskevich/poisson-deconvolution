import json
import os
from matplotlib import pyplot as plt
import numpy as np
from poisson_deconvolution.microscopy.estimators import (
    EstimatorType,
)
from poisson_deconvolution.voronoi.voronoi_split import VoronoiSplit


class EstimationResults:
    @staticmethod
    def from_json(data: dict):
        res = EstimationResults()
        for k, v in data.items():
            res.add_result(EstimatorType(k), np.array(v))

        return res

    def __init__(self):
        self.data: dict[EstimatorType, np.ndarray] = {}

    def add_result(self, estimator: EstimatorType, estimated: np.ndarray):
        self.data[estimator] = estimated

    def to_json(self):
        return {k.value: v.tolist() for k, v in self.data.items()}

    def add_from(self, other: "EstimationResults"):
        for k, v in other.data.items():
            self.add_result(k, v)
        return self

    def plot(self, estimator=EstimatorType.Moment, color="r", alpha=0.5, s=2, **kwargs):
        data = self.data[estimator]
        if len(data) == 0:
            return
        plt.scatter(data[:, 0], data[:, 1], color=color, alpha=alpha, s=s, **kwargs)


class SplitEstimationsResults:
    @staticmethod
    def from_path(path: str):
        with open(path, "r") as file:
            raw = json.load(file)
        return SplitEstimationsResults.from_json(raw)

    @staticmethod
    def from_json(raw: dict):
        denoised = {
            int(num): [EstimationResults.from_json(d) for d in val]
            for num, val in raw.get("denoised", {}).items()
        }
        estimations = {
            int(num): [EstimationResults.from_json(d) for d in val]
            for num, val in raw["estimations"].items()
        }
        split = raw["split"]
        split = VoronoiSplit.from_json(split) if split else None
        return SplitEstimationsResults(estimations, split, denoised)

    def __init__(
        self,
        estimations: dict[int, list[EstimationResults]],
        split: VoronoiSplit,
        denoised: dict[int, list[EstimationResults]] = None,
    ):
        self.estimations = estimations
        self.split = split
        self.denoised = denoised if denoised is not None else {}

    def add_result(self, num_atoms: int, estimations: list[EstimationResults]):
        if num_atoms not in self.estimations:
            self.estimations[num_atoms] = estimations
        else:
            for i in range(len(self.estimations[num_atoms])):
                self.estimations[num_atoms][i].add_from(estimations[i])

    def add_denoised_result(self, num_atoms: int, estimations: list[EstimationResults]):
        if num_atoms not in self.denoised:
            self.denoised[num_atoms] = estimations
        else:
            for i in range(len(self.denoised[num_atoms])):
                self.denoised[num_atoms][i].add_from(estimations[i])

    def to_json(self):
        return {
            "estimations": {
                num: [v.to_json() for v in val] for num, val in self.estimations.items()
            },
            "split": self.split.to_json(),
            "denoised": {
                num: [v.to_json() for v in val] for num, val in self.denoised.items()
            },
        }

    def dump_estimations(
        self,
        path: str,
        num_atoms: int,
        estimators: list[EstimatorType],
        scale: float,
        denoised=False,
    ):
        prefix = "_denoised_" if denoised else "_"
        delta = self.split.delta
        for estim in estimators:
            filename = f"estim{prefix}d{delta}_n{num_atoms}_sc{scale}_{estim.name}.json"
            res = []
            estimations = self.denoised if denoised else self.estimations
            for estim_results in estimations[num_atoms]:
                res += estim_results.data[estim].tolist()

            with open(os.path.join(path, filename), "w") as file:
                json.dump(res, file)
