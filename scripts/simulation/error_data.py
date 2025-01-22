from enum import Enum
from typing import Dict, List
import numpy as np
from matplotlib import pyplot as plt


from poisson_deconvolution.microscopy import (
    AtomsType,
    AtomsData,
    EstimatorType,
    BASE_ESTIMATORS,
)
from scripts.simulation.plot_utils import COLORS_ESTIMATORS, LINESTYLE_ESTIMATORS


class ErrorType(Enum):
    OT = 0
    TV = 1


class StatsData:
    def from_json(data):
        data["stats"] = {EstimatorType(k): v for k, v in data["stats"].items()}
        data["conv_errors"] = {
            EstimatorType(k): v for k, v in data["conv_errors"].items()
        }
        data["atoms_data"] = AtomsData.from_json(data["atoms_data"])

        return StatsData(**data)

    def __init__(
        self,
        stats: Dict[EstimatorType, List],
        t_values: List,
        atoms_data: AtomsData,
        num_exp=10,
        scale=0.1,
        n_bins=10,
        time_elapsed=None,
        conv_errors: Dict[EstimatorType, List] = {},
        real_scale=None,
    ):
        self.stats = stats
        self.t_values = t_values
        self.atoms_data = atoms_data
        self.num_exp = num_exp
        self.scale = scale
        self.n_bins = n_bins
        self.time_elapsed = time_elapsed
        self.conv_errors = conv_errors
        self.real_scale = real_scale if real_scale is not None else scale

        self.filename = get_filename(
            num_exp, scale, n_bins, atoms_data.n_points, atoms_data.type.name
        )

    def to_json(self):
        return {
            "atoms_data": self.atoms_data.to_json(),
            "num_exp": self.num_exp,
            "scale": self.scale,
            "n_bins": self.n_bins,
            "t_values": self.t_values,
            "time_elapsed": self.time_elapsed,
            "stats": {k.value: v for k, v in self.stats.items()},
            "conv_errors": {k.value: v for k, v in self.conv_errors.items()},
            "real_scale": self.real_scale,
        }

    def plot(
        self,
        estimators=BASE_ESTIMATORS,
        legend=False,
        title=None,
        error_type=ErrorType.OT,
    ):
        estimators = [EstimatorType(v) for v in self.stats.keys()]

        for estimator in estimators:
            mean_errors = [v[0][error_type.value] for v in self.stats[estimator]]
            plt.loglog(
                self.t_values,
                mean_errors,
                label=estimator.value,
                alpha=0.6,
                linestyle=LINESTYLE_ESTIMATORS[estimator],
                lw=2,
                c=COLORS_ESTIMATORS[estimator],
            )
        plt.xlabel("t")
        plt.ylim(1e-5, 5)

        if title is None:
            title = f"Errors for '{self.atoms_data.name}' scale={self.scale}, n_bins={self.n_bins}"

        plt.title(title)

        if legend:
            plt.legend(loc="center right", bbox_to_anchor=(1.95, 0.5))


def get_filename(
    num_exp: int, scale: float, n_bins: int, num_points: str, atoms_type_name: str
):
    scale_str = str(scale).replace(".", "_")
    return f"err_np{num_points}_rep{num_exp}_t_bins{n_bins}_{atoms_type_name}_sc{scale_str}"
