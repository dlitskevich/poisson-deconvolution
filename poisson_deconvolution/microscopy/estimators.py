from collections import defaultdict
import time
from enum import Enum
from typing import List, Dict, Tuple

from matplotlib import pyplot as plt
import numpy as np


from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy import (
    MicroscopySampler,
    MicroscopyExperiment,
    mode_from_std_data,
    from_moment_estimator,
    wasserstein_distance,
)


class EstimatorType(Enum):
    Mode = "Mode"
    Moment = "Moment"
    MLEMode = "MLE (mode)"
    MLEMoment = "MLE (moment)"
    EMMode = "EM (mode)"
    EMMoment = "EM (moment)"


BASE_ESTIMATORS = [
    EstimatorType.Mode,
    EstimatorType.Moment,
    EstimatorType.MLEMode,
    EstimatorType.MLEMoment,
    EstimatorType.EMMode,
    EstimatorType.EMMoment,
]


class MicroscopyEstimators:
    def __init__(
        self,
        experiment: MicroscopyExperiment,
        scale: float,
        config: Config,
        num_atoms=None,
    ):
        """
        Class for performing various microscopy estimations.

        Parameters:
            experiment (MicroscopyExperiment): The microscopy experiment object.
            scale (float): The scale parameter for the estimation.

        """
        self.experiment = experiment
        self.atoms = experiment.atoms
        if self.atoms is not None:
            self.num_atoms = len(self.atoms)
        else:
            assert num_atoms is not None, "Number of atoms must be specified"
            self.num_atoms = num_atoms
        self.scale = scale
        self.config = config

        self.estimators = {
            EstimatorType.Mode: self.mode,
            EstimatorType.Moment: self.moment,
            EstimatorType.MLEMode: self.MLEMode,
            EstimatorType.MLEMoment: self.MLEMoment,
            EstimatorType.EMMode: self.EMMode,
            EstimatorType.EMMoment: self.EMMoment,
        }
        if self.num_atoms == 0:
            self.estimators = defaultdict(lambda: lambda: np.array([]))

    def mode(self) -> np.ndarray:
        """
        Perform mode estimation.

        Returns:
            np.ndarray: The estimated atoms locations.

        """
        mu = mode_from_std_data(
            self.experiment, self.num_atoms, self.scale, self.config.sampler
        )
        return mu

    def moment(self) -> np.ndarray:
        """
        Perform moment estimation.

        Returns:
            np.ndarray: The estimated atoms locations.

        """
        estim = self.config.moment(
            self.experiment, self.num_atoms, self.scale, self.config.use_t_in_mom
        )
        return estim.estimate(self.num_atoms, 0)

    def MLEMode(self) -> np.ndarray:
        """
        Perform MLE estimation using mode.

        Returns:
            np.ndarray: The estimated atoms locations.

        """
        mle = self.config.mle(self.experiment)
        mu_0 = mode_from_std_data(
            self.experiment, self.num_atoms, self.scale, self.config.sampler
        )
        return mle.estimate(mu_0, self.scale)

    def MLEMoment(self) -> np.ndarray:
        """
        Perform MLE estimation using moments.

        Returns:
            np.ndarray: The estimated atoms locations.

        """
        mle = self.config.mle(self.experiment)
        mu_0 = from_moment_estimator(
            self.experiment,
            self.num_atoms,
            self.scale,
            0,
            self.config.moment,
            self.config.use_t_in_mom,
        )
        return mle.estimate(mu_0, self.scale)

    def EMMode(self) -> np.ndarray:
        """
        Perform EM estimation using mode.

        Returns:
            np.ndarray: The estimated atoms locations.

        """
        em = self.config.em(self.experiment)
        mu_0 = mode_from_std_data(
            self.experiment, self.num_atoms, self.scale, self.config.sampler
        )
        return em.estimate(mu_0, self.scale)

    def EMMoment(self) -> np.ndarray:
        """
        Perform EM estimation using moments method.

        Returns:
            np.ndarray: The estimated atoms locations.

        """
        em = self.config.em(self.experiment)
        mu_0 = from_moment_estimator(
            self.experiment,
            self.num_atoms,
            self.scale,
            0,
            self.config.moment,
            self.config.use_t_in_mom,
        )
        return em.estimate(mu_0, self.scale)

    def estimate(
        self, estimator: List[EstimatorType]
    ) -> Dict[EstimatorType, np.ndarray]:
        """
        Perform estimation using the specified estimators.

        Parameters:
            estimator (List[EstimatorType]): The list of estimators to use.

        Returns:
            Dict[EstimatorType, np.ndarray]: A dictionary containing the estimated moments for each estimator.

        """
        return {estim: self.estimators[estim]() for estim in estimator}

    def timed_estimate(
        self, estimator: List[EstimatorType]
    ) -> Dict[EstimatorType, Tuple[List[float], float]]:
        """
        Perform estimation and record the execution time for the specified estimators.

        Parameters:
            estimator (List[EstimatorType]): The list of estimators to calculate errors and execution time for.

        Returns:
            Dict[EstimatorType, Tuple[List[float], float]]: A dictionary containing the errors and execution time for each estimator.

        """
        res = {}

        for est in estimator:
            time_start = time.time()
            estimated = self.estimators[est]()
            time_end = time.time()

            res[est] = {"estimated": estimated, "time": time_end - time_start}

        return res

    def error(self, estimator: List[EstimatorType]) -> Dict[EstimatorType, List[float]]:
        """
        Calculate the OT errors for the specified estimators.

        Parameters:
            estimator (List[EstimatorType]): The list of estimators to calculate errors for.

        Returns:
            Dict[EstimatorType, List[float]]: A dictionary containing the OT errors for each estimator.

        """
        errors = {}
        for est in estimator:
            estimated = self.estimators[est]()
            errors[est] = [
                wasserstein_distance(estimated, self.atoms),
                tv_error(
                    estimated,
                    self.atoms,
                    self.experiment.n_bins,
                    self.scale,
                    self.config.sampler,
                ),
            ]
        return errors

    def tv_error(self, estimator: List[EstimatorType]) -> Dict[EstimatorType, float]:
        """
        Calculate the total variation (TV) errors for the specified estimators.

        Parameters:
            estimator (List[EstimatorType]): The list of estimators to calculate TV errors for.

        Returns:
            Dict[EstimatorType, float]: A dictionary containing the TV errors for each estimator.

        """
        errors = {}
        for est in estimator:
            estimated = self.estimators[est]()
            errors[est] = tv_error(
                estimated, self.atoms, self.experiment.n_bins, self.scale
            )
        return errors

    def error_time(
        self, estimator: List[EstimatorType]
    ) -> Dict[EstimatorType, Tuple[List[float], float]]:
        """
        Calculate the errors and execution time for the specified estimators.

        Parameters:
            estimator (List[EstimatorType]): The list of estimators to calculate errors and execution time for.

        Returns:
            Dict[EstimatorType, Tuple[List[float], float]]: A dictionary containing the errors and execution time for each estimator.

        """
        error_time = {}

        for est in estimator:
            time_start = time.time()
            estimated = self.estimators[est]()
            time_end = time.time()

            error_time[est] = [
                [
                    wasserstein_distance(estimated, self.atoms),
                    tv_error(
                        estimated,
                        self.atoms,
                        self.experiment.n_bins,
                        self.scale,
                        self.config.sampler,
                    ),
                ],
                time_end - time_start,
            ]

        return error_time

    def plot(
        self,
        estimators: List[EstimatorType] = [
            EstimatorType.EMMoment,
            EstimatorType.Moment,
        ],
        colors: List[str] = ["red", "orange"],
        sc: List[int] = [70, 50],
        markers: List[str] = ["o", "s"],
        names: List[str] = ["EM", "MoM"],
    ):
        """
        Plot the estimations.

        Parameters:
            estimators (List[EstimatorType], optional): The list of estimators to plot. Defaults to [EstimatorType.MLEMoment, EstimatorType.EMMoment, EstimatorType.Moment].
            colors (List[str], optional): The colors for each estimator. Defaults to ["red", "orange"].
            sc (List[int], optional): The sizes for each estimator. Defaults to [70, 60].
            markers (List[str], optional): The markers for each estimator. Defaults to ["o", "s"].
            names (List[str], optional): The names for each estimator. Defaults to ["EM", "MoM"].

        """
        estimations = [self.estimators[estim]() for estim in estimators]
        for i, estim in enumerate(estimations):
            err = (
                wasserstein_distance(estim, self.atoms)
                if self.atoms is not None
                else None
            )
            err_str = f" ({err:.1e})" if err is not None else ""
            # tv_err = tv_error(estim, self.atoms, self.experiment.n_bins, self.scal,self.config.samplere)
            plt.scatter(
                estim[:, 0],
                estim[:, 1],
                c=colors[i],
                linewidths=0.25,
                edgecolors="grey",
                s=sc[i],
                marker=markers[i],
                label=f"{names[i]}" + err_str,
                # label=f"{names[i]} ({err:.1e}/{tv_err:.0e})",
            )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=3,
        )
        self.experiment.plot(c="black")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 1)
        plt.ylim(0, 1)


def tv_error(
    true_atoms: np.ndarray,
    atoms: np.ndarray,
    n_bins: tuple[int] | int,
    scale: float,
    sampler: type[MicroscopySampler] = None,
) -> float:
    """
    Compute the total variation error between two microscopy experiments.

    Parameters:
        true_atoms (np.ndarray): Array of true atom positions.
        atoms (np.ndarray): Array of estimated atom positions.
        n_bins (tuple[int] | int): Number of bins.
        scale (float): Scale parameter for the convolution.
        t (float, optional): Illumination time. Defaults to 1.
        sampler (class): The sampler class to use.

    Returns:
        float: The total variation error.
    """
    conv_a = sampler(true_atoms, n_bins, scale).sample_convolution().data
    conv_b = sampler(atoms, n_bins, scale).sample_convolution().data

    return np.sum(np.abs(conv_a - conv_b))
