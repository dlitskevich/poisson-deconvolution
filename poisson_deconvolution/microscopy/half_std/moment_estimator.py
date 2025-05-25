from poisson_deconvolution.microscopy.moment_estimator import (
    StdMicroscopyMomentEstimator,
)
from poisson_deconvolution.moment_estimator.complex_moment_approximator import (
    HalfStdComplexNormMomentApproximator,
)
from poisson_deconvolution.microscopy import MicroscopyExperiment


class HalfStdMicroscopyMomentEstimator(StdMicroscopyMomentEstimator):
    """
    Class for estimating moments and atoms in microscopy experiments with halved scale for horizontal axis.
    """

    atoms = None

    def __init__(
        self, experiment: MicroscopyExperiment, max_atoms: int, scale: float, **kwargs
    ):
        """
        Initialize the HalfStdMicroscopyMomentEstimator.

        Parameters:
        - experiment (MicroscopyExperiment): The microscopy experiment object.
        - max_atoms (int): The maximum number of atoms to consider.
        - scale (float): The scale of Gaussian distribution (kernel).
        """
        super().__init__(experiment, max_atoms, **kwargs)
        self.moment_estimator = HalfStdComplexNormMomentApproximator(max_atoms, scale)
