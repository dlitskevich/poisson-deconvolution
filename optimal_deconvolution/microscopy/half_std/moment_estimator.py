from optimal_deconvolution.microscopy.moment_estimator import (
    StdMicroscopyMomentEstimator,
)
from optimal_deconvolution.moment_estimator.complex_moment_approximator import (
    HalfStdComplexNormMomentApproximator,
)
from optimal_deconvolution.microscopy import MicroscopyExperiment


class HalfStdMicroscopyMomentEstimator(StdMicroscopyMomentEstimator):
    """
    Class for estimating moments and atoms in microscopy experiments with halved scale for horizontal axis.
    """

    atoms = None

    def __init__(self, experiment: MicroscopyExperiment, max_atoms: int, scale: float):
        """
        Initialize the HalfStdMicroscopyMomentEstimator.

        Parameters:
        - experiment (MicroscopyExperiment): The microscopy experiment object.
        - max_atoms (int): The maximum number of atoms to consider.
        - scale (float): The scale of Gaussian distribution (kernel).
        """
        super().__init__(experiment, max_atoms)
        self.moment_estimator = HalfStdComplexNormMomentApproximator(max_atoms, scale)
