from optimal_deconvolution.microscopy.moment_estimator import (
    StdMicroscopyMomentEstimator,
)
from optimal_deconvolution.moment_estimator.complex_moment_approximator import (
    ComplexNormMomentApproximator,
)
from optimal_deconvolution.microscopy import MicroscopyExperiment


class NormMicroscopyMomentEstimator(StdMicroscopyMomentEstimator):
    """
    Class for estimating moments and atoms in microscopy experiments.
    """

    atoms = None

    def __init__(self, experiment: MicroscopyExperiment, max_atoms: int, scale: float):
        """
        Initialize the NormMicroscopyMomentEstimator.

        Parameters:
        - experiment (MicroscopyExperiment): The microscopy experiment object.
        - max_atoms (int): The maximum number of atoms to consider.
        - scale (float): The scale of Gaussian distribution (kernel).
        """
        super().__init__(experiment, max_atoms)
        self.moment_estimator = ComplexNormMomentApproximator(max_atoms, scale)
