from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.microscopy.microscopy_mle import StdMicroscopyMLE
from .common import HalfStdDensityMicroscopyCommon


class HalfStdMicroscopyMLE(StdMicroscopyMLE, HalfStdDensityMicroscopyCommon):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the HalfStdMicroscopyMLE (with halved scale for horizontal axis).

        Parameters:
        - experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)
