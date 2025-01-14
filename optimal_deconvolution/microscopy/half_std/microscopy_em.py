from optimal_deconvolution.microscopy.experiment import MicroscopyExperiment
from optimal_deconvolution.microscopy.microscopy_em import StdMicroscopyEM
from .common import HalfStdDensityMicroscopyCommon


class HalfStdMicroscopyEM(StdMicroscopyEM, HalfStdDensityMicroscopyCommon):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the HalfStdMicroscopyEM class (with halved scale for horizontal axis).

         Parameters:
         - experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)
