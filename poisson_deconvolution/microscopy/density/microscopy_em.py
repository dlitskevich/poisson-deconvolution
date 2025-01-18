from poisson_deconvolution.microscopy.density.common import StdDensityMicroscopyCommon
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.microscopy.microscopy_em import StdMicroscopyEM


class StdDensityMicroscopyEM(StdMicroscopyEM, StdDensityMicroscopyCommon):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the StdDensityMicroscopyEM class
        (approximate integrals by density multiplied by the size of a bin).

         Parameters:
         - experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)
