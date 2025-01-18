from poisson_deconvolution.microscopy.density.common import StdDensityMicroscopyCommon
from poisson_deconvolution.microscopy.microscopy_mle import StdMicroscopyMLE


class StdDensityMicroscopyMLE(StdMicroscopyMLE, StdDensityMicroscopyCommon):

    def __init__(self, experiment):
        """
        Initialize the StdDensityMicroscopyMLE class
        (approximate integrals by density multiplied by the size of a bin).

        Parameters:
            experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)
