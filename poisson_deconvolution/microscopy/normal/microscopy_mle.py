import numpy as np

from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.microscopy.microscopy_mle import MicroscopyMLE
from poisson_deconvolution.microscopy.normal.sampler import NormMicroscopySampler


class NormMicroscopyMLE(MicroscopyMLE):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the NormMicroscopyMLE.

        Parameters:
        - experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)

    def compute_lambdas(self, mu: np.ndarray, scale, t):
        """
        Computes the lambdas.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale (float): The scale value.
            t (float): The illumination time.

        Returns:
            np.ndarray: The lambdas array.
        """
        exp = NormMicroscopySampler(mu, self.n_bins, scale, 1).sample_convolution()

        self.lambdas = t * exp.data.ravel()
        return self.lambdas
