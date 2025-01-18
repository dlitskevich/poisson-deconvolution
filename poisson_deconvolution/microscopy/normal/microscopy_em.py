import numpy as np
from scipy import stats
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.microscopy.microscopy_em import MicroscopyEM
from poisson_deconvolution.microscopy.normal.sampler import NormMicroscopySampler


class NormMicroscopyEM(MicroscopyEM):
    def __init__(self, experiment: MicroscopyExperiment):
        """
        Initialize the NormMicroscopyEM class.

         Parameters:
         - experiment (MicroscopyExperiment): The microscopy experiment object.
        """
        super().__init__(experiment)

    def compute_atom_lambdas(self, mu: np.ndarray, scale, t) -> np.ndarray:
        """
        Compute the lambdas for each atom.

        Parameters:
            mu (np.ndarray): The mu (atoms locations) values.
            scale: The scale value.
            t: The illumination time.

        Returns:
            np.ndarray: The atom lambdas.
        """
        sampler = (
            lambda atom: NormMicroscopySampler(np.array([atom]), self.n_bins, scale, 1)
            .sample_convolution()
            .data.ravel()
        )
        num_atoms = mu.shape[0]
        lambdas = np.array([sampler(atom) for atom in mu]) / num_atoms

        self.lambdas = t * lambdas.T
        return self.lambdas
