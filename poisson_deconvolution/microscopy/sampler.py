import numpy as np
from scipy import stats
from scipy.integrate import dblquad
from poisson_deconvolution.microscopy.experiment import (
    MicroscopyExperiment,
    generate_bins_loc,
)


class MicroscopySampler:
    """
    Class for sampling microscopy experiments.
    """

    data = None

    def __init__(self, atoms: np.ndarray, n_bins: tuple[int] | int, scale=0.01, t=1e8):
        """
        Initialize the MicroscopySampler.

        Parameters:
            atoms (np.ndarray): Array of atom positions.
            n_bins (tuple[int] | int): Number of bins.
            scale (float, optional): Scale parameter for the convolution. Defaults to 0.01.
            t (float, optional): Illumination time. Defaults to 1e8.
        """
        self.atoms = atoms
        self.n_bins = (n_bins,) * 2 if isinstance(n_bins, int) else n_bins
        self.bins_loc = generate_bins_loc(n_bins)
        self.h = self.bins_loc[0, 0]
        self.t = t
        self.scale = scale
        self.convolution = lambda x: stats.norm.pdf(x, atoms, scale).prod(1).mean()

    def sample(self, empirical_t=True) -> MicroscopyExperiment:
        """
        Sample a microscopy experiment.

        Returns:
            MicroscopyExperiment: The sampled microscopy experiment.
        """
        shape = self.n_bins
        self.data = np.array([self.sample_one(i) for i in range(np.prod(shape))])

        t = self.data.sum() if empirical_t else self.t

        return MicroscopyExperiment(self.data.reshape(shape), t=t, atoms=self.atoms)

    def sample_convolution(self) -> MicroscopyExperiment:
        """
        Computes all lambdas.
        Corresponds to the sample divided by t, in case of t -> infinity.

        Returns:
            MicroscopyExperiment: The sampled microscopy experiment. The data is already divided by t, hence t=1.
        """
        shape = self.n_bins
        sampler = lambda bin_id: self.convolution_measure_bin(bin_id)
        self.data = np.array([sampler(i) for i in range(np.prod(shape))])

        return MicroscopyExperiment(self.data.reshape(shape), t=1, atoms=self.atoms)

    def sample_one(self, bin_id: int) -> np.ndarray:
        """
        One sample for the bin of the microscopy experiment.

        Parameters:
            bin_id (int): The bin ID.

        Returns:
            np.ndarray: The sample for the bin.
        """
        # link function g = id
        return np.random.poisson(self.convolution_measure_bin(bin_id) * self.t)

    def convolution_measure_bin(self, bin_id: int) -> float:
        """
        Compute the convolution measure for a given bin.

        Parameters:
            bin_id (int): The bin ID.

        Returns:
            float: The convolution measure for the bin.
        """
        x_bin, y_bin = self.bins_loc[bin_id]

        h = self.h
        integrand = lambda x, y: self.convolution([x, y])
        integral, _ = dblquad(
            integrand,
            y_bin - h,
            y_bin + h,
            lambda x: x_bin - h,
            lambda x: x_bin + h,
        )
        return integral


class StdMicroscopySampler(MicroscopySampler):
    """
    Class for sampling microscopy experiments with standard parameters.
    """

    def __init__(
        self, atoms: np.ndarray, n_bins: tuple[int] | int, scale: float, t=1e8
    ):
        """
        Initialize the StdMicroscopySampler.

        Parameters:
            atoms (np.ndarray): Array of atom positions.
            n_bins (tuple[int] | int): Number of bins.
            scale (float): Scale parameter for the convolution.
            t (float, optional): Illumination time. Defaults to 1e8.
        """
        super().__init__(atoms, n_bins, scale, t)
        self.convolution = lambda x: stats.norm.pdf(x, atoms, scale).prod(1).mean()
        self.x_prob, self.y_prob = self.convolution_all_bins()

    def convolution_all_bins(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the convolution measure for all bins.

        Returns:
            tuple[np.ndarray, np.ndarray]: The x and y probabilities for all bins.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        x_bins = np.linspace(0, n_x / n, n_x + 1)
        y_bins = np.linspace(0, n_y / n, n_y + 1)
        x_prob = np.diff(
            [stats.norm.cdf(x, self.atoms[:, 0], self.scale) for x in x_bins], axis=0
        )
        y_prob = np.diff(
            [stats.norm.cdf(y, self.atoms[:, 1], self.scale) for y in y_bins], axis=0
        )

        return x_prob, y_prob

    def convolution_measure_bin(self, bin_id: int) -> float:
        """
        Compute the convolution measure for a given bin.

        Parameters:
            bin_id (int): The bin ID.

        Returns:
            float: The convolution measure for the bin.
        """
        x_id = bin_id // self.n_bins[1]
        y_id = bin_id % self.n_bins[1]

        return np.mean(self.x_prob[x_id] * self.y_prob[y_id])


class HalfStdMicroscopySampler(StdMicroscopySampler):
    """
    Class for sampling microscopy experiments with halved scale for horizontal axis.
    """

    def __init__(
        self, atoms: np.ndarray, n_bins: tuple[int] | int, scale: float, t=1e8
    ):
        """
        Initialize the HalfStdMicroscopySampler.

        Parameters:
            atoms (np.ndarray): Array of atom positions.
            n_bins (tuple[int] | int): Number of bins.
            scale (float): Scale parameter for the convolution.
            t (float, optional): Illumination time. Defaults to 1e8.
        """
        super().__init__(atoms, n_bins, scale, t)
        self.convolution = lambda x: (_ for _ in ()).throw(
            NotImplementedError("convolution shouldn't be called")
        )

    def convolution_all_bins(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the convolution measure for all bins.

        Returns:
            tuple[np.ndarray, np.ndarray]: The x and y probabilities for all bins.
        """
        n_x, n_y = self.n_bins
        n = max(n_x, n_y)
        x_bins = np.linspace(0, n_x / n, n_x + 1)
        y_bins = np.linspace(0, n_y / n, n_y + 1)
        x_prob = np.diff(
            [stats.norm.cdf(x, self.atoms[:, 0], self.scale / 2) for x in x_bins],
            axis=0,
        )
        y_prob = np.diff(
            [stats.norm.cdf(y, self.atoms[:, 1], self.scale) for y in y_bins], axis=0
        )

        return x_prob, y_prob
