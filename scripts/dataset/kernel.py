import numpy as np

from poisson_deconvolution.microscopy.config import Config


def get_kernel(shape: tuple[int, int], scale: float, config: Config) -> np.ndarray:
    center = np.array([shape])
    center = center / 2 / np.max(center)
    kernel = config.sampler(center, shape, scale, 1).sample_convolution().data
    return kernel
