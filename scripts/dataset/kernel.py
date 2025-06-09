import numpy as np

from poisson_deconvolution.microscopy.config import Config


def get_kernel(shape: tuple[int, int], scale: float, config: Config) -> np.ndarray:
    center = np.array([shape])
    center = center / 2 / np.max(center)
    kernel = config.sampler(center, shape, scale, 1).sample_convolution().data
    return kernel


def get_uniform_kernel(
    shape: tuple[int, int], scale: float, config: Config = None
) -> np.ndarray:
    center = shape[0] // 2, shape[1] // 2
    width = int(np.round(scale * shape[0])), int(np.round(scale * shape[1]))
    kernel = np.zeros(shape)
    print(f"Creating uniform kernel with center {center} and width {width}")
    kernel[
        center[0] - width[0] : center[0] + width[0],
        center[1] - width[1] : center[1] + width[1],
    ] = 1
    return kernel
