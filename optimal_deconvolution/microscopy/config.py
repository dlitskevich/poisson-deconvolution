import numpy as np
from optimal_deconvolution.microscopy.density.microscopy_em import (
    StdDensityMicroscopyEM,
)
from optimal_deconvolution.microscopy.density.microscopy_mle import (
    StdDensityMicroscopyMLE,
)
from optimal_deconvolution.microscopy.half_std.microscopy_em import HalfStdMicroscopyEM
from optimal_deconvolution.microscopy.half_std.microscopy_mle import (
    HalfStdMicroscopyMLE,
)
from optimal_deconvolution.microscopy.half_std.moment_estimator import (
    HalfStdMicroscopyMomentEstimator,
)
from optimal_deconvolution.microscopy.microscopy_em import StdMicroscopyEM
from optimal_deconvolution.microscopy.microscopy_mle import StdMicroscopyMLE
from optimal_deconvolution.microscopy.normal.microscopy_em import NormMicroscopyEM
from optimal_deconvolution.microscopy.normal.microscopy_mle import NormMicroscopyMLE
from optimal_deconvolution.microscopy.normal.moment_estimator import (
    NormMicroscopyMomentEstimator,
)
from optimal_deconvolution.microscopy.normal.sampler import NormMicroscopySampler
from optimal_deconvolution.microscopy.sampler import (
    HalfStdMicroscopySampler,
    MicroscopySampler,
    StdMicroscopySampler,
)

from optimal_deconvolution.microscopy.moment_estimator import (
    StdMicroscopyMomentEstimator,
)


class Config:
    @staticmethod
    def from_str(config_str: str):
        configs = {
            "std": lambda: Config.std(),
            "std-density": lambda: Config.std(True),
            "half_std": lambda: Config.half_std(),
            "ex_normal": lambda: Config.ex_normal(),
            "normal": lambda: Config.normal(),
        }
        try:
            return configs[config_str]()
        except KeyError:
            raise ValueError(
                f"Unknown config string: {config_str}.\n Possible values: {list(configs.keys())}"
            )

    @staticmethod
    def std(density_approx: bool = False):
        if density_approx:
            return Config(
                sampler=StdMicroscopySampler,
                moment=StdMicroscopyMomentEstimator,
                mle=StdDensityMicroscopyMLE,
                em=StdDensityMicroscopyEM,
                density_approx=True,
            )
        return Config(
            sampler=StdMicroscopySampler,
            moment=StdMicroscopyMomentEstimator,
            mle=StdMicroscopyMLE,
            em=StdMicroscopyEM,
        )

    @staticmethod
    def half_std():
        return Config(
            sampler=HalfStdMicroscopySampler,
            moment=HalfStdMicroscopyMomentEstimator,
            mle=HalfStdMicroscopyMLE,
            em=HalfStdMicroscopyEM,
        )

    @staticmethod
    def ex_normal():
        return Config.normal(np.array([[1, -0.3], [-0.3, 0.25]]))

    @staticmethod
    def normal(inner_scale: np.ndarray):
        return Config(
            sampler=NormMicroscopySampler,
            moment=NormMicroscopyMomentEstimator,
            mle=NormMicroscopyMLE,
            em=NormMicroscopyEM,
            inner_scale=inner_scale,
        )

    def __init__(
        self,
        sampler: type[MicroscopySampler],
        moment: type[StdMicroscopyMomentEstimator],
        mle: type[StdMicroscopyMLE],
        em: type[StdMicroscopyEM],
        inner_scale: float = 1,
        density_approx: bool = False,
    ):
        """
        Initializes the Config class.

        Parameters:
            sampler (type[MicroscopySampler]): The sampler class.
            moment (type[StdMicroscopyMomentEstimator]): The moment estimator class.
            mle (type[StdMicroscopyMLE]): The MLE class.
            em (type[StdMicroscopyEM]): The EM class.
            inner_scale (float, optional): The inner scale. Defaults to 1.
            density_approx (bool, optional): Whether to use density approximation of probabilities. Defaults to False.
        """
        self.sampler = sampler
        self.moment = moment
        self.mle = mle
        self.em = em
        self.inner_scale = inner_scale
        self.density_approx = density_approx
        self.empirical_t = True
