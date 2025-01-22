import sys
import numpy as np
from scripts.simulation.generate_error_data import generate_error_data
from poisson_deconvolution.microscopy.atoms import AtomsData, AtomsType
from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy.estimators import EstimatorType

config = Config.std()
num_exp = 20
n_bins_list = [10, 20, 40, 80]
t_values = np.logspace(0, 8, 39)[::2]
scales = [0.01, 0.05, 0.1]
atoms_data_list = [
    AtomsData.from_type(AtomsType.Corners, 2),
    AtomsData.from_type(AtomsType.Clusters6, 2),
    AtomsData.from_type(AtomsType.Grid, 4),
]

if __name__ == "__main__":
    idx = int(sys.argv[1])
    print(f"Running idx={idx}")

    scale = scales[idx % len(scales)]
    n_bins = n_bins_list[idx // len(scales)]

    for atoms_data in atoms_data_list:
        atoms_type = atoms_data.type
        num_points = atoms_data.n_points
        print(
            f"Starting error data generation... {num_exp} experiments {scale} scale {n_bins} bins {num_points} points"
        )

        generate_error_data(
            config,
            num_exp,
            scale,
            n_bins,
            num_points=num_points,
            t_values=t_values,
            atoms_types=[atoms_type],
            estimators=[
                EstimatorType.Moment,
                EstimatorType.EMMoment,
            ],
            save_path="std",
        )
