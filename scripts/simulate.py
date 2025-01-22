import os
import sys

from scripts.dataset.path_constants import SIMULATIONS_DIR
from scripts.simulation.generate_error_data import generate_error_data
from poisson_deconvolution.microscopy.atoms import AtomsData, AtomsType
from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy.estimators import EstimatorType
from scripts.simulation.simulation_config import SimulationConfig

config = Config.std()


if __name__ == "__main__":
    sim_name = sys.argv[1]
    idx = int(sys.argv[2])
    print(f"Running idx={idx}")
    config_path = os.path.join(SIMULATIONS_DIR, sim_name)
    sim_config = SimulationConfig.from_dir(config_path)
    print(f"Loaded config from {config_path}")

    num_exp = sim_config.num_experiments
    n_bins_list = sim_config.n_bins
    t_values = sim_config.t_values
    scales = sim_config.scales
    atoms_data_list = sim_config.atoms_data

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
            estimators=[
                EstimatorType.Moment,
                EstimatorType.EMMoment,
            ],
            num_points=num_points,
            t_values=t_values,
            atoms_types=[atoms_type],
            sim_name=sim_name,
        )
