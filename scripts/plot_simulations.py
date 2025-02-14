import os
import pathlib
import sys

from matplotlib import pyplot as plt

from poisson_deconvolution.microscopy.atoms import AtomsType
from poisson_deconvolution.microscopy.config import Config
from scripts.dataset.path_constants import SIMULATIONS_DIR
from scripts.simulation.data_utils import StatsDataArray
from scripts.simulation.plot import plot_data, plot_errors_bin_scale, plot_time_types
from scripts.simulation.plot_utils import plot_estimations
from scripts.simulation.simulation_config import SimulationConfig

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
    }
)

if __name__ == "__main__":
    sim_name = sys.argv[1]
    config_dir = os.path.join(SIMULATIONS_DIR, sim_name)
    sim_config = SimulationConfig.from_dir(config_dir)
    print(f"Loaded config from {config_dir}")

    img_path = os.path.join(config_dir, "img")
    pathlib.Path(img_path).mkdir(parents=True, exist_ok=True)

    num_exp = sim_config.num_experiments
    n_bins_list = sim_config.n_bins
    scales = sim_config.scales
    atoms_data_list = sim_config.atoms_data
    atoms_types = [a.type for a in atoms_data_list]

    data_dir = os.path.join(config_dir, "data")
    data = StatsDataArray.read_data_files(rep=num_exp, folder=data_dir)

    plot_data(atoms_data_list, img_path)

    plot_errors_bin_scale(data, atoms_types, n_bins_list, scales, img_path)

    plot_time_types(data, atoms_types, n_bins_list, scales, img_path)

    # Individual plots
    # for atoms_type in [AtomsType.Grid, AtomsType.Corners]:
    #     plot_estimations(
    #         Config.std(),
    #         scales[1],
    #         [1e8, 1e5, 1e3],
    #         n_bins_list[1:],
    #         atoms_type,
    #         3.5,
    #         img_path,
    #     )
