import os
import pathlib
import sys

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from poisson_deconvolution.microscopy.config import Config
from poisson_deconvolution.microscopy.experiment import MicroscopyExperiment
from poisson_deconvolution.voronoi import VoronoiSplit
from scripts.dataset.kernel import get_kernel
from scripts.dataset.read_dataset import read_dataset
from scripts.dataset.path_constants import DATASET_DIR, OUTPUT_DIR
from scripts.estimation.mode import mode_from_data
from scripts.params.search_config import read_search_file

if __name__ == "__main__":
    dataset = sys.argv[1]

    dataset_path = os.path.join(DATASET_DIR, dataset)
    out_path = os.path.join(OUTPUT_DIR, dataset)
    img_out_path = os.path.join(out_path, "img", "search")
    # also creates 'out_path'
    pathlib.Path(img_out_path).mkdir(parents=True, exist_ok=True)

    search_config = read_search_file(dataset_path)
    scales = search_config.scales
    init_guesses = search_config.init_guesses
    deltas = search_config.deltas

    config = Config.std()
    data, _, kernel = read_dataset(dataset_path, no_config=True)
    exp = MicroscopyExperiment.from_data(data)

    if kernel is not None:
        print(f"Kernel found in {dataset_path}, the scale will be ignored")
        scales = [1]

    for scale in scales:
        kernel = get_kernel(data.shape, scale, config)
        frac = data.shape[1] / data.shape[0]
        n_rows = 2 + len(deltas)
        n_cols = len(init_guesses)
        plt.figure(figsize=(4 * n_cols, n_rows * 4 * frac))
        for j, init_guess in enumerate(init_guesses):
            guesses, data_denoised = mode_from_data(exp, init_guess, kernel)
            exp_diff = MicroscopyExperiment.from_data(data - data_denoised)

            plt.subplot(n_rows, n_cols, j + 1)
            im = exp.plot_data(cmap="binary")
            plt.scatter(*guesses.T, c="r", s=1)

            plt.title(f"Init guess: {init_guess}")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(n_rows, n_cols, n_cols + j + 1)
            im = exp_diff.plot_data(cmap="binary")
            plt.xticks([])
            plt.yticks([])

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(im.axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            for i, delta in enumerate(deltas):
                split = VoronoiSplit(guesses, delta, data.shape)
                plt.subplot(n_rows, n_cols, (2 + i) * n_cols + j + 1)
                exp.plot_data("binary")
                split.plot_split("gist_ncar", components=False, alpha=0.7)
                plt.title(f"Components: {split.n_components}")
                if j == 0:
                    plt.ylabel(f"Delta: {delta}")
                plt.xticks([])
                plt.yticks([])

            print(f"Scale: {scale}, init guess: {init_guess}")
        plt.suptitle(f"Scale: {scale}", x=0.0, horizontalalignment="left")
        plt.tight_layout()
        plt.savefig(os.path.join(img_out_path, f"search_{scale}.pdf"), dpi=600)
        plt.close()
