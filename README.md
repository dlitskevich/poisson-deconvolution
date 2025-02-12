# [Minimax Rates for Discrete Signal Recovery with Applications to Photonic Imaging](#)

TODO: upd link

**Authors**: Shayan Hundrieser, Tudor Manole, Danila Litskevich, Axel Munk

<p align='center'><img src='assets/overview.png' alt='Overview.' width='100%'> </p>

We analyze the statistical problem of recovering a discrete signal, modeled as a $k$-atomic uniform distribution $\mu$, from a binned Poisson convolution model. This question is motivated by super-resolution microscopy where precise estimation of $\mu$ provides insights into spatial configurations, such as protein colocalization in cellular imaging. Our main result quantifies the minimax risk of estimating $\mu$ under the Wasserstein distance for Gaussian and compactly supported, smooth convolution kernels. Specifically, we show that the global minimax risk scales with $t^{-1/2k}$ for $t \to \infty$, where $t$ denotes the illumination time of the probe, and that this rate is achieved by the method of moments and the maximum likelihood estimator.

This repository contains the implementation of the methods and evaluation scripts to reproduce the results of the estimations for experimental STED microscopy data to locate single DNA origami. In addition, we provide numerical experiments on simulated data that showcase the practical performance of the estimators.

## Installation

To set up the corresponding `conda` environment, run:

```sh
conda create -n poisson-deconvolution python=3.12.3
source activate poisson-deconvolution
```

Install dependencies via:

```sh
pip install -r requirements.txt
```

To install the library, run:

```sh
pip install -e .
```

## Datasets

The experimental STED microscopy data is considered from [here](https://doi.org/10.1214/17-AOS1669).

<p align="center">
  <img src="assets/data.png" width="49%"/>
  <img src="assets/data_zoom.png" width="49%"/> 
</p>

See [`datasets/readme`](datasets/readme.md) for the setup instructions.

## Estimations

After setting up the datasets, the estimations can be run via the [`scripts/estimate.py`](scripts/estimate.py) script:

```sh
python -m scripts.estimate `dataset_name`
```

Once the estimations are done, the results can be plotted via the [`scripts/plot.py`](scripts/plot.py) script:

```sh
python -m scripts.plot `dataset_name`
```

The estimations with plots will be saved in the `results` folder. See [`results/readme`](results/readme.md) for the plotting options.

![estimations](assets/estimations_zoom_example.png)

## Experiments

The experiments can be run via the [`scripts/simulate.py`](scripts/simulate.py) script:

```sh
python -m scripts.simulate `simulation_name` `setting_id`
```

For more details, see [`simulations/readme`](simulations/readme.md).

After the simulations are done, the results can be plotted via the [`scripts/plot_simulations.py`](scripts/plot_simulations.py) script:

```sh
python -m scripts.plot_simulations `simulation_name`
```

## Citation

If you found our work useful, please consider citing us:

```bibtex
@article{hundrieser2025minimax,
  title   = {Minimax Rates for Discrete Signal Recovery with Applications to Photonic Imaging},
  author  = {{Shayan Hundrieser, Tudor Manole, Danila Litskevich, and Axel Munk}},
  year    = {2025},
  journal = {In preparation}
}
```

## Contact

If you have questions, please contact us via GitHub Issues.
