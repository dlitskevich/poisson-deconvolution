# [Minimax Rates for Discrete Signal Recovery with Applications to Photonic Imaging]()

TODO: upd link

**Authors**: Shayan Hundrieser, Tudor Manole, Danila Litskevich,
Axel Munk

<p align='center'><img src='assets/overview.png' alt='Overview.' width='100%'> </p>

We analyze the statistical problem of recovering a discrete signal, modeled as a $k$-atomic uniform distribution $\mu$, from a binned Poisson convolution model. This question is motivated from super-resolution microscopy where precise estimation of $\mu$ provides insights into spatial configurations, such as protein colocalization in cellular imaging. Our main result quantifies the minimax risk of estimating $\mu$ under the Wasserstein distance for Gaussian and compactly supported, smooth convolution kernels.

As an application we use our methods on experimental STED microscopy data to locate single DNA origami. In addition, we complement our findings with numerical experiments that showcase the practical performance of both estimators and their trade-offs.

## Installation

To setup the corresponding `conda` environment run:

```
conda create -n poisson-deconvolution python=3.12.3
source activate poisson-deconvolution
```

Install dependencies via:

```
pip install -r requirements.txt
```

To install the library run:

```
pip install -e .
```

## Datasets

DNA origami sample from [here](https://doi.org/10.1214/17-AOS1669).

<p align="center">
  <img  src="assets/data.png" width="49%"/>
  <img  src="assets/data_zoom.png" width="49%"/> 
</p>

See `datasets/readme` for the setup instruction.

## Experiments

After setting up the datasets, the estimations can be run via the `scripts/estimate.py` script:

```
python -m scripts.estimate `dataset_name`
```

Once the estimations are done, the results can be plotted via the `scripts/plot.py` script:

```
python -m scripts.plot `dataset_name`
```

The estimations with plots will be saved in the `results` folder. See `results/readme` for the plotting options.

## Simulations

TODO:

## Citation

In case you found our work useful, please consider citing us:

```
@article{
}
```

## Contact

In case you have questions, please contact us via the Github Issues.
