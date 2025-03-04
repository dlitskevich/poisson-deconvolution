## Store your data here

1. create a folder in this `datasets` folder with a desired name (this name is used to identify the dataset)
2. store your data, named `data` in the folder, supported file formats: `.npy` (numpy array), `.csv` (',' delimiter), `.txt` (',' delimiter), .png, .tiff (use images with white background).
3. create a `config.json` file in the folder, with the following structure:
   <!--add that it is used only for the initialization 4. (optional) store `kernel` in the folder, supported file formats: `.npy`. It should be the same size as `data`. If not provided the normal distribution with `scale` will be used. Only for initialization. -->
   <!-- 4. explain how to set kernel_bandwidth i.e scale -->

```json
{
  "num_atoms": [40, 50, 60, 70, 80],
  "scale": 0.01,
  "estimators": ["EM (moment)", "Moment"],
  "init_guess": 110,
  "deltas": [0.025, 0.05, 0.075, 0.1]
}
```

<!-- explain the algorithm
that algorithm works well only for small amount of points, that's why first we segment the image via Voronoi cells, using delta.
Then Estimations are made for each sub domain. -->

`init_guess` is the number of initial guesses to create Voronoi diagrams and to denoise data.

`deltas` used for Voronoi diagrams, the smaller the value, the smaller the cell size.

`num_atoms` is the number of mixture components.

`scale` is the standard deviation of the normal distribution, which is used as a convolution kernel in the data. Note that the data space is [0,1]^2

`estimators` are the estimators used to estimate the locations of the Gaussian mixture components (i.e. molecule locations).
The supported estimators are: `EM (moment)`, `EM (mode)`, `Moment`.

### Parameters comparison

If you don't know which `scale` or `init_guess` to use, create a `search.json` file in the folder, with the following structure:

```json
{
  "scales": [0.01, 0.02, 0.03, 0.04, 0.05],
  "init_guesses": [40, 60, 80],
  "deltas": [0.025, 0.05, 0.075]
}
```

then run [`../scripts/params/search.py`](../scripts/params/search.py) by

```sh
python -m scripts.params.search `dataset_name`
```

In the [`../results/{dataset_name}/img/search`](../results) folder, you will find the illustrations of the initial guesses together with residual noise for each parameter combination.
