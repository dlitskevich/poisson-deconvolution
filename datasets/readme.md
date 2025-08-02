## Setting up a dataset

1. Create a folder in this `datasets` directory with a desired name (this `dataset_name` is used to identify the dataset).
2. Store your data, named `data`, in the folder. Supported file formats: `.npy` (numpy array), `.csv` (comma delimited), `.txt` (comma delimited), `.png`, `.tiff` (use images with a white background).

## Configuring estimation parameters

To run the estimation procedure, you will need to specify certain parameters. Create a `config.json` file in the folder with the following structure:

```json
{
  "num_atoms": [40, 50, 60, 70, 80],
  "scale": 0.007,
  "estimators": ["EM (moment)", "Moment", "EM (mode)", "Mode"],
  "init_guess": 39,
  "init_scale": 0.03,
  "split_num_atoms_factor": 2,
  "deltas": [0.03, 0.045, 0.06, 0.075],
  "n_processes": 8
}
```

If you do not know which parameter to choose, go to the [Parameters comparison](#parameters-comparison) section.

It is not recommended to redo the estimation procedure (otherwise, some results will be overwritten).
Instead, create a new folder with a new `config.json` file and run the estimation procedure again.

### Parameter descriptions

- `init_guess`: The number of seeds to create Voronoi diagrams and to denoise the data.
- `init_scale`: The initial scale for the Uniform kernel.
- `deltas`: Used to join the cells in the Voronoi diagram, if the distance between the seeds is less than `delta`. The smaller the value, the smaller the resulting cell size.
- `num_atoms`: The number of mixture components (i.e. the number of molecules).
- `scale`: The standard deviation of the normal distribution, which is used as a convolution kernel in the data. Note that the data space is $[0,1]^2$.
- `estimators`: The estimators used to estimate the locations of the Gaussian mixture components (i.e., molecule locations). Supported estimators: `EM (moment)`, `EM (mode)`, `Moment`, `Mode`.
- `split_num_atoms_factor`: The number of atoms in each split will be rounded to the nearest multiple of `split_num_atoms_factor`. 
- `n_processes`: The number of processes to use for parallelization. If not provided, 1 process will be used.

### Algorithm overview

The algorithm consists of the following steps:

1. **Initial mode selection**: The `init_guess` number of modes are selected from the data. The data is denoised by removing the modes convolved with the uniform distribution on $[-s, s]^2$ (where $s$ is the `init_scale`).
2. **Voronoi segmentation**: The data is segmented into Voronoi cells using the initial modes as seeds. The cells are joined if the distance between the seeds is less than `delta`.
3. **Estimations**: The estimations are made for each cell with the number of mixture components proportional to the mass of the data enclosed by the cell and rounded to the nearest multiple of `split_num_atoms_factor`.

### <a name="parameters-comparison"></a> Parameters comparison

If you don't know which `init_scale`, `init_guess`, or `deltas` to use, create a `search.json` file in the folder with the following structure:

```json
{
  "scales": [0.01, 0.02, 0.03, 0.04, 0.05],
  "init_guesses": [35, 39, 45],
  "deltas": [0.03, 0.045, 0.06, 0.075]
}
```

Use [`../scripts/params/search.py`](../scripts/params/search.py) script to search the parameters. Replace `dataset_name` with the name of your dataset.

```sh
python -m scripts.params.search dataset_name
```

## Additional parameters

The following additional parameters can be provided in `config.json` (do not specify, if unknown):

```json
{
  "t": 1,
  "scale_data_by": 1,
  "use_t_in_mom": false
}
```

- The `t` parameter is the illumination time. By default the sum of the data matrix entries is used instead.
- The `scale_data_by` parameter is used to scale the data, i.e. multiply by `scale_data_by` the data matrix. Useful when reading images, since they are normalized to sum up to 1.
- The `use_t_in_mom` parameter is used to use the `t` parameter in the moment estimator, by default the sum of the data matrix entries is used instead.

## Results

In the [`../results/{dataset_name}/img/search`](../results) folder, the illustrations can be found depicting the initial guesses together with residual noise and Voronoi diagrams for each parameter combination.
