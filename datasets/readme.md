## Store your data here

1. create a folder with a desired name
2. store your data, named `data` in the folder, supported file formats: `.npy`
3. create a `config.json` file in the folder, with the following structure:
4. (optional) store `kernel` in the folder, supported file formats: `.npy`. It should be the same size as `data`. If not provided the normal distribution with `scale` will be used.

```json
{
  "num_atoms": [40, 50, 60, 70, 80],
  "scale": 0.01,
  "estimators": ["EM (moment)", "EM (mode)", "Moment"],
  "init_guess": 110,
  "deltas": [0.025, 0.05, 0.075, 0.1]
}
```

`num_atoms` is the number of mixture components.

`scale` is the standard deviation of the normal distribution.

`estimators` are the estimators used to estimate the parameters of the normal distribution. The supported estimators are: `EM (moment)`, `EM (mode)`, `Moment`.

`init_guess` is the number of initial guesses to create Voronoi diagrams and to denoise data.

`deltas` used for Voronoi diagrams, the smaller the value, the smaller the cell size.
