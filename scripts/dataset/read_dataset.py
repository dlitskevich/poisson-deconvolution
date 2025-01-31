import os
import pathlib

import numpy as np

from scripts.dataset.parse_config import EstimationConfig

DATA_EXTS = [".npy", ".csv", ".txt"]


def get_filenames(dir_path: str) -> list:
    return [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]


def read_data_file(dir: str) -> np.ndarray:
    filenames = get_filenames(dir)
    exts = [ext for ext in DATA_EXTS if f"data{ext}" in filenames]
    if len(exts) == 0:
        raise ValueError(
            f"Data file 'data.' not found in {filenames}\nSupported extensions: {DATA_EXTS}"
        )
    ext = exts[0]

    match ext:
        case ".npy":
            return np.load(os.path.join(dir, f"data{ext}"))
        case ".csv":
            return np.loadtxt(os.path.join(dir, f"data{ext}"), delimiter=",")
        case ".txt":
            return np.loadtxt(os.path.join(dir, f"data{ext}"), delimiter=",")
        case _:
            raise ValueError(f"Unsupported extension: {ext}")


def read_kernel(dir: str) -> np.ndarray:
    filenames = get_filenames(dir)
    exts = [ext for ext in DATA_EXTS if f"kernel{ext}" in filenames]
    if len(exts) == 0:
        return None
    ext = exts[0]

    match ext:
        case ".npy":
            return np.load(os.path.join(dir, f"kernel{ext}"))
        case _:
            raise ValueError(f"Unsupported extension: {ext}")


def read_config_file(dir: str) -> EstimationConfig:
    name = "config.json"
    try:
        return EstimationConfig.from_path(os.path.join(dir, name))
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {name}")


def read_dataset(dir_path: str, no_config=False):

    data = read_data_file(dir_path)
    data = data[::-1, :].T  # for correct orientation

    config = read_config_file(dir_path) if not no_config else None
    kernel = read_kernel(dir_path)
    kernel = None if kernel is None else kernel[::-1, :].T  # for correct orientation

    if kernel is not None and kernel.shape != data.shape:
        raise ValueError(
            f"Kernel shape {kernel.shape} does not match data shape {data.shape}"
        )

    return data, config, kernel


def save_dataset(
    data: np.ndarray, config: EstimationConfig, kernel: np.ndarray, path: str
):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(path, "data.npy"), data.T[::-1, :])
    config.dump(os.path.join(path, "config.json"))
    np.save(os.path.join(path, "kernel.npy"), kernel.T[::-1, :])
