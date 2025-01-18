import os

import numpy as np

from scripts.parse_config import EstimationConfig

DATA_EXTS = [".npy"]


def get_filenames(dir_path: str) -> list:
    return [
        f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
    ]


def read_data_file(dir: str) -> np.ndarray:
    filenames = get_filenames(dir)
    exts = [ext for ext in DATA_EXTS if f"data{ext}" in filenames]
    if len(exts) == 0:
        raise ValueError(
            f"Data file not found in {filenames}\nSupported extensions: {DATA_EXTS}"
        )
    ext = exts[0]

    match ext:
        case ".npy":
            return np.load(os.path.join(dir, f"data{ext}"))
        case _:
            raise ValueError(f"Unsupported extension: {ext}")


def read_config_file(dir: str) -> EstimationConfig:
    name = "config.json"
    try:
        return EstimationConfig.from_path(os.path.join(dir, name))
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {name}")


def read_dataset(dir_path: str):

    data = read_data_file(dir_path)
    data = data[::-1, :].T  # for correct orientation
    config = read_config_file(dir_path)

    return data, config
