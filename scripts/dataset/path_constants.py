import os
import pathlib


ROOT = pathlib.Path(__file__).parent.parent.parent
DATASET_DIR = os.path.join(ROOT, "datasets")
OUTPUT_DIR = os.path.join(ROOT, "results")
SIMULATIONS_DIR = os.path.join(ROOT, "simulations")

RUN_NAME = os.environ.get("RUN_NAME", "")
get_output_path = lambda dataset: os.path.join(
    OUTPUT_DIR, dataset if RUN_NAME == "" else f"{dataset}-{RUN_NAME}"
)
