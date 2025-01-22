import os
import pathlib


ROOT = pathlib.Path(__file__).parent.parent.parent
DATASET_DIR = os.path.join(ROOT, "datasets")
OUTPUT_DIR = os.path.join(ROOT, "results")
SIMULATIONS_DIR = os.path.join(ROOT, "simulations")
