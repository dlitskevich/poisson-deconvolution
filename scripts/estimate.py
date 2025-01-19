import sys

from scripts.data_estimator import DataEstimator


if __name__ == "__main__":
    dataset = sys.argv[1]
    data_estimator = DataEstimator(dataset)
    data_estimator.run_estimations()
