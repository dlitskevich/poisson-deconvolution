import sys

from scripts.plot_results import PlotResults


if __name__ == "__main__":
    dataset = sys.argv[1]
    data_estimator = PlotResults(dataset)
    data_estimator.plot_data()
    data_estimator.plot_best_estimations()
