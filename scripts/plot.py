import sys

from scripts.plotting.plot_results import PlotResults


if __name__ == "__main__":
    dataset = sys.argv[1]
    data_estimator = PlotResults(dataset)
    data_estimator.plot_data()
    data_estimator.plot_best_estimations()
    data_estimator.plot_splits()
    data_estimator.plot_split_estimations()
    data_estimator.plot_estimated_zoomed()
