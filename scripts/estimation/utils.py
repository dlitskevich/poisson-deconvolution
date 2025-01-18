from matplotlib import pyplot as plt


def plot_box(xs=[0.7, 0.9], ys=[0.5, 0.7]):
    x_min = xs[0]
    x_max = xs[1]
    y_min = ys[0]
    y_max = ys[1]
    x = [x_min, x_max, x_max, x_min, x_min]
    y = [y_min, y_min, y_max, y_max, y_min]
    plt.plot(x, y, c="gray", lw=1, ls="--")
