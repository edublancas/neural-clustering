import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..util import _make_grid_plot_by_axis


def params_over_iterations(param, axis, sharex=True,
                           sharey=False):
    def single_parameter(data, ax):
        ax.plot(data)

    values = param.params.eval()

    _make_grid_plot_by_axis(single_parameter, values, axis, sharex, sharey,
                            max_cols=None)


def params_distribution(data, axis, sharex=True, sharey=False):
    def single_parameter(data, ax):
        sns.distplot(data, ax=ax)

    _make_grid_plot_by_axis(single_parameter, data, axis, sharex, sharey,
                            max_cols=None)


def cluster_counts(clusters, ax=None):
    ax = plt.gca() if ax is None else ax
    names, values = np.unique(clusters, return_counts=True)
    ax.bar([str(n) for n in names], values)
    ax.set_title('Cluster counts')
