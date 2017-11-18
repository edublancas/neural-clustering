import matplotlib.pyplot as plt
import seaborn as sns

from ..util import _make_grid_plot_by_axis


def params_over_iterations(param, axis, ax=None, sharex=True,
                           sharey=False):
    def single_parameter(data, ax):
        ax.plot(data)

    ax = plt if ax is None else ax

    values = param.params.eval()

    _make_grid_plot_by_axis(single_parameter, values, axis, ax, sharex, sharey,
                            max_cols=None)


def params_distribution(data, axis, ax=None, sharex=True, sharey=False):
    def single_parameter(data, ax):
        sns.distplot(data, ax=ax)

    ax = plt if ax is None else ax

    _make_grid_plot_by_axis(single_parameter, data, axis, ax, sharex, sharey,
                            max_cols=None)
