import collections
from math import sqrt, ceil, floor
import matplotlib.pyplot as plt


def _is_iter(obj):
    return isinstance(obj, collections.Iterable)


def _grid_size(n_groups, max_cols=None):
    sq = sqrt(n_groups)
    cols = floor(sq)
    rows = ceil(sq)
    rows = rows + 1 if rows * cols < n_groups else rows

    if max_cols and cols > max_cols:
        rows = ceil(n_groups/max_cols)
        cols = max_cols

    return rows, cols


def _make_grid_plot(fn, group_ids, sharex, sharey, max_cols=None):
    rows, cols = _grid_size(group_ids, max_cols)

    f, axs = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)

    axs = axs if _is_iter(axs) else [axs]

    if cols > 1:
        axs = [item for sublist in axs for item in sublist]

    for g, ax in zip(group_ids, axs):
        fn(group_id=g, ax=ax)


def _make_grid_plot_by_axis(fn, data, axis, sharex, sharey, max_cols=None):
    group_ids = data.shape[axis]

    rows, cols = _grid_size(group_ids, max_cols)

    f, axs = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)

    axs = axs if _is_iter(axs) else [axs]

    if cols > 1:
        axs = [item for sublist in axs for item in sublist]

    for i, ax in enumerate(axs):
        indexes = [slice(None) for _ in range(data.ndim)]
        indexes[axis] = i
        fn(data[indexes], ax=ax)
        ax.set_title('Index {}'.format(i))
