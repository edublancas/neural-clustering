"""
Plotting functions
"""
import matplotlib.pyplot as plt
import numpy as np


def geometry(geom):
    x, y = geom.T
    plt.scatter(x, y)


def multi_channel_series(data, channels, time=None, dt=None):
    f, axs = plt.subplots(len(channels), 1)

    for ax, ch in zip(axs, channels):
        series(data, ch, time, dt, ax=ax)


def series(data, channel=None, time=None, dt=None, ax=None):
    ax = ax if ax else plt
    spike = (data[:, channel] if not time
             else data[time-dt-1:time+dt+1, channel])
    ax.plot(spike)

    if dt:
        ax.axvline(x=dt + 1)


def waveform(data):
    """Single channel scatterplot
    """
    x = data[:, 0]
    y = data[:, 1]
    plt.scatterplot(x, y)


def waveforms(data):
    """Multi channel scatterplot
    """
    n_samples, n_features, n_channels = data.shape

    rows = int(np.ceil(np.sqrt(n_channels)))
    cols = n_channels - rows

    f, axs = plt.subplots(rows, cols)

    for ax, ch in zip(axs, n_channels):
        data(data[:, :, ch], ax=ax)
