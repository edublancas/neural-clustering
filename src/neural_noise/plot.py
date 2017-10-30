"""
Plotting functions
"""
import matplotlib.pyplot as plt


def geometry(geom):
    x, y = geom.T
    plt.scatter(x, y)


def waveforms(data, channels, time=None, dt=None):
    f, axs = plt.subplots(len(channels), 1)

    for ax, ch in zip(axs, channels):
        waveform(data, ch, time, dt, ax=ax)


def waveform(data, channel=None, time=None, dt=None, ax=None):
    ax = ax if ax else plt
    spike = (data[:, channel] if not time
             else data[time-dt-1:time+dt+1, channel])
    ax.plot(spike)

    if dt:
        ax.axvline(x=dt + 1)


def score(data, ax=None):
    """Single channel score plot
    """
    ax = ax if ax else plt
    x = data[:, 0]
    y = data[:, 1]

    ax.scatter(x, y)


def scores(data):
    """Multi channel score plot
    """
    n_samples, n_features, n_channels = data.shape

    rows = 4
    cols = 2

    f, axs = plt.subplots(rows, cols)
    # flatten axs
    axs = [item for sublist in axs for item in sublist]

    for ax, ch in zip(axs, range(n_channels)):
        score(data[:, :, ch], ax=ax)
