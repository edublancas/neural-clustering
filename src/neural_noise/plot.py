"""
Plotting functions
"""
import matplotlib.pyplot as plt


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
