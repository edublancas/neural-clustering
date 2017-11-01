import numpy as np
import matplotlib.pyplot as plt


class Explorer(object):

    def __init__(self, path_to_readings, dtype, window_size,
                 n_channels):
        self.data = np.fromfile(path_to_readings, dtype)
        self.n_channels = n_channels

        obs = int(self.data.shape[0]/n_channels)
        self.data = self.data.reshape(obs, n_channels)

        self.window_size = window_size

    def read_waveform(self, time, channels='all'):
        """Read a waveform over 2*window_size + 1, centered at time
        """
        start = time - self.window_size
        end = time + self.window_size + 1

        if channels == 'all':
            channels = range(self.n_channels)

        return self.data[start:end, channels]

    def plot_waveform(self, time, channels, ax=None, line_at_t=False,
                      overlay=False):
        """
        Plot a waveform around a window size in selected channels
        """
        ax = ax if ax else plt

        n_channels = len(channels)

        if overlay:
            axs = [ax] * n_channels
        else:
            f, axs = ax.subplots(n_channels, 1)

        for ch, ax in zip(channels, axs):
            waveform = self.read_waveform(time, ch)
            ax.plot(waveform)

        if line_at_t:
            ax.axvline(x=time + 1)
