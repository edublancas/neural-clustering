import collections
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from yass import geometry


def _is_iter(obj):
    return isinstance(collections.Iterable)


class SpikeTrainExplorer(object):
    """
        templates
        spike_train
    """

    def __init__(self, templates, spike_train, recording_explorer=None):
        self.spike_train = spike_train
        self.templates = templates
        self.recording_explorer = recording_explorer

    def times_for_group(self, group_id):
        """
        """
        matches_group = self.spike_train[:, 1] == group_id
        return self.spike_train[:, matches_group]

    def main_channel_for_group(self, group_id):
        """
        """
        template = self.templates[:, :, group_id]
        main = np.argmax(np.max(template, axis=1))
        return main

    def neighbor_channels_for_group(self, group_id):
        main = self.main_channel_for_group(group_id)
        neigh_matrix = self.recording_explorer.neigh_matrix
        return np.where(neigh_matrix[main])[0]

    def template_for_group(self, group_id):
        return self.templates[:, :, group_id]

    def template_components(self, group_id, channels):
        # get all spike times that form this group
        times = self.times_for_group(group_id)

        # find main channel
        main = self.main_channel_for_group(group_id)

        # get waveforms around the group
        around = self.recording_explorer.read_waveform_around_channel
        return [around(t, main) for t in times]

    def close_templates(self, group_id, k):
        """return K similar templates
        """
        difference = np.sum(np.square(self.templates -
                                      self.templates[:, :, [group_id]]),
                            axis=(0, 1))
        close_to_far_idx = np.argsort(difference)
        return close_to_far_idx[:k]

    def _plot_template(self, group_id, ax=None):
        """Plot a single template
        """
        ax = ax if ax else plt
        template = self.template_for_group(group_id)
        ax.plot(template.T)

    def plot_templates(self, group_ids, ax=None):
        """Plot templates

        group_ids: int or list
            Groups to plot, it can be either a single group or a list of groups
        """
        group_ids = group_ids if _is_iter(group_ids) else [group_ids]

        cols = sqrt(len(group_ids))
        rows = cols + 1

        f, axs = ax.subplots(rows, cols)
        ax = [item for sublist in axs for item in sublist]

        for g, ax in zip(group_ids, axs):
            self._plot_template(g, ax)


class RecordingExplorer(object):

    def __init__(self, path_to_readings, path_to_geom, dtype, window_size,
                 n_channels, neighbor_radius):
        self.data = np.fromfile(path_to_readings, dtype)
        self.geom = geometry.parse(path_to_geom, n_channels)
        self.neigh_matrix = geometry.find_channel_neighbors(self.geom,
                                                            neighbor_radius)

        # TODO: infer from geom?
        self.n_channels = n_channels

        obs = int(self.data.shape[0]/n_channels)
        self.data = self.data.reshape(obs, n_channels)

        self.window_size = window_size

    def neighbors_for_channel(self, channel):
        """Get the neighbors for the channel
        """
        return np.where(self.neigh_matrix[channel])[0]

    def read_waveform(self, time, channels='all'):
        """Read a waveform over 2*window_size + 1, centered at time
        """
        start = time - self.window_size
        end = time + self.window_size + 1

        if channels == 'all':
            channels = range(self.n_channels)

        return self.data[start:end, channels]

    def read_waveform_around_channel(self, time, channel):
        return self.read_waveform(time,
                                  channels=self.neighbors_for_channel(channel))

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

            # if line_at_t:
            #     ax.axvline(x=time + 1)

    def plot_waveform_around_channel(self, time, channel, ax=None, line_at_t=False,
                                     overlay=False):
        return self.plot_waveform(time,
                                  channels=self.neighbors_for_channel(channel),
                                  ax=ax, line_at_t=line_at_t, overlay=overlay)