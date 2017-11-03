import collections
from math import sqrt
from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

from yass import geometry


def _is_iter(obj):
    return isinstance(obj, collections.Iterable)


def _grid_size(group_ids):
    sq = sqrt(len(group_ids))
    cols = int(sq)
    rows = cols if sq.is_integer() else cols + 1
    return rows, cols


def _make_grid_plot(fn, group_ids, ax, sharex, sharey):
    rows, cols = _grid_size(group_ids)

    f, axs = ax.subplots(rows, cols, sharex=sharex, sharey=sharey)
    axs = axs if _is_iter(axs) else [axs]

    if cols > 1:
        axs = [item for sublist in axs for item in sublist]

    for g, ax in zip(group_ids, axs):
        fn(group_id=g, ax=ax)


class SpikeTrainExplorer(object):
    """Explore spike trains and templates

    Parameters
    ----------
    templates: np.ndarray
        Templates
    spike_train: np.ndarray
        Spike train
    recording_explorer: RecordingExplorer, optional
        Recording explorer instance
    projection_matrix: np.ndarray, optional
        Projection Matrix
    """

    def __init__(self, templates, spike_train, recording_explorer=None,
                 projection_matrix=None):
        self.spike_train = spike_train
        self.templates = templates
        self.recording_explorer = recording_explorer
        self.projection_matrix = projection_matrix

        if projection_matrix is not None:
            ft_space = self._reduce_dimension
            self.templates_feature_space = ft_space(self.templates)
        else:
            self.templates_feature_space = None

    def _reduce_dimension(self, data, flatten=False):
        """Reduce dimensionality
        """
        R, n_features = self.projection_matrix.shape
        nchannel, R, n_data = data.shape

        reduced = np.transpose(np.reshape(np.matmul(np.reshape(np.transpose(
                    data, [0, 2, 1]), (-1, R)), [self.projection_matrix]),
                    (nchannel, n_data, n_features)), (0, 2, 1))

        if flatten:
            reduced = np.reshape(reduced, [reduced.shape[0], -1])

        return reduced

    def scores_for_groups(self, group_ids, flatten=True):
        """Get scores for one or more groups

        Parameters
        ----------
        group_id: int or list
            The id for one or more group
        flatten: bool, optional
            Flatten scores along channels, defaults to True
        """
        group_ids = group_ids if _is_iter(group_ids) else [group_ids]

        waveforms = [self.waveforms_for_group(g) for g in group_ids]
        lengths = np.hstack([np.ones(w.shape[0]) * g for g, w in zip(group_ids,
                             waveforms)])

        return self._reduce_dimension(np.vstack(waveforms),
                                      flatten=flatten), lengths

    def times_for_group(self, group_id):
        """Get the spiking times for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        matches_group = self.spike_train[:, 1] == group_id
        return self.spike_train[matches_group][:, 0]

    def main_channel_for_group(self, group_id):
        """Get the main channel for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        template = self.templates[:, :, group_id]
        main = np.argmax(np.max(template, axis=1))
        return main

    def neighbor_channels_for_group(self, group_id):
        """Get the neighbor channels for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        main = self.main_channel_for_group(group_id)
        neigh_matrix = self.recording_explorer.neigh_matrix
        return np.where(neigh_matrix[main])[0]

    def template_for_group(self, group_id):
        """Get the template for a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """
        return self.templates[:, :, group_id]

    def waveforms_for_group(self, group_id, channels):
        """Get the waveforms around a group

        Parameters
        ----------
        group_id: int
            The id for the group
        """

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
        return close_to_far_idx[:k+1]

    def _plot_template(self, group_id, ax=None):
        """Plot a single template
        """
        ax = ax if ax else plt
        template = self.template_for_group(group_id)
        ax.plot(template.T)
        ax.set_title('Template {}'.format(group_id))
        plt.tight_layout()

    def plot_templates(self, group_ids, ax=None, sharex=True, sharey=False):
        """Plot templates

        group_ids: int or list
            Groups to plot, it can be either a single group or a list of groups
        """
        ax = ax if ax else plt
        group_ids = group_ids if _is_iter(group_ids) else [group_ids]
        _make_grid_plot(self._plot_template, group_ids, ax, sharex, sharey)

    def plot_pca(self, group_ids, sample=None, ax=None):
        """
        Reduce dimensionality using PCA and plot data
        """
        ax = ax if ax else plt

        scores, labels = self.scores_for_groups(group_ids)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(scores)

        for color in np.unique(group_ids).astype('int'):
            x = reduced[labels == color, 0]
            y = reduced[labels == color, 1]

            if sample:
                x = np.random.choice(x, size=int(sample*len(x)), replace=False)
                y = np.random.choice(x, size=int(sample*len(y)), replace=False)

            plt.scatter(x, y, label='Group {}'.format(color, alpha=0.7))

        ax.legend()

    def plot_lda(self, group_ids, sample=None, ax=None):
        """
        Reduce dimensionality using LDA and plot data
        """
        ax = plt if ax is None else ax

        scores, labels = self.scores_for_groups(group_ids)

        lda = LDA(n_components=2)
        reduced = lda.fit_transform(scores, labels)

        for color in np.unique(group_ids).astype('int'):
            x = reduced[labels == color, 0]
            y = reduced[labels == color, 1]

            if sample:
                x = np.random.choice(x, size=int(sample*len(x)), replace=False)
                y = np.random.choice(x, size=int(sample*len(y)), replace=False)

            ax.scatter(x, y, label='Group {}'.format(color), alpha=0.7)

        ax.legend()

    def visualize_closest_clusters(self, group_id, k, mode='LDA', sample=None,
                                   ax=None):
        """Visualize close clusters
        """
        ax = plt if ax is None else ax

        groups = self.close_templates(group_id, k)

        if mode == 'LDA':
            self.plot_lda(groups, sample=sample, ax=ax)
        elif mode == 'PCA':
            self.plot_pca(groups, sample=sample, ax=ax)
        else:
            raise ValueError('Only PCA and LDA modes are supported')

    def visualize_all_clusters(self, k, mode='LDA', sample=None, ax=None,
                               sharex=True, sharey=False):
        ax = plt if ax is None else ax
        all_ids = range(self.templates.shape[2])
        rows, cols = _grid_size(all_ids)

        fn = partial(self.visualize_closest_clusters, k=k, mode=mode,
                     sample=sample)

        _make_grid_plot(fn, all_ids, ax, sharex, sharey)


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

    def plot_waveform_around_channel(self, time, channel, ax=None,
                                     line_at_t=False, overlay=False):
        return self.plot_waveform(time,
                                  channels=self.neighbors_for_channel(channel),
                                  ax=ax, line_at_t=line_at_t, overlay=overlay)
