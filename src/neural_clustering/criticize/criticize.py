import logging
import os

import tensorflow as tf
from edward.models import Normal
import edward as ed
import numpy as np


logger = logging.getLogger(__name__)


def relevel_clusters(clusters):
    group_ids = np.unique(clusters)
    groups = len(group_ids)
    mapping = {k: v for k, v in zip(group_ids, range(groups))}
    return np.array([mapping[k] for k in clusters])


def find_cluster_assignments(x_train, qmu, params):
    """Find cluster assignments
    """
    N, D = x_train.shape
    K = params.get('truncation_level') or params.get('k')

    # http://edwardlib.org/api/ed/MonteCarlo
    total = 1000
    burn_in = 400
    SC = total - burn_in

    mu_sample = qmu.sample(total)[burn_in:, :, :]

    mu_sample

    x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                    tf.ones([N, 1, 1, 1]) * 1.0)

    x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])
    # is it ok to convert like this?
    x_broadcasted = tf.to_float(x_broadcasted)

    log_liks = x_post.log_prob(x_broadcasted)
    log_liks = tf.reduce_sum(log_liks, 3)
    log_liks = tf.reduce_mean(log_liks, 1)

    clusters = tf.argmax(log_liks, 1).eval()

    return relevel_clusters(clusters)


def store_cluster_assignments(cfg, x_train, qmu, params):
    """Store cluster assignments for experiment
    """
    clusters = find_cluster_assignments(x_train, qmu, params)

    path_to_sessions = os.path.join(cfg['root'], 'sessions')
    path_to_output = os.path.join(path_to_sessions, params['name'],
                                  'clusters.npy')
    np.save(path_to_output, clusters)
    logger.info('Cluster assignmens stored in {}'.format(path_to_output))

    return clusters


def ppc_plot(fn, stat_name, x_pred, x_train, n_samples=1000, bins=10):
    y_rep, y = ed.ppc(fn, data={x_pred: x_train}, n_samples=n_samples)
    ed.ppc_stat_hist_plot(y[0], y_rep,
                          stat_name=r'$T \equiv$ {}'.format(stat_name),
                          bins=bins)
