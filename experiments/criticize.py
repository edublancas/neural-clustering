import os

import numpy as np

from neural_clustering import config
from neural_clustering.criticize import restore_session
from neural_clustering.criticize import find_cluster_assignments

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cfg = config.load('config.yaml')
session_name = '15-Nov-2017@10-59-03-DPMM'

qmu, qbeta, x_train, params = restore_session(cfg, session_name)

# TODO: the number of samples affects this, adds as a param
clusters = find_cluster_assignments(x_train, qmu, params)

clusters

np.unique(clusters, return_counts=True)


import tensorflow as tf
import edward as ed

# http://edwardlib.org/api/ed/evaluate
x_pred = ed.copy(x, {mu: qmu, beta: qbeta})

# log-likelihood performance
ed.evaluate('log_likelihood', data={x_pred: x_train})

# http://edwardlib.org/api/ed/ppc
# http://edwardlib.org/api/ed/ppc_density_plot
# http://edwardlib.org/api/ed/ppc_stat_hist_plot

# posterior predictive check
# T is a user-defined function of data, T(data)
T = lambda xs, mus: tf.reduce_mean(xs[x_pred])
y_rep, y = ed.ppc(T, data={x_pred: x_train})


ed.ppc_stat_hist_plot(
    y[0], y_rep, stat_name=r'$T \equiv$mean', bins=10)

plt.show()

ed.ppc_density_plot(y, y_rep)
plt.show()

# TODO: plot params values at eveery training iteration

