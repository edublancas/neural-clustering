#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import tensorflow as tf

from edward.models import \
    Normal, MultivariateNormalDiag, Mixture, Categorical, Beta
from matplotlib import pyplot as plt


def build_toy_dataset(N):
  pi = np.array([0.4, 0.6])
  mus = [[5, 5], [-5, -5]]
  stds = [[0.1, 0.1], [0.1, 0.1]]
  x = np.zeros((N, 2), dtype=np.float32)
  for n in range(N):
    k = np.argmax(np.random.multinomial(1, pi))
    x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

  return x


def stick_breaking(v):
  remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
  return v * remaining_pieces


ed.set_seed(42)
N = 500
D = 2
T = K = 5  # truncation level in DP
alpha = 0.5

# DATA
x_train = build_toy_dataset(N)
# plt.scatter(x_train[:, 0], x_train[:, 1])
# plt.axis([-3, 3, -3, 3])
# plt.title("Data")
# plt.show()

# MODEL
beta = Beta(tf.ones(T), tf.ones(T) * alpha)
pi = stick_breaking(beta)

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigma = Gamma(tf.ones(D), tf.ones(D), sample_shape=K)
cat = Categorical(probs=pi, sample_shape=N)
components = [
    MultivariateNormalDiag(mu[k], sigma[k])
    for k in range(K)]
x = Mixture(cat=cat, components=components, sample_shape=N)

# INFERENCE
qmu = Normal(tf.Variable(tf.random_normal([K, D])),
             tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))) + 1e-5)
qbeta = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([T]))) + 1e-5,
             tf.nn.softplus(tf.Variable(tf.random_normal([T]))) + 1e-5)

inference = ed.KLqp({beta: qbeta, mu: qmu}, data={x: x_train})
inference.initialize(n_samples=5, n_iter=500, n_print=25)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    t = info_dict['t']

    if t % inference.n_print == 0:
        print("Inferred cluster means:")
        print(sess.run(qmu.mean()))


# CRITICISM
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)
x_post = Normal(mu=tf.ones([N, 1, 1, 1]) * mu_sample,
                sigma=tf.ones([N, 1, 1, 1]) * 0.1)
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()
plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()