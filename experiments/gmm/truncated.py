#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.cm as cm
import numpy as np
# import seaborn as sns
import tensorflow as tf

from edward.models import \
    Normal, MultivariateNormalDiag, Mixture, Categorical, Beta, InverseGamma, \
    ParamMixture, Empirical, Gamma
from matplotlib import pyplot as plt


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[1, 1], [-1, -1]]
    stds = [[0.1, 0.1], [0.1, 0.1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
      k = np.argmax(np.random.multinomial(1, pi))
      x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


def stick_breaking(v):
  remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
  return v * remaining_pieces


ed.set_seed(0)
N = 500
D = 2
T = K = 10  # truncation level in DP
alpha = 0.5

# DATA
x_train = build_toy_dataset(N)
# plt.scatter(x_train[:, 0], x_train[:, 1])
# plt.axis([-3, 3, -3, 3])
# plt.title("Data")
# plt.show()

# MODEL
beta = Beta(tf.ones(T), tf.ones(T))
beta

pi = stick_breaking(beta)
pi

cat = Categorical(probs=pi, sample_shape=N)
cat

# mu = Normal(tf.zeros([K, D]), tf.ones([K, D]))

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
# sigma = Gamma(tf.ones(D), tf.ones(D), sample_shape=K)

sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)
sigma = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)



# components = [MultivariateNormalDiag(tf.ones([N, 1]) * tf.gather(mu, k),
#                                      tf.ones([N, D]) * 0.1)
#               for k in range(K)]

# components = [MultivariateNormalDiag(tf.ones([N, 1]) * tf.gather(mu, k),
#                                      tf.ones([N, D]) * 0.1)
#               for k in range(K)]

components = [
    MultivariateNormalDiag(mu[k], sigma[k], sample_shape=N)
    for k in range(K)]

components

pi, cat, components


# x = Mixture(cat=cat, components=components, sample_shape=N)
# x

x = ParamMixture(pi, {'loc': mu, 'scale_diag': sigmasq},
                 MultivariateNormalDiag,
                 sample_shape=N)

# z = x.cat



# INFERENCE

# find the parameters for each mixture model component
# qmu = Normal(tf.Variable(tf.random_normal([K, D])),
             # tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

# qbeta = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([T]))),
#              tf.nn.softplus(tf.Variable(tf.random_normal([T]))))

# inference = ed.KLqp({beta: qbeta, mu: qmu}, data={x: x_train})
# 

S = 100000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, K])))
qbeta

# inference = ed.HMC({beta: qbeta, mu: qmu}, data={x: x_train})
# inference.initialize(n_steps=5)

inference = ed.SGLD({beta: qbeta, mu: qmu}, data={x: x_train})
inference.initialize()



# inference.initialize(n_samples=5, n_iter=100, n_print=25)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


# for _ in range(inference.n_iter):
#     info_dict = inference.update()
#     inference.print_progress(info_dict)
#     t = info_dict['t']

#     if t % inference.n_print == 0:
#       print(t)


for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  t = info_dict['t']

  if t % inference.n_print == 0:
    print("Inferred cluster means:")
    print(sess.run(qmu.mean()))
    print('scale')
    # print(sess.run(qmu.scale))
    print('beta params')
    # print(sess.run(qbeta.concentration1))
    # print(sess.run(qbeta.concentration0))

# CRITICISM
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
SC = 1000
mu_sample = qmu.sample(SC)
x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                tf.ones([N, 1, 1, 1]) * 0.1)

x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])

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