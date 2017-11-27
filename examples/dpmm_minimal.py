"""
Truncated DPMM using Edward

Based on:
https://gist.github.com/dustinvtran/d8cc112636b219776621444324919928
http://edwardlib.org/tutorials/unsupervised
http://docs.pymc.io/notebooks/dp_mix.html
https://discourse.edwardlib.org/t/dpm-model-for-clustering/97/6
"""
from edward.models import (Normal, MultivariateNormalDiag, Beta,
                           ParamMixture, Empirical, Gamma)
import edward as ed
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[10, 10], [-10, -10]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
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
T = K = 3  # truncation level

x_train = build_toy_dataset(N)

# plot toy dataset
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()

# Model
alpha = Gamma(1.0, 1.0)
beta = Beta(tf.ones(T), tf.ones(T) * alpha)

pi = stick_breaking(beta)
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigmasq = tf.ones((K, D))

x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)


# Inference
S = 10000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, K])))
qalpha = Empirical(tf.Variable(tf.ones(S)))

inference = ed.HMC({alpha: qalpha, mu: qmu}, data={x: x_train})
inference.initialize()

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()
inference.run()
