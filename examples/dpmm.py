"""
Truncated DPMM using Edward

Based on:
https://gist.github.com/dustinvtran/d8cc112636b219776621444324919928
http://edwardlib.org/tutorials/unsupervised
"""
from edward.models import (Normal, MultivariateNormalDiag, Beta,
                           InverseGamma,  ParamMixture, Empirical)
import edward as ed
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[5, 5], [-5, -5]]
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
T = K = 5  # truncation level in DP

x_train = build_toy_dataset(N)

# plot toy dataset
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()

# Model
beta = Beta(tf.ones(T), tf.ones(T))
pi = stick_breaking(beta)

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)


# Inference with KLqp - getting nans
qmu = Normal(tf.Variable(tf.random_normal([K, D])),
             tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))
qbeta = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([T]))),
             tf.nn.softplus(tf.Variable(tf.random_normal([T]))))

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


# Inference with HMC - works (SGLD also works)
S = 20000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

inference = ed.HMC({beta: qbeta, mu: qmu}, data={x: x_train})
inference.initialize()

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()

inference.run()


# Criticism
# TODO: plot params values at eveery training iteration
plt.plot(qbeta.params.eval())
plt.show()
qbeta.params.eval()[-10:, :]

qmu.params.eval()[-2:, :, :]

SC = 1000

mu_sample = qmu.sample(SC)

x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                tf.ones([N, 1, 1, 1]) * 1.0)

x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])

log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

clusters = tf.argmax(log_liks, 1).eval()

print('Found {} clusters'.format(len(np.unique(clusters))))
np.unique(clusters, return_counts=True)

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters)
plt.title("Predicted cluster assignments")
plt.show()
