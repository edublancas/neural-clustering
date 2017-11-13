"""
Truncated DPMM
"""
from edward.models import (Normal, MultivariateNormalDiag, Categorical, Beta,
                           InverseGamma,  ParamMixture, Empirical)

import edward as ed
import numpy as np
import tensorflow as tf


from matplotlib import pyplot as plt


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    # mus = [[1, 1], [-1, -1]]
    mus = [[5, 5], [-5, -5]]
    # stds = [[0.1, 0.1], [0.1, 0.1]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


def stick_breaking(v):
    remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
    return v * remaining_pieces


# ed.set_seed(0)

# toy data
N = 500
D = 2
T = K = 15  # truncation level in DP

x_train = build_toy_dataset(N)


# data subset
N, D = x_train.shape
T = K = 15  # truncation level in DP


plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()


# mu init
idxs = np.random.choice(range(N), size=K, replace=False)
mu_init = x_train[idxs]

# plt.scatter(mu_init[:, 0], mu_init[:, 1])
# plt.show()


# Model


beta = Beta(tf.ones(T), tf.ones(T))
pi = stick_breaking(beta)

cat = Categorical(probs=pi, sample_shape=N)

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
# seems like making it worse...
# mu = Normal(mu_init, tf.ones([K, D]))

sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)

sigma = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


components = [MultivariateNormalDiag(mu[k], sigma[k], sample_shape=N)
              for k in range(K)]
# add tf.sqrt
x = ParamMixture(pi, {'loc': mu, 'scale_diag': sigmasq},
                 MultivariateNormalDiag,
                 sample_shape=N)

# z = x.cat

# Inference

# find the parameters for each mixture model component
# qmu = Normal(tf.Variable(tf.random_normal([K, D])),
#              tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))

# qbeta = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([T]))),
#              tf.nn.softplus(tf.Variable(tf.random_normal([T]))))

# inference = ed.KLqp({beta: qbeta, mu: qmu}, data={x: x_train})


# number of samples
S = 100000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

# evaluating HMC
# https://github.com/blei-lab/edward/issues/408
# http://edwardlib.org/api/ed/HMC
inference = ed.HMC({beta: qbeta, mu: qmu}, data={x: x_train})
inference.initialize(step_size=0.1, n_steps=3)

# http://edwardlib.org/api/ed/SGLD
# inference = ed.SGLD({beta: qbeta, mu: qmu}, data={x: x_train})


# inference.initialize(step_size=0.1)


sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


# for _ in range(inference.n_iter):
#     info_dict = inference.update()
#     inference.print_progress(info_dict)
#     t = info_dict['t']

#     if t % inference.n_print == 0:
#       print(t)

# also works...
inference.run()

# for _ in range(inference.n_iter):
#     info_dict = inference.update()
#     inference.print_progress(info_dict)

#     t = info_dict['t']

#     if t % inference.n_print == 0:
#         print("Inferred cluster means:")
#         print(sess.run(qmu.mean()))

# Criticism
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).

# http://edwardlib.org/api/ed/MonteCarlo
total = 10000
burn_in = 4000
SC = total - burn_in

mu_sample = qmu.sample(total)[burn_in:, :, :]

mu_sample

x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                tf.ones([N, 1, 1, 1]) * 1.0)

x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()


print('found {} clusters'.format(len(np.unique(clusters))))

np.unique(clusters, return_counts=True)

# need to check how to plot this stuff....
plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters)
plt.title("Predicted cluster assignments")
plt.show()

plt.plot(qbeta.params.eval())
plt.show()


plt.plot(qmu.params.eval()[:, 3])
plt.show()
