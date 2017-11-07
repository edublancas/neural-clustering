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


# ed.set_seed(0)

N = 2000
D = 2
T = K = 5  # truncation level in DP

x_train = build_toy_dataset(N)

plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()


# Model


beta = Beta(tf.ones(T), tf.ones(T))
pi = stick_breaking(beta)

cat = Categorical(probs=pi, sample_shape=N)

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)

sigma = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


# components = [MultivariateNormalDiag(mu[k], sigma[k], sample_shape=N)
#               for k in range(K)]

x = ParamMixture(pi, {'loc': mu, 'scale_diag': sigmasq},
                 MultivariateNormalDiag,
                 sample_shape=N)

# z = x.cat

# Inference


# qmu = Normal(tf.Variable(tf.random_normal([K, D])),
#              tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))
# qbeta = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([T]))),
#              tf.nn.softplus(tf.Variable(tf.random_normal([T]))))

# doesnt work - nans
# inference = ed.KLqp({beta: qbeta, mu: qmu}, data={x: x_train})


# number of samples
S = 10000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qsigmasq = Empirical(tf.Variable(tf.zeros([S, K, D])))

qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

# doesnt work - KeyError
# inference = ed.SGHMC({beta: qbeta, mu: qmu}, data={x: x_train})
# doesnt work - NotImplementedError
# inference = ed.Gibbs({beta: qbeta, mu: qmu}, data={x: x_train})

# works
# added  sigmasq
inference = ed.HMC({beta: qbeta, mu: qmu}, #sigmasq: qsigmasq},
                   data={x: x_train})

# also works but empirically HMC works better...
# inference = ed.SGLD({beta: qbeta, mu: qmu}, data={x: x_train})


inference.initialize()

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    t = info_dict['t']

    # if t % inference.n_print == 0:
    #     print("Inferred cluster means:")
    #     print(sess.run(qmu.mean()))

# Criticism
# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
SC = 1000

# each qmu is the mu vector for each component (?)
mu_sample = qmu.sample(SC)
# sigmasq_sample = qsigmasq.sample(SC)

x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                tf.ones([N, 1, 1, 1]) * 0.1) # tf.sqrt(sigmasq_sample)

x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])
# is it okay to do this?
# x_broadcasted = tf.to_float(x_broadcasted)

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()

print('Found {} clusters'.format(len(np.unique(clusters))))

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters)
plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()
