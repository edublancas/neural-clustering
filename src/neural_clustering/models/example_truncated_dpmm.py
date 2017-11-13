"""
Truncated DPMM
"""
# https://discourse.edwardlib.org/t/dpm-model-for-clustering/97/6
from edward.models import (Normal, MultivariateNormalDiag, Categorical, Beta,
                           InverseGamma,  ParamMixture, Empirical, Gamma)

import edward as ed
import numpy as np
import tensorflow as tf


from matplotlib import pyplot as plt


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[5, 5], [-5, -5]]
    #stds = [[0.1, 0.1], [0.1, 0.1]]
    stds = [[1, 1], [1, 1]]
    x = np.zeros((N, 2), dtype=np.float32)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


def stick_breaking(v):
    remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
    return v * remaining_pieces


# ed.set_seed(0)

N = 1000
D = 2
T = K = 10  # truncation level in DP

x_train = build_toy_dataset(N)

plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()


# Model
idxs = np.random.choice(range(N), size=K, replace=False)
mu_init = x_train[idxs]

plt.scatter(mu_init[:, 0], mu_init[:, 1])
plt.show()


beta = Beta(tf.ones(T), tf.ones(T))
pi = stick_breaking(beta)

# cat = Categorical(probs=pi, sample_shape=N)


# mu priors - centered at 0 and with scale (stddev) 1
# mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
mu = Normal(mu_init, tf.ones([K, D]))

# sigmasq priors - inverse gamma  with alpha and beta 1
# InverseGamma?
sigmasq = Gamma(tf.ones(D), tf.ones(D), sample_shape=K)

# sigma = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


# components = [MultivariateNormalDiag(mu[k], sigma[k], sample_shape=N)
#               for k in range(K)]

# tf.sqrt??
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
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
S = 30000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
# qsigmasq = Empirical(tf.Variable(tf.ones([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, K])))


# works
# params = {beta: qbeta, mu: qmu, sigmasq: qsigmasq}
params2 = {beta: qbeta, mu: qmu}


# doesnt work - NotImplementedError
# inference = ed.Gibbs(params, data={x: x_train})
# doesnt work - KeyError when initialize
# inference = ed.SGHMC(params, data={x: x_train})

# need tuning for step_size and n_steps
# https://discourse.edwardlib.org/t/confusion-about-empirical-distributions-in-edward/154
# inference = ed.HMC(params2, data={x: x_train})
inference = ed.SGLD(params2, data={x: x_train})

# inference = ed.MetropolisHastings(params, data={x: x_train})


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
SC = 100

# draw SC samples from the posterior
mu_sample = qmu.sample(SC)
# sigmasq_sample = qsigmasq.sample(SC)

# sigmasq_sample.eval()

# this is introducing nans!
# tf.sqrt(sigmasq_sample).eval()

x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                tf.ones([N, 1, 1, 1]) * 1.0) #tf.sqrt(sigmasq_sample)

x_post.eval()

x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])
# is it okay to do this?
# x_broadcasted = tf.to_float(x_broadcasted)

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)

# NANS?????
log_liks.eval()

log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

# Choose the cluster with the highest likelihood for each data point.
clusters = tf.argmax(log_liks, 1).eval()

print('Found {} clusters'.format(len(np.unique(clusters))))

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters)
# plt.axis([-3, 3, -3, 3])
plt.title("Predicted cluster assignments")
plt.show()
