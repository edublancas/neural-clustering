"""
Truncated DPMM using Edward

Based on:
https://gist.github.com/dustinvtran/d8cc112636b219776621444324919928
http://edwardlib.org/tutorials/unsupervised
http://docs.pymc.io/notebooks/dp_mix.html
https://discourse.edwardlib.org/t/dpm-model-for-clustering/97/6
"""
from edward.models import (Normal, MultivariateNormalDiag, Beta,
                           InverseGamma,  ParamMixture, Empirical,
                           Gamma, Categorical)
import edward as ed
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[5.0, 5.0], [-5.0, -5.0]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
    x = np.zeros((N, 2), dtype=np.float32)

    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


def stick_breaking(v):
    remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
    weights = v * remaining_pieces
    return tf.concat([weights, [1.0 - tf.reduce_sum(weights)]], axis=0)


# def stick_breaking(v):
#     remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
#     weights = v * remaining_pieces
#     return weights


ed.set_seed(0)

N = 500
D = 2
T = K = 3  # truncation level

x_train = build_toy_dataset(N)

# plot toy dataset
plt.scatter(x_train[:, 0], x_train[:, 1])
sns.jointplot(x_train[:, 0], x_train[:, 1], kind='kde')
plt.show()

# Model
# beta = Beta(tf.ones(T - 1), tf.ones(T - 1))

# sns.distplot(beta.sample(1000).eval()[:, 0])
# plt.show()


# sns.distplot(Beta(1.0, 1.0).sample(100).eval())
# plt.show()

alpha = Gamma(1.0, 1.0)
beta = Beta(tf.ones(T - 1), tf.ones(T - 1) * alpha)
beta

# s = beta.sample(1)[0, :]
# stick_breaking(s).eval()

pi = stick_breaking(beta)
pi

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
mu

# m = np.array([[10.0, 10.0], [0.0, 0.0], [-10.0, -10.0]]).astype('float32')
# mu = Normal(m, tf.ones((K, D)))
# mu

# sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)
# sigmasq
# sigmasq = InverseGamma(tf.ones((K, D)), tf.ones((K, D)))
# sigmasq

sigmasq = tf.ones((K, D))

pi = tf.constant([0.39, 0.01, 0.60])
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)
z = x.cat

# original model
x_original = x.sample(500).eval()
sns.jointplot(x_original[:, 0], x_original[:, 1], kind='kde')
plt.show()


# Inference with KLqp - getting nans
qmu = Normal(tf.Variable(tf.random_normal([K, D])),
             tf.nn.softplus(tf.Variable(tf.random_normal([K, D]))))
# qbeta = Beta(tf.nn.softplus(tf.Variable(tf.random_normal([T - 1]))),
             # tf.nn.softplus(tf.Variable(tf.random_normal([T - 1]))))

# inference = ed.KLqp({beta: qbeta, mu: qmu}, data={x: x_train})
inference = ed.KLqp({mu: qmu}, data={x: x_train})
inference.initialize(n_samples=5, n_iter=1000, n_print=25)

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
S = 5000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
# qbeta = Empirical(tf.Variable(tf.zeros([S, K])))
qalpha = Empirical(tf.Variable(tf.ones(S)))
qz = Empirical(tf.Variable(tf.zeros([S, N], dtype=tf.int32)))

# inference = ed.SGLD({alpha: qalpha, mu: qmu}, data={x: x_train})
# inference = ed.SGLD({beta: qbeta, mu: qmu}, data={x: x_train})

# inference = ed.Gibbs({mu: qmu, z: qz}, data={x: x_train})
inference = ed.SGLD({alpha: qalpha, mu: qmu, z: qz}, data={x: x_train})


# galpha = Gamma(1.0, 1.0)
# gmu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
# gz = Categorical(tf.zeros([N, K]))

# nope https://github.com/blei-lab/edward/blob/master/examples/mixture_gaussian_mh.py
# inference = ed.MetropolisHastings(latent_vars={alpha: qalpha, mu: qmu, z: qz},
#                                   proposal_vars={alpha: galpha, mu: gmu, z: gz},
#                                   data={x: x_train})

# inference = ed.MetropolisHastings(latent_vars={mu: qmu, z: qz},
#                                   proposal_vars={mu: gmu, z: gz},
#                                   data={x: x_train})


inference.initialize()

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()


t_ph = tf.placeholder(tf.int32, [])
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)

for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']

    if t % inference.n_print == 0:
        print("\nInferred cluster means:")
        print(sess.run(running_cluster_means, {t_ph: t - 1}))


# inference.run(step_size=0.1, n_steps=2)
inference.run()

# Criticism

# plotting params
plt.plot(qalpha.params.eval())
plt.show()

sns.distplot(qalpha.sample(1000).eval())
plt.show()

# plt.plot(qbeta.params.eval())
# plt.show()

qmu_params = qmu.params.eval()
qmu_params[-1:]

plt.plot(qmu_params[:, 2, :])
plt.show()


# posterior predictive
qbeta = Beta(tf.ones(T - 1), tf.ones(T - 1) * qalpha)
qpi = stick_breaking(beta)
x_pred = ParamMixture(pi,
                      {'loc': qmu, 'scale_diag': tf.sqrt(sigmasq)},
                      MultivariateNormalDiag,
                      sample_shape=N)

x_pred_sample = x_pred.sample(500).eval()
sns.jointplot(x_pred_sample[:, 0], x_pred_sample[:, 1], kind='kde')
plt.show()


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
