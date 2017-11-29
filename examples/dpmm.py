"""
Truncated DPMM using Edward.

It's been several attempts to implement truncated DPMM on Edward but there are
many problems. First, inference cannot properly be made with MCMC since Edward
throws this error when trying to make inference:

ValueError: 'transform' does not handle supports of type 'categorical'

Gibbs throws a different error:

NotImplementedError: Conditional distribution has sufficient statistics (...)

Seems like the only options is to use variational inference. Here I am using
KLqp, I bumped into several errors which have already been reported in
discourse, I managed to have a working truncated DPMM model but inference
is still bad, not sure exactly why. Since I had limited time and invested
too much time on this I decided to also use GMM.

Based on:
https://gist.github.com/dustinvtran/d8cc112636b219776621444324919928
http://edwardlib.org/tutorials/unsupervised
http://docs.pymc.io/notebooks/dp_mix.html
https://discourse.edwardlib.org/t/dpm-model-for-clustering/97/6
https://discourse.edwardlib.org/t/variational-inference-for-dirichlet-process-mixtures/251

Cannot use MetropolisHastings
https://github.com/blei-lab/edward/blob/master/examples/mixture_gaussian_mh.py
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
K = 3
T = 2  # truncation

sess = ed.get_session()

x_train = build_toy_dataset(N)

# plot toy dataset
sns.jointplot(x_train[:, 0], x_train[:, 1], kind='scatter')
plt.show()


# Model
beta = Beta(tf.ones(T), tf.ones(T))

# alpha = Gamma(tf.ones(T), tf.ones(T))
# beta = Beta(tf.ones(T), alpha)

pi = stick_breaking(beta)

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
mu

# m = np.array([[10.0, 10.0], [0.0, 0.0], [-10.0, -10.0]]).astype('float32')
# mu = Normal(m, tf.ones((K, D)))

# sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)
sigmasq = tf.ones((K, D))

# pi = tf.constant([0.39, 0.01, 0.60])
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)
z = x.cat

# plot prior
x_original = x.sample(500).eval()
sns.jointplot(x_original[:, 0], x_original[:, 1], kind='kde')
plt.show()


# Inference with KLqp
# qalpha = Gamma(tf.Variable(1.0), tf.Variable(1.0))
# qsigmasq = InverseGamma(tf.Variable(tf.zeros([K, D])), tf.Variable(tf.zeros([K, D])))

qbeta = Beta(tf.ones([T]), tf.nn.softplus(tf.Variable(tf.ones([T]))))

m = np.array([[-15.0, -10.0], [5.0, 0.0], [20.0, 20.0]]).astype('float32')
qmu = Normal(tf.Variable(m),
             tf.nn.softplus(tf.Variable(tf.zeros([K, D]))))

qz = Categorical(tf.nn.softmax(tf.Variable(tf.zeros([N, K]))))


inference = ed.KLqp({mu: qmu, z: qz}, data={x: x_train})

inference = ed.KLqp({mu: qmu, z: qz, beta: qbeta}, data={x: x_train})
inference = ed.KLqp({mu: qmu, beta: qbeta}, data={x: x_train})

# inference = ed.KLqp({mu: qmu, z: qz, beta: qbeta, sigmasq: qsigmasq}, data={x: x_train})

# inference = ed.KLqp({beta: qbeta, mu: qmu, z: qz, sigmasq: qsigmasq}, data={x: x_train})
# inference = ed.KLqp({beta: qbeta, mu: qmu}, data={x: x_train})
# inference = ed.KLqp({mu: qmu, z: qz, beta: qbeta, alpha: qalpha}, data={x: x_train})
# inference = ed.KLqp({mu: qmu, z: qz, alpha: qalpha}, data={x: x_train})

inference.initialize(n_samples=3, n_iter=5000, n_print=100)

init = tf.global_variables_initializer()
init.run()


for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    t = info_dict['t']

    if t % inference.n_print == 0:
        print("Inferred cluster means:")
        print(sess.run(qmu.mean()))
        print("Beta")
        print(qbeta.concentration0.eval())

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
