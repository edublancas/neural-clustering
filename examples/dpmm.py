"""
Truncated DPMM using Edward.

It's been several attempts to implement truncated DPMM on Edward but there are
many problems. First, inference in the joint version (using ParamMixture)
cannot properly be made with MCMC since Edward throws this error when
trying to make inference:

ValueError: 'transform' does not handle supports of type 'categorical'

Gibbs throws a different error:

NotImplementedError: Number of components must be statically determined.

Seems like the only option (for the joint version) is to use variational
inference. I tried using KLqp, but bumped into several errors which have
already been reported in discourse, I managed to have a working truncated DPMM
with KLqp but inference is bad, more on that here:
https://discourse.edwardlib.org/t/gaussian-mixture-with-covariance-matrix-via-klqp-method/277/2

Bus as I mention before, Gibbs does not work in the joint model.
I also tried the collapsed version (using Mixture) but inference is bad,
and with Gibbs I get this error:

NotImplementedError: conjugate_log_prob not implemented for
<class 'abc.Mixture'>

Also, there seems to be an error in Edward, when truncation level > 5

References:
https://gist.github.com/dustinvtran/d8cc112636b219776621444324919928
http://edwardlib.org/tutorials/unsupervised
http://docs.pymc.io/notebooks/dp_mix.html
https://discourse.edwardlib.org/t/dpm-model-for-clustering/97/6
https://discourse.edwardlib.org/t/variational-inference-for-dirichlet-process-mixtures/251

Mixtures and MetropolisHastings:
https://github.com/blei-lab/edward/blob/master/examples/mixture_gaussian_mh.py
"""
from edward.models import (Normal, MultivariateNormalDiag, Beta,
                           ParamMixture, Empirical, Gamma, Categorical,
                           Mixture)
import edward as ed
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[30.0, 30.0], [-30.0, -30.0]]
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
T = 2

sess = ed.get_session()

x_train = build_toy_dataset(N)

# plot toy dataset
sns.jointplot(x_train[:, 0], x_train[:, 1], kind='scatter')
plt.show()


# Model
beta = Beta(tf.ones(T), tf.ones(T))

alpha = Gamma(tf.ones(T), tf.ones(T))
beta = Beta(tf.ones(T), alpha)

pi = stick_breaking(beta)

pi = tf.constant([0.39, 0.01, 0.60])

mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)

# m = np.array([[-15.0, -10.0], [5.0, 0.0], [20.0, 20.0]]).astype('float32')
# mu = Normal(m, tf.ones((K, D)))

# sigmasq = Gamma(tf.ones(D), tf.ones(D), sample_shape=K)
sigmasq = tf.ones((K, D))

# joint model
x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)
z = x.cat

# collapsed model
cat = Categorical(probs=pi, sample_shape=N)
components = [MultivariateNormalDiag(tf.zeros(D), tf.ones(D), sample_shape=N)
              for k in range(K)]
x = Mixture(cat, components, sample_shape=N)


# plot prior
x_original = x.sample(500).eval()
sns.jointplot(x_original[:, 0], x_original[:, 1], kind='kde')
plt.show()

# Inference with KLqp
qsigmasq = Gamma(tf.Variable(tf.zeros([K, D])),
                 tf.Variable(tf.zeros([K, D])))
qalpha = Gamma(tf.Variable(tf.ones(T)), tf.Variable(tf.ones(T)))
qbeta = Beta(tf.ones([T]), tf.nn.softplus(tf.Variable(tf.ones([T]))))
qmu = Normal(tf.Variable(tf.zeros([K, D])), tf.ones([K, D]))
qz = Categorical(tf.nn.softmax(tf.Variable(tf.zeros([N, K]))))

inference = ed.KLqp({mu: qmu}, data={x: x_train})

inference = ed.KLqp({mu: qmu, beta: qbeta}, data={x: x_train})

inference = ed.KLqp({mu: qmu, alpha: qalpha}, data={x: x_train})

inference = ed.KLqp({mu: qmu, z: qz, alpha: qalpha}, data={x: x_train})

inference = ed.KLqp({mu: qmu, z: qz, beta: qbeta, sigmasq: qsigmasq},
                    data={x: x_train})

inference = ed.KLqp({mu: qmu, z: qz, beta: qbeta},
                    data={x: x_train})

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


S = 5000
qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, T])))
qalpha = Empirical(tf.Variable(tf.zeros([S, T])))
qz = Empirical(tf.Variable(tf.zeros([S, N], dtype=tf.int32)))

# below doesn't work, so I need to use the collapsed version...
# inference = ed.SGLD({mu: qmu, z: qz}, data={x: x_train})

inference = ed.SGLD({mu: qmu, beta: qbeta}, data={x: x_train})
inference = ed.SGLD({mu: qmu, alpha: qalpha}, data={x: x_train})

# inference for joint model throws NotImplementedError: Number of components
# must be statically determined.
inference = ed.Gibbs({mu: qmu, z: qz, beta: qbeta}, data={x: x_train})

# inference for collapsed model throws error
# NotImplementedError: conjugate_log_prob not implemented for
# <class 'abc.Mixture'>
inference = ed.Gibbs({mu: qmu, beta: qbeta}, data={x: x_train})

inference.initialize()

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
