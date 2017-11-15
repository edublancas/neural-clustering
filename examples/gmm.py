"""
Gaussian mixture modelusing Edward

Based on:
http://edwardlib.org/tutorials/unsupervised
"""
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import (Dirichlet, InverseGamma, MultivariateNormalDiag,
                           Normal, ParamMixture, Empirical)
import matplotlib.pyplot as plt


ed.set_seed(0)


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[5, 5], [-5, -5]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
    x = np.zeros((N, 2), dtype=np.float32)

    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


##############
# Parameters #
##############

# number of data points
N = 500
# number of components
K = 2
# dimensionality of data
D = 2
# number of MCMC samples
T = 5000


x_train = build_toy_dataset(N)

# plot toy dataset
plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()


#########
# Model #
#########

pi = Dirichlet(tf.ones(K))
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                 MultivariateNormalDiag,
                 sample_shape=N)
z = x.cat


#############
# Inference #
#############


qpi = Empirical(tf.Variable(tf.ones([T, K]) / K))
qmu = Empirical(tf.Variable(tf.zeros([T, K, D])))
qsigmasq = Empirical(tf.Variable(tf.ones([T, K, D])))
qz = Empirical(tf.Variable(tf.zeros([T, N], dtype=tf.int32)))

# http://edwardlib.org/api/ed/Gibbs
inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},
                     data={x: x_train})
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


#############
# Criticism #
#############

SC = 200

mu_sample = qmu.sample(SC)
sigmasq_sample = qsigmasq.sample(SC)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])

log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)

clusters = tf.argmax(log_liks, 1).eval()

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters)
plt.title("Predicted cluster assignments")
plt.show()


x_pred = ed.copy(x, {pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz})

# log-likelihood performance
ed.evaluate('log_likelihood', data={x_pred: x_train})

# ppc
y_rep, y = ed.ppc(lambda xs, mus: tf.reduce_mean(xs[x_pred]),
                  data={x_pred: x_train},
                  n_samples=1000)

ed.ppc_stat_hist_plot(y[0], y_rep,
                      stat_name=r'$T \equiv$mean', bins=10)

plt.show()
