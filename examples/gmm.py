"""
Gaussian mixture model using Edward

Based on:
http://edwardlib.org/tutorials/unsupervised
"""
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import (Dirichlet, InverseGamma, MultivariateNormalDiag,
                           Normal, ParamMixture, Empirical)
import matplotlib.pyplot as plt
import seaborn as sns


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
sns.jointplot(x_train[:, 0], x_train[:, 1], kind='kde')
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

# turorial http://edwardlib.org/tutorials/inference
# api http://edwardlib.org/api/inference

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

    # if t % inference.n_print == 0:
    #     print("\nInferred cluster means:")
    #     print(sess.run(running_cluster_means, {t_ph: t - 1}))


#############
# Criticism #
#############

# visualizing parameters
plt.plot(qpi.params.eval())
plt.show()

qmu_params = qmu.params.eval()
plt.plot(qmu_params[:, :, 0])
plt.show()

qmu_params = qmu.params.eval()
plt.plot(qmu_params[:, 1, :])
plt.show()

# tutorial on criticism: http://edwardlib.org/tutorials/criticism

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


# plotting pis
# https://seaborn.pydata.org/tutorial/distributions.html
pi_sample = pi.sample(5000).eval()
qpi_sample = qpi.sample(5000).eval()

sns.distplot(pi_sample[:, 0])
plt.show()

sns.distplot(qpi_sample[:, 0])
plt.show()


sns.distplot(pi_sample[:, 1])
plt.show()

sns.distplot(qpi_sample[:, 1])
plt.show()

# plottng mus
mu_sample = mu.sample(5000).eval()
qmu_sample = qmu.sample(5000).eval()

sns.distplot(mu_sample[:, 0])
plt.show()

# get obsevations from mixture component means
first = qmu_sample[:, 0, :]
second = qmu_sample[:, 1, :]

sns.jointplot(first[:, 0], first[:, 1], kind='kde')
plt.show()

sns.jointplot(second[:, 0], second[:, 1], kind='kde')
plt.show()

# sample from mixture model with prior parameters
x_original = x.sample(500).eval()
sns.jointplot(x_original[:, 0], x_original[:, 1], kind='kde')
plt.show()


# posterior predictive distribution
# this doesnt look right...
x_pred = ed.copy(x, {pi: qpi, mu: qmu, sigmasq: qsigmasq})
# this looks better...
x_pred = ParamMixture(qpi, {'loc': qmu, 'scale_diag': tf.sqrt(qsigmasq)},
                      MultivariateNormalDiag,
                      sample_shape=N)

x_pred_sample = x_pred.sample(500).eval()
sns.jointplot(x_pred_sample[:, 0], x_pred_sample[:, 1], kind='kde')
plt.show()

# predictive distribution
mus = np.mean(qmu.sample(1000).eval(), axis=0)
stds = np.sqrt(np.mean(qsigmasq.sample(1000).eval(), axis=0))
pis = np.mean(qpi.sample(1000).eval(), axis=0)

x_ = ParamMixture(pis, {'loc': mus, 'scale_diag': stds},
                  MultivariateNormalDiag,
                  sample_shape=N)
sample = x_.sample(500).eval()
sns.jointplot(sample[:, 0], sample[:, 1], kind='kde')
plt.show()


# log-likelihood performance
ed.evaluate('log_likelihood', data={x: x_train})
ed.evaluate('log_likelihood', data={x_pred: x_train})
ed.evaluate('log_likelihood', data={x_: x_train})

# ppc
y_rep, y = ed.ppc(lambda xs, mus: tf.reduce_mean(xs[x]),
                  data={x: x_train},
                  n_samples=1000)
ed.ppc_stat_hist_plot(y[0], y_rep,
                      stat_name=r'$T \equiv$mean', bins=10)
plt.show()

y_rep, y = ed.ppc(lambda xs, mus: tf.reduce_mean(xs[x_pred]),
                  data={x_pred: x_train},
                  n_samples=1000)
ed.ppc_stat_hist_plot(y[0], y_rep,
                      stat_name=r'$T \equiv$mean', bins=10)
plt.show()

y_rep, y = ed.ppc(lambda xs, mus: tf.reduce_mean(xs[x_]),
                  data={x_: x_train},
                  n_samples=1000)
ed.ppc_stat_hist_plot(y[0], y_rep,
                      stat_name=r'$T \equiv$mean', bins=10)
plt.show()
