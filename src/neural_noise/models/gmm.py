"""
Gaussian mixture model with Edward
"""
import tensorflow as tf
import edward as ed
from edward.models import (Dirichlet, InverseGamma, MultivariateNormalDiag,
                           Normal, ParamMixture, Empirical)


# x_train = build_toy_dataset(N)

##############
# Parameters #
##############

# number of data points
N = 500
# number of components
K = 8
# dimensionality of data
D = 2
# number of MCMC samples
T = 10000


##########
# Priors #
##########

# prior distribution of the pi_k
pi = Dirichlet(tf.ones(K))

# prior over the normal means
mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)

# prior over the variances
sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


#########
# Model #
#########

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


inference = ed.Gibbs({pi: qpi, mu: qmu, sigmasq: qsigmasq, z: qz},
                     data={x: x_train})
inference.initialize()

sess = ed.get_session()
tf.global_variables_initializer().run()

t_ph = tf.placeholder(tf.int32, [])
running_cluster_means = tf.reduce_mean(qmu.params[:t_ph], 0)


for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']

    if t % inference.n_print == 0:
        print("\nInferred cluster means:")
        print(sess.run(running_cluster_means, {t_ph: t - 1}))


# Calculate likelihood for each data point and cluster assignment,
# averaged over many posterior samples. ``x_post`` has shape (N, 100, K, D).
mu_sample = qmu.sample(100)
sigmasq_sample = qsigmasq.sample(100)
x_post = Normal(loc=tf.ones([N, 1, 1, 1]) * mu_sample,
                scale=tf.ones([N, 1, 1, 1]) * tf.sqrt(sigmasq_sample))
x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, 100, K, 1])

# Sum over latent dimension, then average over posterior samples.
# ``log_liks`` ends up with shape (N, K).
log_liks = x_post.log_prob(x_broadcasted)
log_liks = tf.reduce_sum(log_liks, 3)
log_liks = tf.reduce_mean(log_liks, 1)


clusters = tf.argmax(log_liks, 1).eval()
