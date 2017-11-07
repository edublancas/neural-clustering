"""
Truncated DPMM
"""
from .util import stick_breaking


from edward.models import (Normal, MultivariateNormalDiag, Categorical, Beta,
                           InverseGamma,  ParamMixture, Empirical)
import edward as ed
import tensorflow as tf


def DPMM(x_train, truncation_level, simulation_samples):
    N, D = x_train.shape
    T = K = truncation_level
    S = simulation_samples

    # Model
    beta = Beta(tf.ones(T), tf.ones(T))

    pi = stick_breaking(beta)

    cat = Categorical(probs=pi, sample_shape=N)

    mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
    sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)

    sigma = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)

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

    qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
    qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

    # doesnt work - KeyError
    inference = ed.SGHMC({beta: qbeta, mu: qmu}, data={x: x_train})
    # doesnt work - NotImplementedError
    inference = ed.Gibbs({beta: qbeta, mu: qmu}, data={x: x_train})

    # works
    inference = ed.HMC({beta: qbeta, mu: qmu}, data={x: x_train})
    inference = ed.SGLD({beta: qbeta, mu: qmu}, data={x: x_train})

    inference.initialize()

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
