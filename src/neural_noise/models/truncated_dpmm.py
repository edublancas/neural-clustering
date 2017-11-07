"""
Truncated DPMM
"""
from .util import stick_breaking


from edward.models import (Normal, MultivariateNormalDiag, Categorical, Beta,
                           InverseGamma,  ParamMixture, Empirical)
import edward as ed
import tensorflow as tf


import os
from neural_noise import config
import numpy as np

cfg = config.load('config.yaml')

x_train = np.load(os.path.join(cfg['root'], 'training.npy'))
x_train.shape

truncation_level = 20
simulation_samples = 10000


def DPMM(x_train, truncation_level, simulation_samples):
    """
    Truncated DPMM with means initialized with mean 0 and constant location 1
    """
    N, D = x_train.shape
    T = K = truncation_level
    S = simulation_samples

    # Model
    beta = Beta(tf.ones(T), tf.ones(T))

    pi = stick_breaking(beta)

    mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
    sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)

    x = ParamMixture(pi, {'loc': mu, 'scale_diag': sigmasq},
                     MultivariateNormalDiag,
                     sample_shape=N)

    qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
    qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

    inference = ed.HMC({beta: qbeta, mu: qmu}, data={x: x_train})
    # inference = ed.SGLD({beta: qbeta, mu: qmu}, data={x: x_train})

    inference.initialize()

    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    for _ in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)
