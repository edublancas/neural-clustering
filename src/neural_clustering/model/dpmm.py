import os
from datetime import datetime
import logging

from edward.models import (Normal, MultivariateNormalDiag, Beta,
                           InverseGamma,  ParamMixture, Empirical)
import edward as ed
import tensorflow as tf
import numpy as np
import yaml


logger = logging.getLogger(__name__)


def stick_breaking(v):
    remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
    return v * remaining_pieces


def fit(x_train, truncation_level, cfg):
    """Fit a truncated DPMM model with Edward
    """
    N, D = x_train.shape
    T = K = truncation_level

    # Model
    beta = Beta(tf.ones(T), tf.ones(T))
    pi = stick_breaking(beta)

    mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
    sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)

    x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                     MultivariateNormalDiag,
                     sample_shape=N)

    # Inference with HMC - works (SGLD also works)
    S = 10000

    qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
    qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

    inference = ed.HMC({beta: qbeta, mu: qmu}, data={x: x_train})
    inference.initialize()

    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    inference.run()

    # Save results
    saver = tf.train.Saver()

    timestamp = datetime.now().strftime('%d-%b-%Y@%H-%M-%S')
    directory_name = '{}-DPMM'.format(timestamp)
    directory = os.path.join(cfg['root'], 'sessions', directory_name)
    os.makedirs(directory)

    output_path = os.path.join(directory, 'session.ckpt')
    saver.save(sess, output_path)
    logger.info('Session saved in {}'.format(output_path))

    output_path = os.path.join(directory, 'training.npy')
    np.save(output_path, x_train)
    logger.info('Training data saved in {}'.format(output_path))

    params = dict(truncation_level=truncation_level,
                  samples=S)

    output_path = os.path.join(directory, 'params.yaml')

    with open(output_path, 'w') as f:
        yaml.dump(params, f)

    logger.info('Params saved in {}'.format(output_path))
