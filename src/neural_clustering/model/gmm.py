import os
from datetime import datetime
import logger

import yaml
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import (Dirichlet, InverseGamma, MultivariateNormalDiag,
                           Normal, ParamMixture, Empirical)

from .util import get_commit_hash


def fit(x_train, k, cfg, samples, inference_alg=ed.Gibbs,
        inference_params=None):
    """Fits a GMM using Edward
    """
    inference_name = inference_alg.__name__

    N, D = x_train.shape

    K = k
    T = samples

    # Model

    pi = Dirichlet(tf.ones(K))
    mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
    sigmasq = InverseGamma(tf.ones(D), tf.ones(D), sample_shape=K)


    x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                     MultivariateNormalDiag,
                     sample_shape=N)
    z = x.cat

    # Inference

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

    inference.run()

    # Save results
    saver = tf.train.Saver()

    timestamp = datetime.now()
    directory_name = '{}-GMM'.format(timestamp.strftime('%d-%b-%Y@%H-%M-%S'))
    directory = os.path.join(cfg['root'], 'sessions', directory_name)
    os.makedirs(directory)

    output_path = os.path.join(directory, 'session.ckpt')
    saver.save(sess, output_path)
    logger.info('Session saved in {}'.format(output_path))

    output_path = os.path.join(directory, 'training.npy')
    np.save(output_path, x_train)
    logger.info('Training data saved in {}'.format(output_path))

    params = dict(model_type='GMM',
                  k=k,
                  inference_algoritm=inference_name,
                  samples=samples, inference_params=inference_params,
                  timestamp=timestamp.isoformat(),
                  git_hash=get_commit_hash())

    output_path = os.path.join(directory, 'params.yaml')

    with open(output_path, 'w') as f:
        yaml.dump(params, f)

    logger.info('Params saved in {}'.format(output_path))
