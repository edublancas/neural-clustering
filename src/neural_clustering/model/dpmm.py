import os
from datetime import datetime
import logging

from edward.models import (Normal, MultivariateNormalDiag, Beta,
                           Gamma,  ParamMixture, Empirical, Categorical)
import edward as ed
import tensorflow as tf
import numpy as np
import yaml

from .util import get_commit_hash


logger = logging.getLogger(__name__)


def stick_breaking(v):
    remaining_pieces = tf.concat([tf.ones(1), tf.cumprod(1.0 - v)[:-1]], 0)
    weights = v * remaining_pieces
    return tf.concat([weights, [1.0 - tf.reduce_sum(weights)]], axis=0)


def fit(x_train, truncation_level, cfg, inference_alg=ed.KLqp,
        inference_params=None):
    """Fit a truncated DPMM model with Edward using Variational Inference

    Notes
    -----
    Only tested with KLqp
    """
    # tf.reset_default_graph()

    inference_name = inference_alg.__name__

    N, D = x_train.shape
    K = truncation_level
    T = K - 1

    # Model
    beta = Beta(tf.ones(T), tf.ones(T))
    pi = stick_breaking(beta)

    mu = Normal(tf.zeros(D), tf.ones(D), sample_shape=K)
    sigmasq = Gamma(tf.ones(D), tf.ones(D), sample_shape=K)

    # joint model
    x = ParamMixture(pi, {'loc': mu, 'scale_diag': tf.sqrt(sigmasq)},
                     MultivariateNormalDiag,
                     sample_shape=N)
    z = x.cat

    qbeta = Beta(tf.ones([T]), tf.nn.softplus(tf.Variable(tf.ones([T]))))
    qmu = Normal(tf.Variable(tf.zeros([K, D])), tf.ones([K, D]))
    qz = Categorical(tf.nn.softmax(tf.Variable(tf.zeros([N, K]))))

    inference = inference_alg({mu: qmu, z: qz, beta: qbeta},
                              data={x: x_train})

    if inference_params:
        inference.initialize(**inference_params)
    else:
        inference.initialize()

    sess = ed.get_session()
    init = tf.global_variables_initializer()
    init.run()

    # inference.run()
    for _ in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)

        t = info_dict['t']

        if t % inference.n_print == 0:
            print("Inferred cluster means:")
            print(sess.run(qmu.mean()))
            print("Beta")
            print(qbeta.concentration0.eval())

    # Save results
    saver = tf.train.Saver()

    timestamp = datetime.now()
    directory_name = '{}-DPMM'.format(timestamp.strftime('%d-%b-%Y@%H-%M-%S'))
    directory = os.path.join(cfg['root'], 'sessions', directory_name)
    os.makedirs(directory)

    output_path = os.path.join(directory, 'session.ckpt')
    saver.save(sess, output_path)
    logger.info('Session saved in {}'.format(output_path))

    output_path = os.path.join(directory, 'training.npy')
    np.save(output_path, x_train)
    logger.info('Training data saved in {}'.format(output_path))

    params = dict(model_type='DPMM',
                  truncation_level=truncation_level,
                  inference_algoritm=inference_name,
                  inference_params=inference_params,
                  timestamp=timestamp.isoformat(),
                  git_hash=get_commit_hash(),
                  name=directory_name)

    output_path = os.path.join(directory, 'params.yaml')

    with open(output_path, 'w') as f:
        yaml.dump(params, f)

    logger.info('Params saved in {}'.format(output_path))
