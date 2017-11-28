import edward as ed
from edward.models import Empirical, ParamMixture, MultivariateNormalDiag
import tensorflow as tf
import numpy as np

import os
import yaml


def experiment(cfg, session_name):
    """Restores an experiment
    """
    # tf.reset_default_graph()

    path_to_params = os.path.join(cfg['root'],
                                  'sessions/{}'.format(session_name),
                                  'params.yaml')

    with open(path_to_params) as f:
        params = yaml.load(f)

    restore_function = dpmm if params['model_type'] == 'DPMM' else gmm

    return restore_function(cfg, session_name)


def dpmm(cfg, session_name):
    """Restores a DPMM session

    Returns
    -------
    qmu
    qbeta
    x_train
    params
    """
    path_to_session = os.path.join(cfg['root'],
                                   'sessions/{}'.format(session_name),
                                   'session.ckpt')

    path_to_training = os.path.join(cfg['root'],
                                    'sessions/{}'.format(session_name),
                                    'training.npy')

    path_to_params = os.path.join(cfg['root'],
                                  'sessions/{}'.format(session_name),
                                  'params.yaml')

    x_train = np.load(path_to_training)

    with open(path_to_params) as f:
        params = yaml.load(f)

    tf.reset_default_graph()

    _, D = x_train.shape
    K = params['truncation_level']
    S = params['samples']

    qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
    qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

    saver = tf.train.Saver()
    sess = ed.get_session()
    saver.restore(sess, path_to_session)

    return dict(qmu=qmu, qbeta=qbeta, x_train=x_train, params=params)


def gmm(cfg, session_name):
    """Restores a GMM session

    Returns
    -------
    qmu
    qbeta
    x_train
    params
    """
    path_to_session = os.path.join(cfg['root'],
                                   'sessions/{}'.format(session_name),
                                   'session.ckpt')

    path_to_training = os.path.join(cfg['root'],
                                    'sessions/{}'.format(session_name),
                                    'training.npy')

    path_to_params = os.path.join(cfg['root'],
                                  'sessions/{}'.format(session_name),
                                  'params.yaml')

    x_train = np.load(path_to_training)

    with open(path_to_params) as f:
        params = yaml.load(f)

    N, D = x_train.shape
    K = params['k']
    T = params['samples']

    qpi = Empirical(tf.Variable(tf.ones([T, K]) / K))
    qmu = Empirical(tf.Variable(tf.zeros([T, K, D])))
    qsigmasq = Empirical(tf.Variable(tf.ones([T, K, D])))
    qz = Empirical(tf.Variable(tf.zeros([T, N], dtype=tf.int32)))

    saver = tf.train.Saver()
    sess = ed.get_session()
    saver.restore(sess, path_to_session)

    x_pred = ParamMixture(qpi, {'loc': qmu, 'scale_diag': tf.sqrt(qsigmasq)},
                          MultivariateNormalDiag, sample_shape=N)

    return dict(qpi=qpi, qmu=qmu, qsigmasq=qsigmasq, qz=qz, x_train=x_train,
                params=params, x_pred=x_pred)
