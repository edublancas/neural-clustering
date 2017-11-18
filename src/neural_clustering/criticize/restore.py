import edward as ed
from edward.models import Empirical
import tensorflow as tf
import numpy as np

import os
import yaml


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

    return qmu, qbeta, x_train, params


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

    tf.reset_default_graph()

    N, D = x_train.shape
    K = params['truncation_level']
    T = params['samples']

    qpi = Empirical(tf.Variable(tf.ones([T, K]) / K))
    qmu = Empirical(tf.Variable(tf.zeros([T, K, D])))
    qsigmasq = Empirical(tf.Variable(tf.ones([T, K, D])))
    qz = Empirical(tf.Variable(tf.zeros([T, N], dtype=tf.int32)))

    saver = tf.train.Saver()
    sess = ed.get_session()
    saver.restore(sess, path_to_session)

    return qpi, qmu, qsigmasq, qz
