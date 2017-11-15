import edward as ed
from edward.models import Empirical
import tensorflow as tf

import os
import yaml


def restore_session(cfg, session_name):
    """Restores a session
    """
    path_to_session = os.path.join(cfg['root'],
                                   'sessions/{}'.format(session_name),
                                   'session.ckpt')

    path_to_params = os.path.join(cfg['root'],
                                  'sessions/{}'.format(session_name),
                                  'params.yaml')

    with open(path_to_params) as f:
        params = yaml.load(f)

    tf.reset_default_graph()

    D = params['D']
    K = params['K']
    S = params['S']

    qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
    qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

    saver = tf.train.Saver()
    sess = ed.get_session()
    saver.restore(sess, path_to_session)

    return qmu, qbeta
