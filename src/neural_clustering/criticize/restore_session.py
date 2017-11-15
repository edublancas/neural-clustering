import edward as ed
from edward.models import Empirical, Normal
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.reset_default_graph()

N, D = x_train.shape
K = 15
S = 100000

qmu = Empirical(tf.Variable(tf.zeros([S, K, D])))
qbeta = Empirical(tf.Variable(tf.zeros([S, K])))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

sess = ed.get_session()

with tf.Session() as sess:
  saver.restore(sess, "/ssd/data/eduardo/sessions/2017-11-14T21:39:56.006582.ckpt")

