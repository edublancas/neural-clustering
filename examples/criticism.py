"""
Based on: https://gist.github.com/dustinvtran/734a0fbe982ded5e53ecdacc4feb0ebb
"""
import edward as ed
import numpy as np
import tensorflow as tf
import seaborn as sns

import matplotlib.pyplot as plt
from edward.models import Bernoulli, Beta
from edward.criticisms import ppc_stat_hist_plot

ed.set_seed(42)

sns.set_style('ticks')

y = np.random.randn(20)
y_rep = np.random.randn(20, 20)

ed.criticisms.ppc_density_plot(y, y_rep)
plt.show()

# DATA
x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

# MODEL
p = Beta(1.0, 1.0)
x = Bernoulli(tf.ones(10) * p)

# INFERENCE
qp_a = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp_b = tf.nn.softplus(tf.Variable(tf.random_normal([])))
qp = Beta(qp_a, qp_b)

inference = ed.KLqp({p: qp}, data={x: x_data})
inference.run(n_iter=500)

# CRITICISM
x_post = ed.copy(x, {p: qp})

sns.distplot(p.sample(1000).eval())
plt.show()

sns.distplot(qp.sample(1000).eval())
plt.show()

x_ = Bernoulli(tf.ones(10) * qp)

y_rep, y = ed.ppc(lambda xs, zs: tf.reduce_mean(tf.cast(xs[x_post],
                  tf.float32)), data={x_post: x_data})

y_rep, y = ed.ppc(lambda xs, zs: tf.reduce_mean(tf.cast(xs[x_],
                  tf.float32)), data={x_: x_data})

ppc_stat_hist_plot(y[0], y_rep,
                   stat_name=r'$T \equiv mean$', bins=10)
plt.show()
