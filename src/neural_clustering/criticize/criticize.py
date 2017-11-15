import tensorflow as tf
from edward.models import Normal


def find_cluster_assigments(x_train, qmu, params):
    """Find cluster assignments
    """
    N, D = x_train.shape
    K = params['truncation_level']

    # http://edwardlib.org/api/ed/MonteCarlo
    total = 10000
    burn_in = 4000
    SC = total - burn_in

    mu_sample = qmu.sample(total)[burn_in:, :, :]

    mu_sample

    x_post = Normal(tf.ones([N, 1, 1, 1]) * mu_sample,
                    tf.ones([N, 1, 1, 1]) * 1.0)

    x_broadcasted = tf.tile(tf.reshape(x_train, [N, 1, 1, D]), [1, SC, K, 1])
    # is it ok to convert like this?
    x_broadcasted = tf.to_float(x_broadcasted)

    log_liks = x_post.log_prob(x_broadcasted)
    log_liks = tf.reduce_sum(log_liks, 3)
    log_liks = tf.reduce_mean(log_liks, 1)

    clusters = tf.argmax(log_liks, 1).eval()

    # TODO: relevel clusters
    return clusters
