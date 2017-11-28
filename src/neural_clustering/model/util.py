import logging
import subprocess
import numpy as np


logger = logging.getLogger(__name__)


def build_toy_dataset(N):
    """
    Generate a toy dataset of size N
    """
    pi = np.array([0.4, 0.6])
    mus = [[5, 5], [-5, -5]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
    x = np.zeros((N, 2), dtype=np.float32)

    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    logger.info('Generated data from two bi-variate normals std=1 (diagonal), '
                'mu1 = [5, 5], mu2=[-5, -5]')
    return x


def get_commit_hash():
    out = subprocess.check_output('git show --oneline -s', shell=True)
    return out.decode('utf-8') .replace('\n', '')
