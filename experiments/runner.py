"""
Running experiments
"""
# %load_ext autoreload
# %autoreload 2

import logging

import edward as ed
import numpy as np

from neural_clustering.model import dpmm
from neural_clustering import config


logging.basicConfig(level=logging.INFO)


def build_toy_dataset(N):
    pi = np.array([0.4, 0.6])
    mus = [[5, 5], [-5, -5]]
    stds = [[1.0, 1.0], [1.0, 1.0]]
    x = np.zeros((N, 2), dtype=np.float32)

    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))

    return x


cfg = config.load('config.yaml')
x_train = build_toy_dataset(500)
truncation_level = 5

# http://edwardlib.org/api/ed/HMC
dpmm.fit(x_train, truncation_level, cfg,
         samples=10000,
         inference_alg=ed.HMC,
         inference_params=dict(step_size=0.25, n_steps=2))

# http://edwardlib.org/api/ed/SGLD
dpmm.fit(x_train, truncation_level, cfg,
         samples=10000,
         inference_alg=ed.SGLD,
         inference_params=dict(step_size=0.25))
