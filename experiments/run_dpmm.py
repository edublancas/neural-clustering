"""
Run a truncated DPMM experiment

Restart the Python session before running another experiment

There is a bug in Edward that *sometimes* throws an error when fitting, it
occurs more often with truncation levels > 5 more on that here:
https://discourse.edwardlib.org/t/variational-inference-for-dirichlet-process-mixtures/251/2
# noqa
"""
import os

import numpy as np

from neural_clustering import config
from neural_clustering.model import dpmm


TRUNCATION_LEVEL = 10
ITERATIONS = 50000
N_SAMPLES = 3

cfg = config.load('server_config.yaml')

path = os.path.join(cfg['root'], 'training.npy')

x_train = np.load(path)
x_train.shape

dpmm.fit(x_train, truncation_level=TRUNCATION_LEVEL, cfg=cfg,
         inference_params=dict(n_samples=N_SAMPLES, n_iter=ITERATIONS))
