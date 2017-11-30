"""
Run a truncated DPMM experiment
"""
import os

import numpy as np

from neural_clustering import config
from neural_clustering.model import dpmm

# seems like there is a bug in edward that throws an error when truncation
# level > 5
TRUNCATION_LEVEL = 5
ITERATIONS = 50000
N_SAMPLES = 3

cfg = config.load('server_config.yaml')

path = os.path.join(cfg['root'], 'training.npy')

x_train = np.load(path)
x_train.shape

dpmm.fit(x_train, truncation_level=TRUNCATION_LEVEL, cfg=cfg,
         inference_params=dict(n_samples=N_SAMPLES, n_iter=ITERATIONS))
