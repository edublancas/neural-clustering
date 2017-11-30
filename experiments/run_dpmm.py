"""
Run a truncated DPMM experiment
"""
import os

import numpy as np

from neural_clustering import config
from neural_clustering.model import dpmm

TRUNCATION_LEVEL = 20
ITERATIONS = 50000

cfg = config.load('server_config.yaml')

path = os.path.join(cfg['root'], 'training.npy')

x_train = np.load(path)
x_train.shape

dpmm.fit(x_train, truncation_level=TRUNCATION_LEVEL, cfg=cfg,
         inference_params=dict(n_iter=ITERATIONS))
