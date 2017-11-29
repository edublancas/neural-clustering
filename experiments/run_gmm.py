"""
Run a GMM experiment
"""
import os

import numpy as np

from neural_clustering import config
from neural_clustering.model import gmm

K = 20
ITERATIONS = 10000

cfg = config.load('server_config.yaml')

path = os.path.join(cfg['root'], 'training.npy')

x_train = np.load(path)
x_train.shape

gmm.fit(x_train, k=K, cfg=cfg, samples=ITERATIONS)
