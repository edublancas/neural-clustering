import os

import numpy as np
from neural_clustering import config

cfg = config.load('config.yaml')
cfg = config.load('server_config.yaml')

path = os.path.join(cfg['root'], 'training.npy')

x_train = np.load(path)
x_train.shape
