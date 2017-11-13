import os

import numpy as np
from neural_noise import config

cfg = config.load('../config.yaml')

path = os.path.join(cfg['root'], 'training.npy')

train = np.load(path)
