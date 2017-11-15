import os

import numpy as np

from neural_clustering import config
from neural_clustering.criticize.restore import restore_session
from neural_clustering.criticize.criticize import find_cluster_assignments

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cfg = config.load('config.yaml')
session_name = '15-Nov-2017@02-05-21-DPMM'

qmu, qbeta, x_train, params = restore_session(cfg, session_name)

clusters = find_cluster_assignments(x_train, qmu, params)

clusters

np.unique(clusters, return_counts=True)
