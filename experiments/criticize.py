import os

from neural_clustering import config
from neural_clustering.criticize.restore import restore_session
from neural_clustering.criticize.criticize import find_cluster_assignments

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cfg = config.load('config.yaml')
session_name = '2017-11-15T01/52/00.310739-DPMM'

qmu, qbeta, x_train, params = restore_session(cfg, session_name)

clusters = find_cluster_assignments(x_train, qmu, params)
