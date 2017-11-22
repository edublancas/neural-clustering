import os
from glob import glob

import yaml

from ..explore.table import Table


def load_yaml(path):
    with open(path) as f:
        data = yaml.load(f)
    return data


def list_experiments(cfg):
    """List experiments in {root}/sessions/
    """
    path_to_sessions = os.path.join(cfg['root'], 'sessions')
    return glob(os.path.join(path_to_sessions, '*/'))


def summarize_experiments(cfg):
    """Summarizes experiments in {root}/sessions/
    """
    path_to_experiments = list_experiments(cfg)
    path_to_params = [os.path.join(e, 'params.yaml') for e
                      in path_to_experiments]

    params = [load_yaml(p) for p in path_to_params]
    header = params[0].keys()
    content = [p.values() for p in params]

    return Table(content, header)
