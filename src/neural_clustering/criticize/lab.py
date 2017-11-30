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


def load_experiments_params(cfg):
    """Returns a list of dicts with experiments params
    """
    path_to_experiments = list_experiments(cfg)
    path_to_params = [os.path.join(e, 'params.yaml') for e
                      in path_to_experiments]

    return [load_yaml(p) for p in path_to_params]


def summarize_experiments(cfg):
    """Summarizes experiments in {root}/sessions/
    """
    params = load_experiments_params(cfg)
    all_keys = [list(p.keys()) for p in params]
    header = list(set([item for sublist in all_keys for item in sublist]))
    content = [[p.get(h) for h in header] for p in params]
    return Table(content, header)


def summarize_experiment(cfg, name):
    """Summarizes a single experiment
    """
    params = [p for p in load_experiments_params(cfg) if p['name'] == name]
    all_keys = [list(p.keys()) for p in params]
    header = list(set([item for sublist in all_keys for item in sublist]))
    content = [[p.get(h) for h in header] for p in params]
    return Table(content, header)
