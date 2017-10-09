import os.path

import yaml
import numpy as np


def config(path_to_config):
    """Load yaml config file
    """
    def _expand(k, v):
        return os.path.expanduser(v) if k.endswith('root') else v

    with open(path_to_config) as f:
        config = yaml.load(f)

    # expand paths with ~
    config = {k: _expand(k, v) for k, v in config.items()}

    return config


def data(config, n_observations, n_channels):
    """Load MEAs readings
    """
    path_to_data = os.path.join(config['data_root'], config['data_filename'])
    data = np.fromfile(path_to_data, dtype='int16')
    return data.reshape(n_observations, n_channels)
