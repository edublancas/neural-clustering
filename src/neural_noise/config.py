import os.path
import yaml


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
