import logging
import argparse
import os

import yass
from yass import preprocess, process
import numpy as np

from . import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_yass():
    """
    Run yass preprocess and process steps, saves all results to the
    projects root path under yass folder
    """
    parser = argparse.ArgumentParser(description='Run YASS preprocess and'
                                     ' process')
    parser.add_argument('yass_config', type=str,
                        help='Path to YASS configuration file')
    parser.add_argument('config', type=str,
                        help='Path to configuration file')

    args = parser.parse_args()

    yass.set_config(args.yass_config)

    score, clear_index, spike_times = preprocess.run()

    spike_train, spike_left, templates = process.run(score, clear_index,
                                                     spike_times)

    cfg = config.load(args.config)

    path_to_yass_output = os.path.join(cfg.root, 'yass')

    if not os.path.exists(path_to_yass_output):
        os.makedirs(path_to_yass_output)

    # save all results from yass
    np.save(os.path.join(path_to_yass_output, 'score', score))
    np.save(os.path.join(path_to_yass_output, 'clear_index', clear_index))
    np.save(os.path.join(path_to_yass_output, 'spike_times', spike_times))
    np.save(os.path.join(path_to_yass_output, 'spike_train', spike_train))
    np.save(os.path.join(path_to_yass_output, 'spike_left', spike_left))
    np.save(os.path.join(path_to_yass_output, 'templates', templates))

    logger.info(f'Done. Files saved at {path_to_yass_output}')
