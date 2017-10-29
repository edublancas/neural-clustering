import logging
import argparse
import os

import pandas as pd
import numpy as np

import yass
from yass import preprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_waveforms():
    """
    Run yass pipeline to extract ans save scores, clear_indexes and
    spike times
    """
    parser = argparse.ArgumentParser(description='Extract waveforms')
    parser.add_argument('config', type=str,
                        help='Path to YASS configuration file')
    args = parser.parse_args()

    yass.set_config(args.config)
    score, clear_index, spike_times = preprocess.run()

    cfg = yass.read_config()

    def get_score(row):
        if row.clear:
            channel_scores = score[row.channel]
            return channel_scores[int(row.score_idx)].tolist()
        else:
            return np.nan

    def make_spike_times_df(spike_times, channel):
        df = pd.DataFrame(spike_times)

        df.columns = ['time', 'batch']
        df['channel'] = channel
        df.reset_index(inplace=True)
        df['clear'] = df['index'].apply(lambda idx: idx
                                        in clear_index[channel])

        df['score_idx'] = np.nan
        df.loc[df.clear, 'score_idx'] = np.arange(len(df[df.clear]))
        df['score'] = df.apply(get_score, axis=1)
        return df

    dfs = [make_spike_times_df(spt, ch)
           for ch, spt in enumerate(spike_times)]
    df = pd.concat(dfs)

    path_to_pickle = os.path.join(cfg.root, 'noise')
    path_to_pickle_file = os.path.join(path_to_pickle, 'waveforms.pickle')

    if not os.path.exists(path_to_pickle):
        os.makedirs(path_to_pickle)

    df.to_pickle(path_to_pickle_file)

    logger.info(f'Done. Waveforms saved at {path_to_pickle_file}')
