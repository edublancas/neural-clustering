import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

import yass
from yass import preprocess
from yass import process

# configure logging module to get useful information
logging.basicConfig(level=logging.DEBUG)

# set yass configuration parameters
yass.set_config('config.yaml')

cfg = yass.read_config()

# run preprocessor
score, clear_index, spike_times = preprocess.run()

# run processor
spike_train, spt_left, templates = process.run(score, clear_index, spike_times)

plt.plot(templates[0, :, :])
plt.show()


raw = np.fromfile(os.path.join(cfg.root, cfg.filename), dtype='int16')
raw = raw.reshape(100000, cfg.nChan)

time, channel = spike_times[0][10, :]
time, channel





plot_spike_window(time, channel, dt=70)
plt.show()


plot_multi_channel_window(1000, range(6), dt=700)
plt.show()
