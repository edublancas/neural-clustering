import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

import yass
from yass import preprocess
from yass import process

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

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



def plot_multi_channel_window(time, channels, dt):
    f, axs = plt.subplots(len(channels), 1)

    for ax, ch in zip(axs, channels):
        plot_spike_window(time, ch, dt, ax=ax)


def plot_spike_window(time, channel, dt, ax=None):
    ax = ax if ax else plt
    spike = raw[time-dt-1:time+dt+1, channel]
    ax.plot(spike)
    ax.axvline(x=dt + 1)



plot_spike_window(time, channel, dt=70)
plt.show()


plot_multi_channel_window(time, range(6), dt=70)
plt.show()
