import logging

import matplotlib.pyplot as plt

import yass
from yass import preprocess
from yass import process

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config.yaml')

# run preprocessor
score, clr_idx, spt = preprocess.run()

# run processor
spike_train, spt_left, templates = process.run(score, clr_idx, spt)

plt.plot(templates[0, :, :])
plt.show()
