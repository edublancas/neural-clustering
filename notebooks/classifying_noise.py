import os.path
import logging

import numpy as np
import matplotlib.pyplot as plt

import yass
from yass import preprocess
from yass import process
from yass.neuralnet import NeuralNetDetector
from yass.configuration import Configs

# configure logging module to get useful information
logging.basicConfig(level=logging.INFO)

# set yass configuration parameters
yass.set_config('config.yaml')

cfg = Configs('config.yaml')


raw = np.fromfile(os.path.join(cfg.root, 'noise.bin'), dtype='float64')

raw = raw.reshape((49998, 91, 7))


detector = NeuralNetDetector(cfg)
proj = detector.load_w_ae()
proj.shape

detector.get_spikes()