"""
Process pipeline: triage (optional) coreset (optional), masking, clustering,
getting templates and cleaning
"""
import os
import logging
import datetime

from yass.config import Config
from yass.process.triage import triage
from yass.process.coreset import coreset
from yass.process.mask import getmask
from yass.process.cluster import runSorter
from yass.process.clean import clean_output
from yass.process.templates import get_templates

import numpy as np
from neural_noise import config

CONFIG = Config.from_yaml('yass_config/server_49ch.yaml')
cfg = config.load('server_config.yaml')


# load data generated from yass
files = ['score', 'clear_index', 'spike_times', 'spike_train', 'spike_left', 'templates']

(score, clear_index,
 spike_times, spike_train,
 spike_left, templates) = [np.load(os.path.join(cfg['root'], 'yass/{}.npy'.format(f))) for f in  files]

startTime = datetime.datetime.now()

Time = {'t': 0, 'c': 0, 'm': 0, 's': 0, 'e': 0}

logger = logging.getLogger(__name__)

##########
# Triage #
##########

if CONFIG.doTriage:
    _b = datetime.datetime.now()
    logger.info("Triaging...")
    score, clear_index = triage(score, clear_index, CONFIG.nChan,
                                CONFIG.triageK, CONFIG.triagePercent,
                                CONFIG.neighChannels, CONFIG.doTriage)
    Time['t'] += (datetime.datetime.now()-_b).total_seconds()

# FIXME: pipeline will fail if coreset is deactivate as making is using
# coreset output as input

###########
# Coreset #
###########

if CONFIG.doCoreset:
    _b = datetime.datetime.now()
    logger.info("Coresettting...")
    group = coreset(score, CONFIG.nChan, CONFIG.coresetK, CONFIG.coresetTh)
    Time['c'] += (datetime.datetime.now()-_b).total_seconds()

###########
# Masking #
###########

_b = datetime.datetime.now()
logger.info("Masking...")
mask = getmask(score, group, CONFIG.maskTh, CONFIG.nFeat, CONFIG.nChan,
               CONFIG.doCoreset)
Time['m'] += (datetime.datetime.now()-_b).total_seconds()

##############
# Clustering #
##############

_b = datetime.datetime.now()
logger.info("Clustering...")
spike_train_clear = runSorter(score, mask, clear_index, group,
                              CONFIG.channelGroups, CONFIG.neighChannels,
                              CONFIG.nFeat, CONFIG)
Time['s'] += (datetime.datetime.now()-_b).total_seconds()

#################
# Clean output  #
#################
spike_train_clear, spike_times_left = clean_output(spike_train_clear,
                                             spike_times, clear_index,
                                             CONFIG.batch_size,
                                             CONFIG.BUFF)

#################
# Get templates #
#################

_b = datetime.datetime.now()
logger.info("Getting Templates...")
path_to_whiten = os.path.join(CONFIG.root, 'tmp/whiten.bin')
spike_train, templates = get_templates(spike_train_clear,
                                             CONFIG.batch_size,
                                             CONFIG.BUFF,
                                             CONFIG.nBatches,
                                             CONFIG.nChan,
                                             CONFIG.spikeSize,
                                             CONFIG.templatesMaxShift,
                                             CONFIG.scaleToSave,
                                             CONFIG.neighChannels,
                                             path_to_whiten,
                                             CONFIG.tMergeTh)
Time['e'] += (datetime.datetime.now()-_b).total_seconds()


currentTime = datetime.datetime.now()

logger.info("Mainprocess done in {0} seconds.".format(
    (currentTime-startTime).seconds))
logger.info("\ttriage:\t{0} seconds".format(Time['t']))
logger.info("\tcoreset:\t{0} seconds".format(Time['c']))
logger.info("\tmasking:\t{0} seconds".format(Time['m']))
logger.info("\tclustering:\t{0} seconds".format(Time['s']))
logger.info("\tmake templates:\t{0} seconds".format(Time['e']))
