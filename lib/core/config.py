from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

# key setting
config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.BACKBONE_MODEL = 'resnet'
config.MODEL = 'antiflarenet'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100
config.IMAGE_SIZE = 256

# Data directory setting
config.DATA_DIR = 'data'
config.TRAIN_CSV = 'train.csv'
config.TEST_CSV = 'test.csv'
config.TRAIN_INPUT_DIR = 'train_input_img'
config.TRAIN_LABEL_DIR = 'train_label_img'
config.TEST_INPUT_DIR = 'test_input_img'
config.TEST_OUTPUT_DIR = 'sample_submission'

# Dataset setting
config.AUGMENTATION_STRIDE = 128

# Training parameter
config.DATA_CLASS = 'real'
config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 200
config.TRAIN.RESUME = True

config.TEST.BATCH_SIZE = 32

# TODO