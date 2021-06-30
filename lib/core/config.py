from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

# key setting
config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.MODEL = 'AntiFlareNet'
config.GPUS = '0,1'
config.WORKERS = 24
config.PRINT_FREQ = 100
config.IMAGE_SIZE = 256

# Data directory setting
config.DATA_DIR = 'data'
config.TRAIN_CSV = 'train.csv'
config.TEST_CSV = 'test.csv'
config.TRAIN_INPUT_DIR = 'train_input_img'
config.TRAIN_LABEL_DIR = 'train_label_img'
config.TEST_INPUT_DIR = 'test_input_img'
config.TEST_LABEL_DIR = 'test_label_img'
config.TEST_OUTPUT_DIR = 'sample_submission'

# CUDNN
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# Dataset setting
config.AUGMENTATION_STRIDE = 256
config.DATA_CLASS = 'real'
config.RANDOM_FLIP = True
config.VALIDATION_RATIO = 0.2

# Training parameter
config.TRAIN = edict()
config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 200
config.TRAIN.RESUME = True
config.TRAIN.LR = 0.001
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR_FACTOR = 0.1

# Test parameter
config.TEST = edict()
config.TEST.BATCH_SIZE = 32
config.TEST.STATE = 'best'
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = False

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES = True


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def get_model_name(cfg):
    name = '{model}'.format(
        model=cfg.MODEL)
    full_name = '{height}x{width}_{name}'.format(
        height=cfg.IMAGE_SIZE,
        width=cfg.IMAGE_SIZE,
        name=name)

    return name, full_name