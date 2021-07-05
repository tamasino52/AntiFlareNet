from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict
import sys
import scipy.signal
import torch
import torchvision
import torch.nn as nn


def _spline_window(window_size, power=2):
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _window_2D(window_size, power=2):
    wind = _spline_window(window_size, power)
    wind = wind[np.newaxis, :, np.newaxis]
    wind = wind * wind.transpose(0, 2, 1)
    return wind


class PatchOverlap:
    def __init__(self, cfg):
        self.stride = cfg.AUGMENTATION_STRIDE
        self.patch_size = 256
        self.window = _window_2D(self.patch_size)
        self.id = None
        self.img_height = None
        self.img_width = None
        self.canvas = None

    def initialize(self, id, img_height, img_width):
        self.id = id
        self.img_height = img_height
        self.img_width = img_width
        self.canvas = np.zeros((3, img_height + self.stride * 2 + self.patch_size, img_width + self.stride * 2  + self.patch_size), dtype=np.float_)

    def append_patch(self, id, patch, coord):
        top, left = coord[0], coord[1]
        assert id == self.id, 'Patch ID and image ID do not match'
        temp = np.matmul(patch, self.window)
        self.canvas[:, top:top+self.patch_size, left:left+self.patch_size] = temp

    def get_canvas(self):
        return self.canvas[:, self.stride:self.img_height+self.stride, self.stride:self.img_width+self.stride]