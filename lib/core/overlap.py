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
    wind = wind[:, np.newaxis, np.newaxis]
    wind = wind * wind.transpose(1, 0, 2)
    return wind / wind.max()


class PatchOverlap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stride = cfg.STRIDE
        self.patch_size = cfg.PATCH_SIZE
        self.window = _window_2D(self.patch_size)
        self.id = None
        self.img_height = None
        self.img_width = None
        self.canvas = None

    def identify(self, id):
        return self.id == id

    def initialize(self, id, img_height, img_width):
        self.id = id
        self.img_height = img_height
        self.img_width = img_width
        self.canvas = np.zeros((self.cfg.TEST.MAX_HEIGHT, self.cfg.TEST.MAX_WIDTH, 3), dtype=np.float_)

    def append_patch(self, id, patch, top, left):
        assert id == self.id, 'Patch ID and image ID do not match'
        np_patch = patch.mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()
        temp = np_patch * self.window
        self.canvas[top:top+self.patch_size, left:left+self.patch_size, :] += temp

    def get_canvas(self):
        canvas = self.canvas[self.stride:self.img_height+self.stride, self.stride:self.img_width+self.stride, :]
        return canvas