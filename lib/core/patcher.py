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
import torch.nn.functional as F
from torchvision.transforms import transforms


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


def _window_2D(window_size, padding, power=2):
    wind_1d = _spline_window(2 * padding, power)
    wind_1d = wind_1d / wind_1d.max()
    wind = np.ones(window_size, dtype=np.float_)
    border = int(len(wind_1d)/2)
    wind[:border] = wind_1d[:border]
    wind[-border:] = wind_1d[border:]
    wind = wind[np.newaxis, :, np.newaxis]
    wind = wind * wind.transpose(0, 2, 1)
    return torch.from_numpy(wind / wind.max())


class Patcher:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.stride = cfg.STRIDE
        self.kernel = cfg.PATCH_SIZE
        self.window = _window_2D(self.kernel, self.stride)
        self.model = model
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.unfold = nn.Unfold(self.kernel, stride=self.kernel-self.stride)

    def predict(self, img, scale):
        b, c, h, w = img.shape
        assert b == 1, 'Image Batch size is not 1'

        self.window = self.window.to(img.device)

        transform = transforms.Compose([
            transforms.Resize((int(h * scale), int(w * scale))),
            transforms.Pad(self.stride)
        ])

        re_transform = transforms.Compose([
            transforms.CenterCrop((int(h * scale), int(w * scale))),
            transforms.Resize((h, w))
        ])

        scaled_img = transform(img)
        scaled_shape = scaled_img.shape[2:]
        # 이미지 분해
        patches = self.unfold(scaled_img)  # b, c, h, w -> b, c*k*k, l
        patches = patches.view(b, c, self.kernel, self.kernel, -1)  # n, c*k*k, l -> b, c, k, k, l
        patches = patches.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, self.kernel, self.kernel)  # b*l, c, k, k

        # 이미지 패치를 배치 단위로 쪼개서 예측
        for batch in range(0, patches.shape[0], self.batch_size):
            patches[batch:batch + self.batch_size] = self.model(patches[batch:batch + self.batch_size])
            patches[batch:batch + self.batch_size] = patches[batch:batch + self.batch_size] * self.window

        patches = patches.view(b, -1, c, self.kernel, self.kernel)  # b*l, c, k, k -> b, l, c, k, k
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(b, c * self.kernel * self.kernel, -1)  # b, c*k*k, l

        # 패치 재조립
        scaled_img = F.fold(patches, scaled_shape, self.kernel, stride=self.kernel - self.stride)  # b, c*k*k, l -> b, c, k, k

        # 패딩 제거
        scaled_img = re_transform(scaled_img)

        return scaled_img