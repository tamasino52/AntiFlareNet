from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict
import sys
import os
import scipy.signal
import torch
import math
import torchvision
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from utils.vis import save_pred_batch_images


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
    def __init__(self, cfg):
        self.cfg = cfg
        self.stride = cfg.STRIDE
        self.kernel = cfg.PATCH_SIZE
        self.window = _window_2D(self.kernel, self.stride)
        self.batch_size = cfg.TEST.BATCH_SIZE
        self.unfold = nn.Unfold(self.kernel, stride=self.kernel-self.stride)
        self.transform = None
        self.re_transform = None
        self.patches = None
        self.num_patches = None
        self.shape = None
        self.batch_start = 0

    def __len__(self):
        return math.ceil(self.num_patches / self.batch_size)

    def get_batch(self, _input, _label, scale):
        h, w = _input.shape[2:]
        self.window = self.window.to(_input.device)

        self.transform = transforms.Compose([
            transforms.Resize((int(h * scale), int(w * scale))),
            transforms.Pad(self.kernel - self.stride)
        ])

        self.re_transform = transforms.Compose([
            transforms.CenterCrop((int(h * scale), int(w * scale))),
            transforms.Resize((h, w))
        ])

        input = self.transform(_input)
        label = self.transform(_label)
        b, c, h, w = input.shape
        self.shape = (h, w)

        # 이미지 분해
        input_patch = self.unfold(input)  # b, c, h, w -> b, c*k*k, l
        input_patch = input_patch.view(b, c, self.kernel, self.kernel, -1)  # n, c*k*k, l -> b, c, k, k, l
        input_patch = input_patch.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, self.kernel, self.kernel)  # b*l, c, k, k

        label_patch = self.unfold(label)  # b, c, h, w -> b, c*k*k, l
        label_patch = label_patch.view(b, c, self.kernel, self.kernel, -1)  # n, c*k*k, l -> b, c, k, k, l
        label_patch = label_patch.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, self.kernel, self.kernel)  # b*l, c, k, k

        self.patches = torch.zeros_like(input_patch)

        # 이미지 패치를 배치 단위로 쪼개서 예측
        self.num_patches = input_patch.shape[0]
        for batch in range(0, self.num_patches, self.batch_size):
            self.batch_start = batch
            yield input_patch[batch:batch + self.batch_size], label_patch[batch:batch + self.batch_size]

    def merge_batch(self, input_batch):
        self.patches[self.batch_start:self.batch_start + self.batch_size] = input_batch * self.window

    def get_image(self):
        self.patches = self.patches.view(1, self.num_patches, 3, self.kernel, self.kernel)  # b*l, c, k, k -> b, l, c, k, k
        self.patches = self.patches.permute(0, 2, 3, 4, 1).contiguous().view(1, 3 * self.kernel * self.kernel, self.num_patches)  # b, c*k*k, l

        # 패치 재조립
        image = F.fold(self.patches, self.shape, self.kernel, stride=self.kernel - self.stride)  # b, c*k*k, l -> b, c, k, k

        # 패딩 제거
        image = self.re_transform(image)

        return image

'''
    def predict(self, img, scale):
        b, c, h, w = img.shape
        assert b == 1, 'Image Batch size is not 1'

        self.window = self.window.to(img.device)

        transform = transforms.Compose([
            transforms.Resize((int(h * scale), int(w * scale))),
            transforms.Pad(self.kernel)
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
            input = patches[batch:batch + self.batch_size].clone()
            patches[batch:batch + self.batch_size] = self.model(patches[batch:batch + self.batch_size])
            # debug
            prefix = '{}_{:08}'.format(os.path.join('C:\\repo\\AntiFlareNet\\output\\AntiFlareNet\\unet_unet\\merge_img', 'train'), 0)

            save_pred_batch_images(input, patches[batch:batch + self.batch_size], input, prefix)

            # end
            patches[batch:batch + self.batch_size] = patches[batch:batch + self.batch_size] * self.window

        patches = patches.view(b, -1, c, self.kernel, self.kernel)  # b*l, c, k, k -> b, l, c, k, k
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(b, c * self.kernel * self.kernel, -1)  # b, c*k*k, l

        # 패치 재조립
        scaled_img = F.fold(patches, scaled_shape, self.kernel, stride=self.kernel - self.stride)  # b, c*k*k, l -> b, c, k, k

        # 패딩 제거
        scaled_img = re_transform(scaled_img)

        return scaled_img
'''