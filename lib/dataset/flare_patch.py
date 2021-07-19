from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.FlareDataset import FlareDataset
import os.path as osp
import os
import cv2
import logging
import numpy as np
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
import torch
import math
from torch.utils.data import IterableDataset


logger = logging.getLogger(__name__)


class FlarePatchDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg

        self.patch_size = cfg.PATCH_SIZE
        self.stride = cfg.STRIDE
        self.num_recrop = 30

        self.db = self._get_db()
        self.db_size = len(self.db) * self.num_recrop

    def transform(self, input, label):
        aug = self.cfg.AUGMENTATION
        # TODO : Random Rotation

        # to PIL
        input = TF.to_pil_image(input)
        label = TF.to_pil_image(label)

        # Padding
        input = TF.pad(input, (self.stride, self.stride, self.stride, self.stride))
        label = TF.pad(label, (self.stride, self.stride, self.stride, self.stride))

        # Random crop
        crop = transforms.RandomResizedCrop(self.patch_size)
        params = crop.get_params(input, scale=(1./aug.RANDOM_SCALE, 1.0), ratio=(1.0, 1.0))
        input = TF.resized_crop(input, *params, self.patch_size)
        label = TF.resized_crop(label, *params, self.patch_size)

        # Random horizontal flipping
        if aug.RANDOM_HORIZONTAL_FLIP and random.random() > 0.5:
            input = TF.hflip(input)
            label = TF.hflip(label)

        # Random vertical flipping
        if aug.RANDOM_VERTICAL_FLIP and random.random() > 0.5:
            input = TF.vflip(input)
            label = TF.vflip(label)

        # Transform to tensor
        input = TF.pil_to_tensor(input) / 255.
        label = TF.pil_to_tensor(label) / 255.

        return input, label

    def _get_db(self):
        return super()._get_db()

    def __getitem__(self, idx):
        idx = int(idx / self.num_recrop)
        input_np, label_np, meta = super().__getitem__(idx)

        # 이미지 변형
        input_torch, label_torch = self.transform(input_np, label_np)

        return input_torch, label_torch, meta

    def __len__(self):
        return self.db_size
