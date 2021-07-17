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


class FlarePatch2ImageDataset(IterableDataset):
    def __init__(self, cfg, is_train):
        self.flare_dataset = FlareDataset(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg

        self.patch_size = cfg.PATCH_SIZE
        self.stride = cfg.STRIDE

        self.db = self.flare_db._get_db()

    def transform(self, input):
        input = TF.to_tensor(input)
        return input

    def __iter__(self):
        db_size = len(self.db)
        for idx in range(db_size):

            input_np, _, meta = super().__getitem__(idx)
            input_torch = self.transform(input_np)
            patches = input_torch.data.unfold(0, 3, 3)\
                .unfold(1, self.patch_size, self.patch_size)\
                .unfold(2, self.patch_size, self.patch_size)



        # 이미지 변형
        input_torch, label_torch = self.transform(input_np, label_np)

        return input_torch, label_torch, meta

    def __len__(self):
        return self.db_size
