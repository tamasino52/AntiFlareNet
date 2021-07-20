from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.FlareDataset import FlareDataset
import os.path as osp
import os
import cv2
import logging
import numpy as np
from tqdm import tqdm
import pickle
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


class FlareImageDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg
        self.multi_scale = cfg.MULTI_SCALE
        self.patch_size = cfg.PATCH_SIZE
        self.stride = cfg.STRIDE

        self.db = self._get_db()
        self.db_size = len(self.db)

    def transform(self, input, label=None):
        input = TF.to_tensor(input)
        if label is not None:
            label = TF.to_tensor(label)
            return input, label
        else:
            return input

    def _get_db(self):
        return super()._get_db()

    def __getitem__(self, idx):
        if self.is_train:
            input_np, label_np, meta = super().__getitem__(idx)

            # 이미지 변형
            input_torch, label_torch = self.transform(input_np, label_np)

            return input_torch, label_torch, meta
        else:
            input_np, meta = super().__getitem__(idx)

            # 이미지 변형
            input_torch = self.transform(input_np)

            return input_torch, meta

    def __len__(self):
        return self.db_size
