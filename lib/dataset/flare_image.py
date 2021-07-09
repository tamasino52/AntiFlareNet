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

logger = logging.getLogger(__name__)


class FlareImageDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg
        self.multi_scale = cfg.MULTI_SCALE

        transform_list = []
        aug = cfg.AUGMENTATION
        # TODO : 랜덤 변환의 경우 시드값 일치 X 다른 방법 필요
        # 정규 변환
        transform_list.append(transforms.ToTensor())
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ))

        if is_train:
            # 상화 좌우 반전
            if aug.RANDOM_HORIZONTAL_FLIP:
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug.RANDOM_VERTICAL_FLIP:
                transform_list.append(transforms.RandomVerticalFlip())
            # 회전 변환
            transform_list.append(transforms.RandomRotation(degrees=aug.RANDOM_ROTATION))

        self.transform = transforms.Compose(transform_list)
        self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        return super()._get_db()

    def __getitem__(self, idx):
        if self.is_train:
            input_np, label_np, meta = super().__getitem__(idx)

            # 이미지 변형
            input_torch = self.transform(input_np)
            label_torch = self.transform(label_np)

            return input_torch, label_torch, meta
        else:
            input_np, meta = super().__getitem__(idx)

            # 이미지 변형
            input_torch = self.transform(input_np)

            return input_torch, meta

    def __len__(self):
        return self.db_size
