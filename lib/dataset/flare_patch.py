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

logger = logging.getLogger(__name__)


class FlarePatchDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg

        self.patch_size = cfg.PATCH_SIZE
        self.stride = cfg.STRIDE

        if is_train:
            self.db_path = cfg.DATA_DIR + '/train_patch_db.pkl'
        else:
            self.db_path = cfg.DATA_DIR + '/valid_patch_db.pkl'

        transform_list = []
        aug = cfg.AUGMENTATION

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

        # 랜덤 크롭
        if aug.RANDOM_RESIZED_CROP:
            transform_list.append(
                transforms.RandomResizedCrop(
                    size=self.patch_size,
                    scale=(1.0, 1./aug.RANDOM_SCALE),
                    ratio=(1., 1.)
                ))
        else:
            transform_list.append(
                transforms.RandomCrop(self.patch_size)
            )

        self.transform = transforms.Compose(transform_list)
        self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        return super()._get_db()

    def __getitem__(self, idx):
        input_np, label_np, meta = super().__getitem__(idx)

        # 이미지 변형
        input_torch = self.transform(input_np)
        label_torch = self.transform(label_np)

        return input_torch, label_torch, meta

    def __len__(self):
        return self.db_size
