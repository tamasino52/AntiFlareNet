import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import pandas as pd
logger = logging.getLogger(__name__)
import math


class FlareDataset(Dataset):
    def __init__(self, cfg, is_train, only_predict=False):
        self.cfg = cfg
        self.is_train = is_train
        self.flip = cfg.RANDOM_FLIP
        self.only_predict = only_predict

        assert os.path.isfile(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV)), "can't find train csv file"
        self.csv = pd.read_csv(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV))

        self.input_files = \
            [osp.join(self.cfg.DATA_DIR, cfg.TRAIN_INPUT_DIR, file_name) for file_name in self.csv['input_img']]
        if not self.only_predict:
            self.label_files = self.input_files
        else:
            self.label_files = \
                [osp.join(self.cfg.DATA_DIR, cfg.TRAIN_LABEL_DIR, file_name) for file_name in self.csv['label_img']]

        assert len(self.input_files) == len(self.label_files), "Number of Input and Label files don't match"

        self.num_data = len(self.input_files)
        self.val_ratio = cfg.VALIDATION_RATIO
        self.val_bound = int(cfg.VALIDATION_RATIO * self.num_data)

        if is_train:
            self.input_files = self.input_files[self.val_bound:]
            self.label_files = self.label_files[self.val_bound:]
        else:
            self.input_files = self.input_files[:self.val_bound]
            self.label_files = self.label_files[:self.val_bound]

        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        full_input_numpy = np.load(db_rec['image'])
        full_label_numpy = np.load(db_rec['label'])

        top, left = db_rec['location']
        input_numpy = np.zeros([self.img_size, self.img_size, 3], np.uint8)
        temp = full_input_numpy[top:top + self.img_size, left:left + self.img_size, :]
        input_numpy[:temp.shape[0], :temp.shape[1], :] = temp

        label_numpy = np.zeros([self.img_size, self.img_size, 3], np.uint8)
        temp = full_label_numpy[top:top + self.img_size, left:left + self.img_size, :]
        label_numpy[:temp.shape[0], :temp.shape[1], :] = temp

        if input_numpy is None or label_numpy is None:
            logger.error('=> fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
            raise ValueError('Fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
            return None, None, None

        assert input_numpy.shape[0] == label_numpy.shape[0] and input_numpy.shape[1] == label_numpy.shape[1], \
            "Input and Label image have different size"

        # 랜덤하게 좌우 반전
        if self.is_train and self.flip:
            if np.random.randint(2) == 0:
                input_numpy = np.fliplr(input_numpy)
                label_numpy = np.fliplr(label_numpy)

        # 노멀라이즈
        input_numpy = input_numpy.astype(np.float32)/255
        label_numpy = label_numpy.astype(np.float32)/255

        input_torch = torch.from_numpy(input_numpy).permute(2, 0, 1).contiguous()
        label_torch = torch.from_numpy(label_numpy).permute(2, 0, 1).contiguous()

        return input_torch, label_torch, db_rec['meta']
