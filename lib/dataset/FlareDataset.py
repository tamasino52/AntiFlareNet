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


class FlareDataset(Dataset):

    def __init__(self, cfg, is_train):
        self.cfg = cfg

        assert(os.path.isfile(osp.join(cfg.data_dir, cfg.train_csv)), "can't find train csv file")
        self.csv = pd.read_csv(osp.join(cfg.data_dir, cfg.train_csv))

        self.input_files = osp.join(cfg.data_dir, cfg.train_input_dir, self.csv['input_img'])
        self.label_files = osp.join(cfg.data_dir, cfg.train_label_dir, self.csv['label_img'])

        assert(len(self.input_files) == len(self.label_files), "Number of Input and Label files don't match")

        self.num_data = len(self.input_files)
        self.val_ratio = cfg.val_ratio
        self.val_bound = int(cfg.val_ratio * self.num_data)

        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        # TODO
        # 이미지 쪼개는 거 넣어야함
        db_rec = copy.deepcopy(self.db[idx])

        input_image_file = db_rec['input_image']
        label_image_file = db_rec['label_image']

        input_numpy = cv2.imread(
            input_image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        label_numpy = cv2.imread(
            label_image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if input_numpy is None or label_numpy is None:
            logger.error('=> fail to read {} and {}'.format(input_image_file, label_image_file))
            raise ValueError('Fail to read {} and {}'.format(input_image_file, label_image_file))
            return None, None, None

        input_numpy = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB)
        label_numpy = cv2.cvtColor(label_numpy, cv2.COLOR_BGR2RGB)

        assert(input_numpy.shape[0] == label_numpy.shape[0] and input_numpy.shape[1] == label_numpy.shape[1],
               "Input and Label image have different size")

        meta = {
            'image_file': input_image_file,
            'label_file': label_image_file,
            'is_real': True,
            'input_image': input_numpy,
            'label_image': label_numpy
        }

        input_torch = torch.from_numpy(input_numpy)
        label_torch = torch.from_numpy(label_numpy)

        return input_torch, label_torch, meta