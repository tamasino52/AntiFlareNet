import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import pandas as pd
import glob

logger = logging.getLogger(__name__)


class FlareDataset(Dataset):

    def __init__(self, cfg, is_train):
        self.cfg = cfg

        assert(os.path.isfile(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV)), "can't find train csv file")
        self.csv = pd.read_csv(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV))

        self.input_files = osp.join(self.cfg.DATA_DIR, cfg.TRAIN_INPUT_DIR, self.csv['input_img'])
        self.label_files = osp.join(self.cfg.DATA_DIR, cfg.TRAIN_LABEL_DIR, self.csv['label_img'])

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

        input_numpy = glob(db_rec['image'])
        label_numpy = glob(db_rec['label'])

        if input_numpy is None or label_numpy is None:
            logger.error('=> fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
            raise ValueError('Fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
            return None, None, None

        assert(input_numpy.shape[0] == label_numpy.shape[0] and input_numpy.shape[1] == label_numpy.shape[1],
               "Input and Label image have different size")

        input_torch = torch.from_numpy(input_numpy)
        label_torch = torch.from_numpy(label_numpy)

        return input_torch, label_torch, db_rec['meta']