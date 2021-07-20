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
import copy
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


# TODO
class FlareScaledImageDataset:
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg
        self.multi_scale = cfg.MULTI_SCALE
        self.patch_size = cfg.PATCH_SIZE
        self.stride = cfg.STRIDE
        self.db_path = cfg.DATA_DIR + '/scaled_db.pkl'

        self.db = self._get_db()
        self.db_size = len(self.db)

    def transform(self, input, label=None):
        input = TF.to_tensor(input)
        input = TF.normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if label is not None:
            label = TF.to_tensor(label)
            label = TF.normalize(label, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            return input, label
        else:
            return input

    def _get_db(self):
        if osp.exists(self.db_path):
            logger.info('=> load db from pkl : {}'.format(self.db_path))
            pkl_file = open(self.db_path, 'rb')
            db = pickle.load(pkl_file)
            pkl_file.close()
            return db
        else:
            raise FileNotFoundError

    def __getitem__(self, idx):
        if self.is_train:
            db_rec = copy.deepcopy(self.db[idx])

            input_numpy = np.load(db_rec['image'])
            label_numpy = np.load(db_rec['label'])
            scaled_numpy = [np.load(item) for item in db_rec['scaled_images']]

            if input_numpy is None or label_numpy is None:
                logger.error('=> fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
                raise ValueError('Fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
            for item in scaled_numpy:
                if item is None:
                    logger.error('=> fail to read {}'.format(item))
                    raise ValueError('Fail to read {}'.format(item))

            input_torch = self.transform(input_numpy)
            label_torch = self.transform(label_numpy)
            scaled_torch = [self.transform(item) for item in scaled_numpy]

            return input_torch, scaled_torch, label_torch, db_rec['meta']
        else:
            db_rec = copy.deepcopy(self.db[idx])

            input_numpy = np.load(db_rec['image'])
            scaled_numpy = [np.load(item) for item in db_rec['scaled_images']]

            if input_numpy is None:
                logger.error('=> fail to read {}'.format(db_rec['image']))
                raise ValueError('Fail to read {}'.format(db_rec['image']))
            for item in scaled_numpy:
                if item is None:
                    logger.error('=> fail to read {}'.format(item))
                    raise ValueError('Fail to read {}'.format(item))

            input_torch = self.transform(input_numpy)
            scaled_torch = [self.transform(item) for item in scaled_numpy]

            return  input_torch, scaled_torch, db_rec['meta']

    def __len__(self):
        return self.db_size
