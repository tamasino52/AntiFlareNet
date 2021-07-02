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

logger = logging.getLogger(__name__)


class RealFlareDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg
        self.input_dir = osp.join(cfg.DATA_DIR, cfg.TRAIN_INPUT_DIR)
        self.label_dir = osp.join(cfg.DATA_DIR, cfg.TRAIN_LABEL_DIR)
        self.img_size = cfg.IMAGE_SIZE
        self.stride = cfg.AUGMENTATION_STRIDE
        self.db_path = None

        if is_train:
            self.db_path = cfg.DATA_DIR + '/train_db.pkl'
        else:
            self.db_path = cfg.DATA_DIR + '/valid_db.pkl'

        self.db = self._get_db()
        self.db_size = len(self.db)

    def _get_db(self):
        db = []
        if osp.exists(self.db_path):
            logger.info('=> load db from pkl : {}'.format(self.db_path))
            pkl_file = open(self.db_path, 'rb')
            db = pickle.load(pkl_file)
            pkl_file.close()
            return db

        # 분할 이미지 저장 폴더 생성
        input_path = self.input_dir + '/cut_img' + str(self.img_size)
        label_path = self.label_dir + '/cut_img' + str(self.img_size)

        os.makedirs(input_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        num_file = len(self.input_files)
        num = 0
        for input_file, label_file in tqdm(zip(self.input_files, self.label_files), desc="Dataset", total=num_file):
            input_numpy = cv2.imread(input_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            label_numpy = cv2.imread(label_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if input_numpy is None or label_numpy is None:
                logger.error('=> fail to read {} and {}'.format(input_file, label_file))
                raise ValueError('Fail to read {} and {}'.format(input_file, label_file))
                continue

            input_numpy = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB)
            label_numpy = cv2.cvtColor(label_numpy, cv2.COLOR_BGR2RGB)

            # 영상을 Stride 단위로 쪼개서 npy 파일로 저장
            for top in range(0, input_numpy.shape[0], self.stride):
                for left in range(0, input_numpy.shape[1], self.stride):
                    input_piece_path = f'{input_path}/id{num}_t{top}_l{left}.npy'
                    if not osp.exists(input_piece_path):
                        piece = np.zeros([self.img_size, self.img_size, 3], np.uint8)
                        temp = input_numpy[top:top + self.img_size, left:left + self.img_size, :]
                        piece[:temp.shape[0], :temp.shape[1], :] = temp
                        np.save(input_piece_path, piece)

                    label_piece_path = f'{label_path}/id{num}_t{top}_l{left}.npy'
                    if not osp.exists(label_piece_path):
                        piece = np.zeros([self.img_size, self.img_size, 3], np.uint8)
                        temp = label_numpy[top:top + self.img_size, left:left + self.img_size, :]
                        piece[:temp.shape[0], :temp.shape[1], :] = temp
                        np.save(label_piece_path, piece)

                    # 메타데이터 DB 저장
                    meta = {
                        'id': num,
                        'image_file': input_file,
                        'label_file': label_file,
                        'is_real': True,
                        'image': input_piece_path,
                        'label': label_piece_path,
                        'location': (top, left),
                        'stride': self.stride
                    }

                    db.append({
                        'image': input_piece_path,
                        'label': label_piece_path,
                        'meta': meta
                    })

            num += 1

        # 저장
        logger.info('=> save db to pkl : {}'.format(self.db_path))
        pkl_file = open(self.db_path, 'wb')
        pickle.dump(db, pkl_file)
        pkl_file.close()
        return db

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return self.db_size