import copy
import logging
import cv2
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FlareDataset(Dataset):
    def __init__(self, cfg, is_train):
        self.cfg = cfg
        self.is_train = is_train
        self.stride = cfg.STRIDE
        if is_train:
            self.input_dir = osp.join(cfg.DATA_DIR, cfg.TRAIN_INPUT_DIR)
            self.label_dir = osp.join(cfg.DATA_DIR, cfg.TRAIN_LABEL_DIR)

            assert os.path.isfile(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV)), "can't find train csv file"
            self.csv = pd.read_csv(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV))

            self.input_files = \
                [osp.join(self.cfg.DATA_DIR, cfg.TRAIN_INPUT_DIR, file_name) for file_name in self.csv['input_img']]
            self.label_files = \
                [osp.join(self.cfg.DATA_DIR, cfg.TRAIN_LABEL_DIR, file_name) for file_name in self.csv['label_img']]
            assert len(self.input_files) == len(self.label_files), "Number of Input and Label files don't match"
            self.db_path = cfg.DATA_DIR + '/train_db.pkl'

        else:
            self.input_dir = osp.join(cfg.DATA_DIR, cfg.TEST_INPUT_DIR)

            assert os.path.isfile(osp.join(cfg.DATA_DIR, cfg.TRAIN_CSV)), "can't find test csv file"
            self.csv = pd.read_csv(osp.join(cfg.DATA_DIR, cfg.TEST_CSV))

            self.input_files = \
                [osp.join(self.cfg.DATA_DIR, cfg.TRAIN_INPUT_DIR, file_name) for file_name in self.csv['input_img']]
            self.db_path = cfg.DATA_DIR + '/test_db.pkl'

        self.db = []

    def _get_db(self):
        db = []
        if osp.exists(self.db_path):
            logger.info('=> load db from pkl : {}'.format(self.db_path))
            pkl_file = open(self.db_path, 'rb')
            db = pickle.load(pkl_file)
            pkl_file.close()
            return db

        if self.is_train:
            # 이미지 Numpy 저장 폴더 생성
            input_path = self.input_dir + '/cut_img'
            label_path = self.label_dir + '/cut_img'
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(label_path, exist_ok=True)

            num = 0
            for input_file, label_file in tqdm(zip(self.input_files, self.label_files),
                                               desc="Dataset", total=len(self.input_files)):
                # 이미지 로드
                input_numpy = cv2.imread(input_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                label_numpy = cv2.imread(label_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                if input_numpy is None or label_numpy is None:
                    logger.error('=> fail to read {} and {}'.format(input_file, label_file))
                    raise ValueError('Fail to read {} and {}'.format(input_file, label_file))
                    continue

                # 입력 영상과 타겟 영상의 크기가 다른 경우 타겟 영상의 크기를 입력 영상 크기로 치환
                if not input_numpy.shape[0] == label_numpy.shape[0] and input_numpy.shape[1] == label_numpy.shape[1]:
                    logger.info('Warnning : label image size is modified to input image')
                    dim = (input_numpy.shape[1], input_numpy.shape[0])
                    label_numpy = cv2.resize(label_numpy, dim, interpolation=cv2.INTER_AREA)

                # 이미지 색상 타입 변경
                input_numpy = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB)
                label_numpy = cv2.cvtColor(label_numpy, cv2.COLOR_BGR2RGB)

                # 이미지 사이즈 정보 저장
                height, width = input_numpy.shape[0], input_numpy.shape[1]

                # Numpy 파일로 이미지 저장
                input_np_path = f'{input_path}/id_i{num}.npy'
                label_np_path = f'{label_path}/id_l{num}.npy'

                if not osp.exists(input_np_path):
                    np.save(input_np_path, input_numpy)
                if not osp.exists(label_np_path):
                    np.save(label_np_path, label_numpy)

                # 이미지 메타정보 저장
                meta = {
                    'id': num,
                    'image_file': input_file,
                    'label_file': label_file,
                    'image': input_np_path,
                    'label': label_np_path,
                    'stride': self.stride,
                    'height': height,
                    'width': width
                }

                db.append({
                    'id': num,
                    'image': input_np_path,
                    'label': label_np_path,
                    'meta': meta
                })

                num += 1

            # 저장
            pkl_file = open(self.db_path, 'wb')
            pickle.dump(db, pkl_file)
            pkl_file.close()
            logger.info('=> save db to pkl : {}'.format(self.db_path))

            return db
        else:
            # 이미지 Numpy 저장 폴더 생성
            input_path = self.input_dir + '/cut_img'
            os.makedirs(input_path, exist_ok=True)

            num = 0
            for input_file in tqdm(self.input_files, desc="Dataset", total=len(self.input_files)):
                # 이미지 로드
                input_numpy = cv2.imread(input_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                if input_numpy is None:
                    logger.error('=> fail to read {}'.format(input_file))
                    raise ValueError('Fail to read {}'.format(input_file))
                    continue

                # 이미지 색상 타입 변경
                input_numpy = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB)

                # 이미지 사이즈 정보 저장
                height, width = input_numpy.shape[0], input_numpy.shape[1]

                # Numpy 파일로 이미지 저장
                input_np_path = f'{input_path}/id_i{num}.npy'

                if not osp.exists(input_np_path):
                    np.save(input_np_path, input_numpy)

                # 이미지 메타정보 저장
                meta = {
                    'id': num,
                    'image_file': input_file,
                    'image': input_np_path,
                    'stride': self.stride,
                    'height': height,
                    'width': width
                }

                db.append({
                    'id': num,
                    'image': input_np_path,
                    'meta': meta
                })

                num += 1

            # 저장
            logger.info('=> save db to pkl : {}'.format(self.db_path))
            pkl_file = open(self.db_path, 'wb')
            pickle.dump(db, pkl_file)
            pkl_file.close()
            return db

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        if self.is_train:
            db_rec = copy.deepcopy(self.db[idx])

            input_numpy = np.load(db_rec['image'])
            label_numpy = np.load(db_rec['label'])

            if input_numpy is None or label_numpy is None:
                logger.error('=> fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
                raise ValueError('Fail to read {} and {}'.format(db_rec['image'], db_rec['label']))
                return None, None, None

            assert input_numpy.shape[0] == label_numpy.shape[0] and input_numpy.shape[1] == label_numpy.shape[1], \
                "Input and Label image have different size"

            return input_numpy, label_numpy, db_rec['meta']
        else:
            db_rec = copy.deepcopy(self.db[idx])

            input_numpy = np.load(db_rec['image'])

            if input_numpy is None:
                logger.error('=> fail to read {}'.format(db_rec['image']))
                raise ValueError('Fail to read {}'.format(db_rec['image']))
                return None, None

            return input_numpy, db_rec['meta']
