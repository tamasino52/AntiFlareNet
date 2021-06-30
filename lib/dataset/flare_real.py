from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.FlareDataset import FlareDataset
import os.path as osp
import os
import cv2
import logging

logger = logging.getLogger(__name__)


class RealFlareDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg
        self.db = self._get_db()
        self.db_size = len(self.db)
        self.img_size = cfg.IMAGE_SIZE
        self.stride = cfg.AUGMENTATION_STRIDE

    def _get_db(self):
        db = []

        # 분할 이미지 저장 폴더 생성
        input_path = osp.join(self.cfg.DATA_DIR, self.cfg.TRAIN_INPUT_DIR, 'cut_img' + str(self.img_size))
        label_path = osp.join(self.cfg.DATA_DIR, self.cfg.TRAIN_LABEL_DIR, 'cut_img' + str(self.img_size))

        os.makedirs(input_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        num = 0
        for input_file, label_file in zip(self.input_files, self.label_files):
            input_numpy = cv2.imread(input_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            label_numpy = cv2.imread(label_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if input_numpy is None or label_numpy is None:
                logger.error('=> fail to read {} and {}'.format(input_file, label_file))
                raise ValueError('Fail to read {} and {}'.format(input_file, label_file))
                continue

            input_numpy = cv2.cvtColor(input_numpy, cv2.COLOR_BGR2RGB)
            label_numpy = cv2.cvtColor(label_numpy, cv2.COLOR_BGR2RGB)

            # 입력 영상과 타겟 영상의 크기가 다른 경우 타겟 영상의 크기를 입력 영상 크기로 치환
            if input_numpy.shape[0] == label_numpy.shape[0] and input_numpy.shape[1] == label_numpy.shape[1]:
                label_numpy = cv2.resize(label_numpy, input_numpy.shape[:2])

            # 영상을 Stride 단위로 쪼개서 npy 파일로 저장
            for top in range(0, input_numpy.shape[0], self.stride):
                for left in range(0, input_numpy.shape[1], self.stride):
                    piece = np.zeros([self.img_size, self.img_size, 3], np.uint8)
                    temp = input_numpy[top:top + img_size, left:left + img_size, :]
                    piece[:temp.shape[0], :temp.shape[1], :] = temp
                    input_piece_path = f'{input_path}/id{num}_t{top}_l{left}.npy'
                    np.save(input_piece_path, piece)

                    piece = np.zeros([self.img_size, self.img_size, 3], np.uint8)
                    temp = label_numpy[top:top + img_size, left:left + img_size, :]
                    piece[:temp.shape[0], :temp.shape[1], :] = temp
                    label_piece_path = f'{label_path}/id{num}_t{top}_l{left}.npy'
                    np.save(label_piece_path, piece)

                    meta = {
                        'image_file': input_file,
                        'label_file': label_file,
                        'is_real': True,
                        'image': input_piece_path,
                        'label': label_piece_path,
                        'location': (top, left),
                        'stride': self.stride,
                        'id': num
                    }

                    db.append({
                        'image': input_piece_path,
                        'label': label_piece_path,
                        'meta': meta
                    })
            num += 1
        return db

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return self.db_size