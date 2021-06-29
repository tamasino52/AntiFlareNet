from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.FlareDataset import FlareDataset
import os.path as osp
import os


class RealFlareDataset(FlareDataset):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train)
        self.is_train = is_train
        self.cfg = cfg
        self.db = self._get_db()
        self.db_size = len(self.db)
        self.img_size = cfg.IMAGE_SIZE

    def _get_db(self):
        # TODO
        # 이미지 쪼개는 거 넣어야함
        db, val_db = [], []
        input_path = osp.join(self.cfg.DATA_DIR, self.cfg.TRAIN_INPUT_DIR)
        label_path = osp.join(self.cfg.DATA_DIR, self.cfg.TRAIN_LABEL_DIR)
        os.makedirs(f'{input_path}cut{self.img_size}', exist_ok=True)
        os.makedirs(f'{label_path}cut{self.img_size}', exist_ok=True)

        if self.is_train:
            input_data = self.input_files[:self.val_bound].to_numpy()
            label_data = self.label_files[:self.val_bound].to_numpy()
        else:
            input_data = self.input_files[self.val_bound:].to_numpy()
            label_data = self.label_files[self.val_bound:].to_numpy()

        for input_file, label_file in zip(input_data, label_data):
            db.append({
                'input_image': input_file,
                'label_image': label_file
            })
        return db

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return self.db_size

    def cut_img(self, img_path_list, save_path, stride):
        os.makedirs(f'{save_path}{img_size}', exist_ok=True)
        num = 0
        for path in tqdm(img_path_list):
            img = cv2.imread(path)
            for top in range(0, img.shape[0], stride):
                for left in range(0, img.shape[1], stride):
                    piece = np.zeros([img_size, img_size, 3], np.uint8)
                    temp = img[top:top+img_size, left:left+img_size, :]
                    piece[:temp.shape[0], :temp.shape[1], :] = temp
                    np.save(f'{save_path}{img_size}/{num}.npy', piece)
                    num+=1