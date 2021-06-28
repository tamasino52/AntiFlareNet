import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

logger = logging.getLogger(__name__)


class FlareDataset(Dataset):

    def __init__(self, cfg, is_train):
        self.cfg = cfg

        if is_train:
            assert(os.path.isfile(cfg.train_csv), "can't find train csv file")
            self.csv = pd.read_csv(cfg.train_csv)

            self.input_files = cfg.train_input_dir + self.csv['input_img']
            self.label_files = cfg.train_label_dir + self.csv['label_img']

            assert(len(self.input_files) == len(self.label_files), "Number of Input and Label files don't match")

            self.num_data = len(self.input_files)
            self.val_ratio = cfg.val_ratio
            self.val_bound = int(cfg.val_ratio * self.num_train)
        else:
            assert(os.path.isfile(cfg.test_csv), "can't find test csv file")
            self.csv = pd.read_csv(cfg.test_csv)
            self.input_files = cfg.test_input_dir + self.csv['input_img']
            self.num_data = len(self.input_files)
            self.val_bound = self.num_data

        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            # logger.error('=> fail to read {}'.format(image_file))
            # raise ValueError('Fail to read {}'.format(image_file))
            return None, None, None, None, None, None

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_2d']
        joints_vis = db_rec['joints_2d_vis']
        joints_3d = db_rec['joints_3d']
        joints_3d_vis = db_rec['joints_3d_vis']

        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width, _ = data_numpy.shape
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for n in range(nposes):
            for i in range(len(joints[0])):
                if joints_vis[n][i, 0] > 0.0:
                    joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        if 'pred_pose2d' in db_rec and db_rec['pred_pose2d'] != None:
            # For convenience, we use predicted poses and corresponding values at the original heatmaps
            # to generate 2d heatmaps for Campus and Shelf dataset.
            # You can also use other 2d backbone trained on COCO to generate 2d heatmaps directly.
            pred_pose2d = db_rec['pred_pose2d']
            for n in range(len(pred_pose2d)):
                for i in range(len(pred_pose2d[n])):
                    pred_pose2d[n][i, 0:2] = affine_transform(pred_pose2d[n][i, 0:2], trans)

            input_heatmap = self.generate_input_heatmap(pred_pose2d)
            input_heatmap = torch.from_numpy(input_heatmap)
        else:
            input_heatmap = torch.zeros(self.cfg.NETWORK.NUM_JOINTS, self.heatmap_size[1], self.heatmap_size[0])

        target_heatmap, target_weight = self.generate_target_heatmap(
            joints, joints_vis)
        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, self.num_joints, 2))
        joints_vis_u = np.zeros((self.maximum_person, self.num_joints, 2))
        for i in range(nposes):
            joints_u[i] = joints[i]
            joints_vis_u[i] = joints_vis[i]

        joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]

        target_3d = self.generate_3d_target(joints_3d)
        target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': db_rec['camera']
        }
        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap
