from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import os
import cv2
import logging
import numpy as np
from tqdm import tqdm
import pickle
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import random
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class FlareTrainDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg.DATA_DIR, 'train')

        inp_files = sorted(os.listdir(os.path.join(self.data_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(self.data_dir, 'target')))

        self.inp_filenames = [os.path.join(self.data_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(self.data_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.sizex = len(self.inp_filenames)  # get the size of target
        self.ps = cfg.PATCH_SIZE

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        w, h = tar_img.size
        inp_img = TF.resize(inp_img, [int(h/2), int(w/2)])
        tar_img = TF.resize(tar_img, [int(h/2), int(w/2)])


        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename

    def __len__(self):
        return self.sizex


class FlareTestDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg.DATA_DIR, 'test')
        inp_files = sorted(os.listdir(os.path.join(self.data_dir, 'input')))
        self.inp_filenames = [os.path.join(self.data_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.inp_size = len(self.inp_filenames)  # get the size of target

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)

        return inp, filename

    def __len__(self):
        return self.inp_size


class FlareImageDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg.DATA_DIR, 'train')
        inp_files = sorted(os.listdir(os.path.join(self.data_dir, 'input')))
        self.inp_filenames = [os.path.join(self.data_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.inp_size = len(self.inp_filenames)  # get the size of target

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)

        return inp, filename

    def __len__(self):
        return self.inp_size


class FlareMultiScaleTrainDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = [os.path.join(cfg.OUTPUT_DIR, 's' + str(scale)) for scale in [1, 2, 4, 8]]

        inp_files = sorted(os.listdir(os.path.join(cfg.DATA_DIR, 'train', 'input')))
        inr_files = [sorted(os.listdir(self.data_dir[s])) for s in range(4)]
        tar_files = sorted(os.listdir(os.path.join(cfg.DATA_DIR, 'train', 'target')))

        self.inp_filenames = [os.path.join(cfg.DATA_DIR, 'train', 'input', x) for x in inp_files if is_image_file(x)]
        self.inr_filenames = [[os.path.join(self.data_dir[s], x) for x in inr_files[s] if is_image_file(x)] for s in range(4)]
        self.tar_filenames = [os.path.join(cfg.DATA_DIR, 'train', 'target', x) for x in tar_files if is_image_file(x)]
        self.tar_size = len(self.tar_filenames)  # get the size of target

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        path_inr = [self.inr_filenames[s][index] for s in range(4)]
        path_tar = self.tar_filenames[index]

        vflip = random.random() > 0.5
        hflip = random.random() > 0.5

        inp = Image.open(path_inp)
        inp = TF.resize(inp, [512, 512])
        if vflip:
            inp = TF.vflip(inp)
        if hflip:
            inp = TF.hflip(inp)
        inp = TF.to_tensor(inp)

        inr = []
        for path in path_inr:
            inr_img = Image.open(path)
            inr_img = TF.resize(inr_img, [512, 512])
            if vflip:
                inr_img = TF.vflip(inr_img)
            if hflip:
                inr_img = TF.hflip(inr_img)
            inr.append(TF.to_tensor(inr_img))

        tar = Image.open(path_tar)
        tar = TF.resize(tar, [512, 512])

        if vflip:
            tar = TF.vflip(tar)
        if hflip:
            tar = TF.hflip(tar)

        tar = TF.to_tensor(tar)

        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]

        return inp, inr, tar, filename

    def __len__(self):
        return self.tar_size