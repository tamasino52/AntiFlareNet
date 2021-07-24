import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import random_split
import torch.utils.data.distributed
import torch.utils.data.dataset
import torchvision
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json
import _init_paths
import dataset
from tqdm import tqdm
import numpy as np
import pickle
import time
import math
import random
import cv2

from models.MPRNet import MPRNet
from utils.utils import load_model_state
from core.config import config as cfg
from dataset.flare_image import FlareImageDataset
from utils.vis import save_torch_image
from pathlib import Path
import torch.nn.functional as F
from skimage import img_as_ubyte
from core.scheduler import GradualWarmupScheduler
from core.metrics import torchPSNR, numpyPSNR
from utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
import segmentation_models_pytorch as seg

random.seed(1313)
np.random.seed(1313)
torch.manual_seed(1313)
torch.cuda.manual_seed_all(1313)


def get_optimizer(model):
    lr = cfg.TRAIN.LR

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.END_EPOCH - cfg.TRAIN.WARMUP_EPOCH,
                                                            eta_min=cfg.TRAIN.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCH,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
    return model, optimizer, scheduler


def main():
    # 출력 경로 설정
    this_dir = Path(os.path.dirname(__file__))
    output_dir = (this_dir / '..' / cfg.OUTPUT_DIR).resolve()

    # 모델 생성
    print('=> Constructing models ..')
    model = MPRNet()
    model = load_model_state(model, output_dir, 'model_best.pth.tar')

    model.cuda()
    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    # 데이터셋 생성
    print('=> Loading Image data ..')
    dataset = FlareImageDataset(cfg)

    # 데이터로더 적재
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        drop_last=False,
        pin_memory=True)

    print('=> Making Multi Scale Prediction')
    ps = cfg.PATCH_SIZE
    st = cfg.STRIDE
    # Test Loop
    with torch.no_grad():
        for scale in [1, 2, 4, 8]:
            for i, (input, file) in enumerate(tqdm(train_loader, desc='scale : ' + str(scale)), 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input = TF.resize(input, [int(input.shape[2] / scale), int(input.shape[3] / scale)])

                # Padding in case images are not multiples of 8
                factor = ps - st
                h, w = input.shape[2], input.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                padh += factor if H-h == factor else 0
                padw += factor if W-w == factor else 0
                input = F.pad(input, (0, padw, 0, padh), 'reflect')

                output = torch.zeros_like(input)
                overlap = torch.zeros_like(input)

                input = input.cuda()

                for top in range(0, H, ps-st):
                    for left in range(0, W, ps-st):
                        pred = model(input[:, :, top:top+ps, left:left+ps])
                        pred = torch.clamp(pred[0], 0, 1)
                        overlap[:, :, top:top+ps, left:left+ps] += 1.
                        output[:, :, top:top+ps, left:left+ps] += pred.cpu()

                pred = output / overlap
                # Unpad images to original dimensions
                pred = pred[:, :, :h, :w]

                pred = TF.resize(pred, [int(pred.shape[2] * scale), int(pred.shape[3] * scale)])

                pred = pred.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(pred)):
                    pred_img = img_as_ubyte(pred[batch])
                    data_dir = os.path.join(output_dir, 's' + str(scale))
                    filepath = (os.path.join(data_dir, file[batch] + '.png'))
                    cv2.imwrite(filepath, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
