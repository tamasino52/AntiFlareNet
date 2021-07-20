from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import random_split
import torch.utils.data.distributed
import torch.utils.data.dataset

from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import time

import logging
import json
import torchvision

import _init_paths
import dataset

from models.AntiFlareNet import AntiFlareNet
from utils.utils import save_checkpoint, load_checkpoint, create_logger
from core.config import config, update_config
from core.function import train, validate
from dataset.flare_image import FlareImageDataset
from dataset.flare_patch import FlarePatchDataset
from models.patch_flare_net import PatchFlareNet
from utils.vis import save_pred_batch_images, save_numpy_image


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.cfg is not None:
        update_config(args.cfg)

    return args


def get_optimizer(model):
    lr = config.TRAIN.LR
    for params in model.module.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    return model, optimizer


def main():
    args = parse_args()

    # 학습 로그 처리
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # 데이터 로드
    print('=> Loading data ..')

    dataset = FlareImageDataset(config, is_train=True)
    num_data = dataset.__len__()
    num_valid = int(num_data * config.VALIDATION_RATIO)
    num_train = num_data - num_valid
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    gpus = [int(i) for i in config.GPUS.split(',')]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1, #config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1, #config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info('=> dataloader length : train({}), validation({})'.format(train_loader.__len__(), valid_loader.__len__()))

    # Cudnn 설정
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)

    # 모델 생성
    print('=> Constructing models ..')
    model = PatchFlareNet(config)

    # 모델 병렬화
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # 옵티마이저 설정
    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_precision = 0

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = \
            load_checkpoint(model, optimizer, final_output_dir, filename='patch_checkpoint.pth.tar')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        train(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        precision = validate(config, model, valid_loader, final_output_dir)

        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='patch_checkpoint.pth.tar', model_filename='patch_model_best.pth.tar')

    final_model_state_file = os.path.join(final_output_dir, 'patch_final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()