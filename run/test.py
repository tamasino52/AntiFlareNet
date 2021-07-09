from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json
import torchvision

import _init_paths
import dataset
from models.AntiFlareNet import AntiFlareNet
from utils.utils import save_checkpoint, load_checkpoint, create_logger

from core.config import config, update_config
from core.function import train, validate
from dataset.flare_image import RealFlareDataset


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

    # 테스트 설정
    config.TRAIN_CSV = config.TEST_CSV
    config.TRAIN_INPUT_DIR = config.TEST_INPUT_DIR
    config.TRAIN_LABEL_DIR = config.TEST_OUTPUT_DIR
    config.VALIDATION_RATIO = 1.0
    if os.path.exists(config.DATA_DIR + '/valid_db.pkl'):
        os.remove(config.DATA_DIR + '/valid_db.pkl')

    # 데이터 로드
    print('=> Loading data ..')
    gpus = [int(i) for i in config.GPUS.split(',')]

    test_dataset = RealFlareDataset(config, is_train=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info('=> dataloader length : test({})'.format(test_loader.__len__()))

    # Cudnn 설정
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)

    # 모델 생성
    print('=> Constructing models ..')
    model = AntiFlareNet(config, is_train=True)

    # 모델 병렬화
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # 옵티마이저 설정
    model, optimizer = get_optimizer(model)

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir)

    print('=> Predict...')
    validate(config, model, test_loader, final_output_dir)
    os.remove(config.DATA_DIR + '/valid_db.pkl')


if __name__ == '__main__':
    main()