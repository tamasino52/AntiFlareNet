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

from utils.utils import save_checkpoint, load_checkpoint, create_logger
import _init_paths
import dataset
from models.AntiFlareNet import AntiFlareNet

from core.config import config, update_config
from core.function import train, validate
from dataset.flare_real import RealFlareDataset


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
    train_dataset = None
    if config.DATA_CLASS == 'real':
        train_dataset = RealFlareDataset(config, is_train=True)
    elif config.DATA_CLASS == 'synthetic':
        # TODO
        raise NotImplementedError
    elif config.DATA_CLASS == 'mix':
        raise NotImplementedError
    else:
        assert "Unknown data class error. Set DATA_CLASS of configs to 'real' or 'synthetic' or 'mix'."

    gpus = [int(i) for i in config.GPUS.split(',')]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    test_dataset = RealFlareDataset(config, is_train=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info('=> dataloader length : train({}), test({})'.format(train_loader.__len__(), test_loader.__len__()))

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

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_precision = 0

    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        # lr_scheduler.step()
        train(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        precision = validate(config, model, test_loader, final_output_dir)

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
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()