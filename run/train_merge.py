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
import torchvision
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

from models.AntiFlareNet import AntiFlareNet
from utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
from core.config import config, update_config
from core.function import train, validate
from dataset.flare_image import FlareImageDataset
from dataset.flare_patch import FlarePatchDataset
from dataset.flare_image_to_patch import FlarePatch2ImageDataset
from models.patch_flare_net import PatchFlareNet


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

    # Cudnn 설정
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)

    # 데이터 로드
    print('=> Loading Image data ..')
    image_dataset = FlareImageDataset(config, is_train=True)

    gpus = [int(i) for i in config.GPUS.split(',')]
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    # 모델 생성
    print('=> Constructing Patch models ..')
    patch_model = PatchFlareNet(config)

    # 모델 병렬화
    with torch.no_grad():
        patch_model = torch.nn.DataParallel(patch_model, device_ids=gpus).cuda()
    patch_model = load_model_state(patch_model, final_output_dir, filename='patch_model_best.pth')

    print('=> Converting all patches to images ..')

    with torch.no_grad():
        for i, (input_img, label_img, meta) in tqdm(enumerate(image_loader), desc='Overlapping', total=image_loader.__len__()):
            #TODO
            #patches = input_img.unfold(1, kernel_size, int(stride)).unfold(0, kernel_size, int(stride))  # [31, 31, 16, 16]
            kernel_size = config.PATCH_SIZE
            stride = config.STRIDE
            input_img = input_img.squeeze()

            patches = input_img.unfold(1, kernel_size, kernel_size-stride).unfold(2, kernel_size, kernel_size-stride)
            shape = patches.shape

            # 예측
            pred_img, _ = patch_model(input_img)

            pred = pred_img.detach().cpu().numpy()
            target = target_img.detach().cpu().numpy()















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
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()