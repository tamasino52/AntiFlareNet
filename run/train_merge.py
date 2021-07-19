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

from models.AntiFlareNet import AntiFlareNet
from utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
from core.config import config, update_config
from core.function import train, validate
from core.patcher import _window_2D
from dataset.flare_image import FlareImageDataset
from dataset.flare_scaled_image import FlareScaledImageDataset
from models.patch_flare_net import PatchFlareNet
from models.merge_flare_net import MergeFlareNet
from core.patcher import Patcher
from utils.vis import save_torch_image


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

    gpus = [int(i) for i in config.GPUS.split(',')]

    db = []
    db_path = config.DATA_DIR + '/scaled_db.pkl'

    if not os.path.exists(db_path):
        # 데이터 로드
        print('=> Loading Image data ..')
        image_dataset = FlareImageDataset(config, is_train=True)

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
        print('=> Parallelize Patch models ..')
        with torch.no_grad():
            patch_model = torch.nn.DataParallel(patch_model, device_ids=gpus).cuda()

        # 모델 가중치 로드
        print('=> Loading Best Patch models ..')
        patch_model = load_model_state(patch_model, final_output_dir, filename='patch_model_best.pth.tar')

        # 패치단위 예측
        print('=> Converting all patches to images ..')
        output_dir = os.path.join(final_output_dir, 'merge_img')
        os.makedirs(output_dir, exist_ok=True)
        patcher = Patcher(config, patch_model)
        with torch.no_grad():
            for i, (input_img, label_img, meta) in \
                    tqdm(enumerate(image_loader), desc='Patch Overlapping', total=image_loader.__len__()):
                item = {
                    'id': meta['id'],
                    'scaled_image_files': [],
                    'image_file': meta['image_file'],
                    'label_file': meta['label_file'],
                    'scaled_images': [],
                    'image': meta['image'],
                    'label': meta['label'],
                    'meta': meta
                }

                prefix = os.path.join(output_dir, 'id_i' + str(meta['id'].item()) + '_input')
                save_torch_image(config, input_img[0], prefix)
                prefix = os.path.join(output_dir, 'id_i' + str(meta['id'].item()) + '_label')
                save_torch_image(config, label_img[0], prefix)

                for scale in config.MULTI_SCALE:
                    pred_img = patcher.predict(input_img, scale).squeeze()
                    prefix = os.path.join(output_dir, 'id_i' + str(meta['id'].item()) + '_' + str(scale))
                    save_torch_image(config, pred_img, prefix)
                    np.save(prefix + '.npy', pred_img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())

                    item['scaled_image_files'].append(prefix + '.png')
                    item['scaled_images'].append(prefix + '.npy')
                db.append(item)

        # 데이터베이스 저장
        pkl_file = open(db_path, 'wb')
        pickle.dump(db, pkl_file)
        pkl_file.close()
        logger.info('=> save db to pkl : {}'.format(db_path))

        torch.cuda.empty_cache()

    # 스케일 피라미드 데이터 로드
    print('=> Loading Scaled Image data ..')
    dataset = FlareScaledImageDataset(config, is_train=True)

    num_data = dataset.__len__()
    num_valid = int(num_data * config.VALIDATION_RATIO)
    num_train = num_data - num_valid

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info('=> dataloader length : train({}), validation({})'.format(train_loader.__len__(), valid_loader.__len__()))

    # 모델 생성
    print('=> Constructing Patch models ..')
    merge_model = MergeFlareNet(config)

    # 모델 병렬화
    print('=> Parallelize Patch models ..')
    with torch.no_grad():
        merge_model = torch.nn.DataParallel(merge_model, device_ids=gpus).cuda()

    # 옵티마이저 설정
    merge_model, optimizer = get_optimizer(merge_model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_precision = 0

    if config.TRAIN.RESUME:
        start_epoch, merge_model, optimizer, best_precision = load_checkpoint(merge_model, optimizer, final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        train(config, merge_model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        precision = validate(config, merge_model, valid_loader, final_output_dir)

        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': merge_model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(merge_model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
