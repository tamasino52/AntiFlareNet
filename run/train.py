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
from models.MPRNet import MPRNet

from utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
from core.config import config as cfg
from dataset.flare_image import FlareTrainDataset
from pathlib import Path
from utils.vis import save_pred_batch_images
from core.scheduler import GradualWarmupScheduler
from core.metrics import torchPSNR, numpyPSNR
from core.losses import CharbonnierLoss, EdgeLoss
from utils.utils import AverageMeter

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

    # Cudnn 설정
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)

    # 모델 생성
    print('=> Constructing Patch models ..')
    model = MPRNet()
    model.cuda()

    # 옵티마이저 설정
    model, optimizer, scheduler = get_optimizer(model)

    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    best_precision = 0

    # 이어하기 설정
    if cfg.TRAIN.RESUME:
        start_epoch, model, optimizer, scheduler, precision = load_checkpoint(model, optimizer, output_dir, scheduler)
        print("=> Resuming Training with learning rate: {0:.6f}".format(scheduler.get_lr()[0]))

    # 모델 병렬화
    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    # 손실 정의
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()

    # 데이터셋 생성
    print('=> Loading Image data ..')
    dataset = FlareTrainDataset(cfg)

    num_data = dataset.__len__()
    num_valid = int(num_data * cfg.VALIDATION_RATIO)
    num_train = num_data - num_valid

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    # 데이터로더 적재
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        drop_last=False,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        drop_last=False,
        pin_memory=True)

    print('=> Training patch model ..')
    print('=> Start Epoch {} End Epoch {}'.format(start_epoch, cfg.TRAIN.END_EPOCH))

    for epoch in range(start_epoch, end_epoch + 1):
        # Training Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()

        end = time.time()
        # TODO : Flip, rotation, scale Augmentation
        for i, (input, target, file) in enumerate(train_loader):
            optimizer.zero_grad()

            input = input.cuda()
            target = target.cuda()

            pred = model(input)

            # 연산 시간 계산
            batch_time.update(time.time() - end)
            end = time.time()

            # 손실 역전파
            loss_char = torch.sum(torch.stack([criterion_char(pred[j], target) for j in range(len(pred))]))
            loss_edge = torch.sum(torch.stack([criterion_edge(pred[j], target) for j in range(len(pred))]))
            loss = (loss_char) + (0.05 * loss_edge)
            losses.update(loss.item())
            loss.backward()
            optimizer.step()

            # 학습 정보 출력
            if i % cfg.PRINT_FREQ == 0:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                      'Memory: {memory:.1f}'.format(
                        epoch, i, len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        memory=gpu_memory_usage)
                print(msg)

                # 패치 이미지 출력
                prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)
                save_pred_batch_images(prefix, input, pred[0], target)

        # Validation Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.eval()
        psnr_val_rgb = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target, file) in enumerate(valid_loader):
                input = input.cuda()
                target = target.cuda()
                with torch.no_grad():
                    pred = model(input)
                pred = pred[0]

                for pre, tar in zip(pred, target):
                    psnr_val_rgb.append(torchPSNR(pre, tar))

                # 연산 시간 계산
                batch_time.update(time.time() - end)
                end = time.time()

                # 학습 정보 출력
                if i % cfg.PRINT_FREQ == 0:
                    gpu_memory_usage = torch.cuda.memory_allocated(0)
                    msg = 'Test: [{0}/{1}]\t' \
                          'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed: {speed:.1f} samples/s\t' \
                          'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Memory {memory:.1f}'.format(
                        i, len(valid_loader), batch_time=batch_time,
                        speed=len(input) * input[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                    print(msg)
                    prefix = '{}_{:08}'.format(os.path.join(output_dir, 'valid'), i)
                    save_pred_batch_images(prefix, input, pred, target)

            # PSNR 평가
            precision = torch.stack(psnr_val_rgb).mean().item()

            if precision > best_precision:
                best_precision = precision
                best_epoch = epoch
                best_model = True
            else:
                best_model = False
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, precision, best_epoch, best_precision))

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'precision': best_precision,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
            }, best_model, output_dir)

            print('=> saving checkpoint to {} (Best: {})'.format(output_dir, best_model))

    scheduler.step()

    final_model_state_file = os.path.join(output_dir)
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
