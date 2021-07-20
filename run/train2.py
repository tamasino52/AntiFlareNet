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
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from core.scheduler import CosineAnnealingRestartLR
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

from models.AntiFlareNet import AntiFlareNet
from utils.utils import save_checkpoint, load_checkpoint, create_logger, load_model_state
from core.config import config as cfg
from core.function import train, validate
from core.patcher import _window_2D
from dataset.flare_image import FlareImageDataset
from dataset.flare_scaled_image import FlareScaledImageDataset
from models.patch_flare_net import PatchFlareNet
from models.merge_flare_net import MergeFlareNet
from core.patcher import Patcher
from utils.vis import save_torch_image
from pathlib import Path
from utils.vis import save_pred_batch_images


def get_optimizer(model):
    lr = cfg.TRAIN.LR
    for params in model.module.parameters():
        params.requires_grad = True
    # TODO : beta, max_t, eta 는 configs 파일에 정의해서 사용할 것
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr, betas=[0.9, 0.99], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 400000, 1e-7)
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

    gpus = [int(i) for i in cfg.GPUS.split(',')]

    # 데이터셋 생성
    print('=> Loading Image data ..')
    dataset = FlareImageDataset(cfg, is_train=True)

    num_data = dataset.__len__()
    num_valid = int(num_data * cfg.VALIDATION_RATIO)
    num_train = num_data - num_valid

    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])

    # 데이터로더 적재
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True)

    # 모델 생성
    print('=> Constructing Patch models ..')
    model = PatchFlareNet(cfg)

    # 모델 병렬화
    print('=> Parallelize Patch models ..')
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # 옵티마이저 설정
    model, optimizer, scheduler = get_optimizer(model)

    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    best_precision = 0
    step = 0
    if cfg.TRAIN.RESUME:
        start_epoch, model, optimizer, scheduler, precision = \
            load_checkpoint(model, optimizer, scheduler, output_dir, filename='patch_checkpoint.pth.tar')

    # 패치단위 학습
    print('=> Training patch model ..')
    patcher = Patcher(cfg)

    for epoch in range(start_epoch, end_epoch):

        # Training Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        # TODO : Flip, rotation, scale Augmentation
        for i, (input, label, meta) in enumerate(train_loader):
            # 스케일 단위 분할
            for scale in cfg.MULTI_SCALE:
                # 배치 단위 분할
                for j, (input_patch, label_patch) in enumerate(patcher.get_batch(input, label, scale)):
                    data_time.update(time.time() - end)
                    with torch.autograd.set_detect_anomaly(True):
                        # 예측
                        pred_patch, loss = model(input_patch, label_patch)

                        # 손실 계산
                        losses.update(loss.item())

                        # 손실 역전파
                        optimizer.zero_grad()
                        if loss > 0:
                            loss.backward()
                        optimizer.step()
                        scheduler.step()

                        # 연산 시간 계산
                        batch_time.update(time.time() - end)
                        end = time.time()

                    # 하나의 이미지로 저장
                    pred_patch = pred_patch.detach().cpu()
                    patcher.merge_batch(pred_patch)

                    # 학습 정보 출력
                    if step % cfg.PRINT_FREQ == 0:
                        gpu_memory_usage = torch.cuda.memory_allocated(0)
                        msg = 'Epoch: [{0}][{1}/{2}]\t' \
                              'Process: [{3}/{4}]\t' \
                              'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                              'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                              'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                              'Memory {memory:.1f}'.format(
                            epoch, i, len(train_loader), j, len(patcher),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            memory=gpu_memory_usage)
                        print(msg)

                        # 패치 이미지 출력
                        prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train_patch'), step)
                        save_pred_batch_images(input_patch, pred_patch, label_patch, prefix)
                    step += 1

                # 이미지 출력
                pred = patcher.get_image()
                prefix = '{}_{:05}_S{}'.format(os.path.join(output_dir, 'train_image'), i, int(1 / scale))
                save_pred_batch_images(input, pred, label, prefix)

        # Validation Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.eval()
        input_psnr_list, psnr_list, mse_list = [], [], []
        with torch.no_grad():
            end = time.time()
            for i, (input, label, meta) in enumerate(valid_loader):

                # 스케일 단위 분할
                for scale in cfg.MULTI_SCALE:
                    # 배치 단위 분할
                    for j, (input_patch, label_patch) in enumerate(patcher.get_batch(input, label, scale)):
                        data_time.update(time.time() - end)

                        # 예측
                        pred_patch, _ = model(input_patch, label_patch)

                        # 연산 시간 계산
                        batch_time.update(time.time() - end)
                        end = time.time()

                        # 하나의 이미지로 저장
                        pred_patch = pred_patch.detach().cpu()
                        patcher.merge_batch(pred_patch)

                        # 학습 정보 출력
                        if step % cfg.PRINT_FREQ == 0:
                            gpu_memory_usage = torch.cuda.memory_allocated(0)
                            msg = 'Test: [{0}/{1}]\t' \
                                  'Process: [{2}/{3}]\t' \
                                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                                  'Speed: {speed:.1f} samples/s\t' \
                                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                                  'Memory {memory:.1f}'.format(
                                i, len(train_loader), j, len(patcher), batch_time=batch_time,
                                speed=len(input_patch) * input_patch[0].size(0) / batch_time.val,
                                data_time=data_time, memory=gpu_memory_usage)
                            print(msg)

                            # 이미지로 출력
                            prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train'), step)
                            save_pred_batch_images(input_patch, pred_patch, label_patch, prefix)
                        step += 1

                    # 이미지 출력
                    pred = patcher.get_image()

                    prefix = '{}_{:05}_S{}'.format(os.path.join(output_dir, 'valid_image'), i, int(1 / scale))
                    save_pred_batch_images(input, pred, label, prefix)

                    # 평가
                    input = TF.normalize(input, mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
                    pred = TF.normalize(pred, mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
                    label = TF.normalize(label, mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

                    input = input.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    target = label.detach().cpu().numpy()

                    for _ in range(pred.shape[0]):
                        mse = np.mean((pred - target) ** 2)
                        mse_list.append(mse)

                        if mse <= np.finfo(float).eps:
                            psnr_list.append(100.0)
                        else:
                            psnr = 20 * math.log10(1.0 / math.sqrt(mse))
                            psnr_list.append(psnr)

                        mse = np.mean((input - target) ** 2)

                        if mse <= np.finfo(float).eps:
                            input_psnr_list.append(100.0)
                        else:
                            psnr = 20 * math.log10(1.0 / math.sqrt(mse))
                            input_psnr_list.append(psnr)

            # PSNR 평가
            input_psnr = sum(input_psnr_list, 0.0) / len(input_psnr_list)
            mean_psnr = sum(psnr_list, 0.0) / len(psnr_list)
            mean_mse = sum(mse_list, 0.0) / len(mse_list)
            max_psnr = max(psnr_list)
            min_psnr = min(psnr_list)
            msg = '(Evaluation)\t' \
                  'INPUT_PSNR: {input_psnr:.4f}\t' \
                  'PSNR: {mean_psnr:.4f}\t' \
                  'MSE: {mean_mse:.4f}\t' \
                  'MAX_PSNR: {max_psnr:.4f}\t' \
                  'MIN_PSNR: {min_psnr:.4f}'.format(
                input_psnr=input_psnr, mean_psnr=mean_psnr, mean_mse=mean_mse, max_psnr=max_psnr, min_psnr=min_psnr)
            print(msg)
            precision = mean_psnr

        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

        print('=> saving checkpoint to {} (Best: {})'.format(output_dir, best_model))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, best_model, output_dir,
            filename='patch_checkpoint.pth.tar',
            model_filename='patch_model_best.pth.tar')

    final_model_state_file = os.path.join(output_dir, 'patch_final_state.pth.tar')
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


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