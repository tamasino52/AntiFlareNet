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
from utils.utils import AverageMeter

from models.MPRNet import MPRNet
from utils.utils import load_model_state
from core.config import config as cfg
from dataset.flare_image import FlareImageDataset, FlareMultiScaleTrainDataset
from utils.vis import save_pred_batch_images
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
            break
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

    # 모델 생성
    print('=> Constructing merge models ..')
    model = seg.DeepLabV3Plus(classes=4, activation='softmax2d')
    model.cuda()

    # 옵티마이저 설정
    model, optimizer, scheduler = get_optimizer(model)

    start_epoch = cfg.TRAIN.BEGIN_EPOCH
    end_epoch = cfg.TRAIN.END_EPOCH
    best_precision = 0

    # 이어하기 설정
    if cfg.TRAIN.RESUME:
        start_epoch, model, optimizer, scheduler, precision = load_checkpoint(model, optimizer, output_dir, scheduler, filename='merge_checkpoint.pth.tar')
        print("=> Resuming Training with learning rate: {0:.6f}".format(scheduler.get_lr()[0]))

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.train()

    # 손실 정의
    criterion = nn.MSELoss()

    # 데이터셋 생성
    print('=> Loading Merge data ..')
    dataset = FlareMultiScaleTrainDataset(cfg)

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

    print('=> Training merge model ..')
    print('=> Start Epoch {} End Epoch {}'.format(start_epoch, cfg.TRAIN.END_EPOCH))

    for epoch in range(start_epoch, end_epoch + 1):
        # Training Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()

        end = time.time()
        # TODO : Flip, rotation, scale Augmentation
        for i, (input, interim, target, file) in enumerate(train_loader):
            optimizer.zero_grad()

            input = input.cuda()
            interim = [inr.cuda() for inr in interim]
            target = target.cuda()

            pred = model(input)

            # 연산 시간 계산
            batch_time.update(time.time() - end)
            end = time.time()

            # 손실 역전파
            merge = 0
            for s in range(4):
                merge += pred[:, s].unsqueeze(1) * interim[s]
            loss = criterion(merge, target)

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
                mat = torch.ones_like(input)
                att = []
                for s in range(4):
                    att.append(pred[:, s].unsqueeze(1) * mat)

                # 패치 이미지 출력
                prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)
                save_pred_batch_images(prefix, input, merge, target, *att)

        # Validation Loop
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.eval()
        psnr_val_rgb = []
        with torch.no_grad():
            end = time.time()
            for i, (input, interim, target, file) in enumerate(valid_loader):
                input = input.cuda()
                interim = [inr.cuda() for inr in interim]
                target = target.cuda()

                with torch.no_grad():
                    pred = model(input)

                merge = 0
                for s in range(4):
                    merge += pred[:, s].unsqueeze(1) * interim[s]

                for pre, tar in zip(merge, target):
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
                    mat = torch.ones_like(input)
                    att = []
                    for s in range(4):
                        att.append(pred[:, s].unsqueeze(1) * mat)

                    prefix = '{}_{:08}'.format(os.path.join(output_dir, 'valid'), i)
                    save_pred_batch_images(prefix, input, merge, target, *att)

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
                'state_dict': model.module.state_dict(),
                'precision': best_precision,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
            }, best_model, output_dir,
                filename='merge_checkpoint.pth.tar', model_filename='merge_model_best.pth.tar')

            print('=> saving checkpoint to {} (Best: {})'.format(output_dir, best_model))

    scheduler.step()

    final_model_state_file = os.path.join(output_dir)
    print('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
