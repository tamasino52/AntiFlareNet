from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import math
import torch
import numpy as np
from utils.vis import save_pred_batch_images

logger = logging.getLogger(__name__)


def train(config, model, optimizer, loader, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (input_img, target_img, meta) in enumerate(loader):

        data_time.update(time.time() - end)
        with torch.autograd.set_detect_anomaly(True):
            # 예측
            pred_img, loss = model(input_img, target_img)

            # 예측값과 타겟 간의 손실 계산
            losses.update(loss.item())

            # 손실 역전파
            optimizer.zero_grad()
            if loss > 0:
                loss.backward()
            optimizer.step()

            # 연산 시간 계산
            batch_time.update(time.time() - end)
            end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{:08}'.format(os.path.join(output_dir, 'train'), i)
            save_pred_batch_images(input_img, pred_img, target_img, prefix)


def validate(config, model, loader, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()

    psnr_list, mse_list = [], []

    with torch.no_grad():

        end = time.time()
        for i, (input_img, target_img, meta) in enumerate(loader):
            data_time.update(time.time() - end)

            # 예측
            pred_img, _ = model(input_img, target_img)

            # 연산 시간 계산
            batch_time.update(time.time() - end)
            end = time.time()

            # PSNR 값 계산
            pred = pred_img.detach().cpu().numpy()
            target = target_img.detach().cpu().numpy()

            for b in range(pred.shape[0]):
                mse = np.mean((pred - target) ** 2)
                mse_list.append(mse)

                if mse <= np.finfo(float).eps:
                    psnr_list.append(100.0)
                else:
                    psnr = 20 * math.log10(1.0 / math.sqrt(mse))
                    psnr_list.append(psnr)


                '''
                # 예측 이미지 저장
                prefix = '{}_{:08}_{}'.format(os.path.join(output_dir, 'valid'), meta['id'][b], 'pred')
                save_numpy_image(config, pred, prefix)

                # 입력 이미지 저장
                full_input_numpy = np.load(meta['image'][b])
                full_input_numpy = full_input_numpy[meta['stride'][b]:, meta['stride'][b]:, :]
                prefix = '{}_{:08}_{}'.format(os.path.join(output_dir, 'valid'), meta['id'][b], 'input')
                save_numpy_image(config, full_input_numpy, prefix)

                # 라벨 이미지 저장
                full_label_numpy = np.load(meta['label'][b])
                full_label_numpy = full_label_numpy[meta['stride'][b]:, meta['stride'][b]:, :]
                prefix = '{}_{:08}_{}'.format(os.path.join(output_dir, 'valid'), meta['id'][b], 'label')
                save_numpy_image(config, full_label_numpy, prefix)
                '''

            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                        i, len(loader), batch_time=batch_time,
                        speed=len(input_img) * input_img[0].size(0) / batch_time.val,
                        data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

                prefix = '{}_{:08}'.format(os.path.join(output_dir, 'valid'), i)
                save_pred_batch_images(input_img, pred_img, target_img, prefix)
    '''
    # 종합 PSNR 평가
    total_psnr = sum(total_psnr_list, 0.0) / len(total_psnr_list)
    total_mse = sum(total_mse_list, 0.0) / len(mse_list)
    total_max_psnr = max(total_psnr_list)
    total_min_psnr = min(total_psnr_list)

    msg = '(Image)\tPSNR: {mean_psnr:.4f}\t' \
          'MSE: {mean_mse:.4f}\t' \
          'MAX_PSNR: {max_psnr:.4f}\t' \
          'MIN_PSNR: {min_psnr:.4f}\n'.format(
        mean_psnr=total_psnr, mean_mse=total_mse, max_psnr=total_max_psnr, min_psnr=total_min_psnr,
    )
    '''
    # 패치 단위 PSNR 평가
    mean_psnr = sum(psnr_list, 0.0) / len(psnr_list)
    mean_mse = sum(mse_list, 0.0) / len(mse_list)
    max_psnr = max(psnr_list)
    min_psnr = min(psnr_list)
    msg = '(Patch)\tPSNR: {mean_psnr:.4f}\t' \
          'MSE: {mean_mse:.4f}\t' \
          'MAX_PSNR: {max_psnr:.4f}\t' \
          'MIN_PSNR: {min_psnr:.4f}'.format(
            mean_psnr=mean_psnr, mean_mse=mean_mse, max_psnr=max_psnr, min_psnr=min_psnr)
    logger.info(msg)

    return mean_psnr


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
