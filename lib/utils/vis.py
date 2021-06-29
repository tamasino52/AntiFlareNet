import math
import numpy as np
import torchvision
import cv2
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def save_pred_images(config, input_img, pred_img, target_img, prefix, normalize=False):
    file_name = prefix + ".png"

    if normalize:
        input_img = input_img.clone()
        min = float(input_img.min())
        max = float(input_img.max())
        input_img.add_(-min).div_(max - min + 1e-5)

        pred_img = pred_img.clone()
        min = float(pred_img.min())
        max = float(pred_img.max())
        pred_img.add_(-min).div_(max - min + 1e-5)

        target_img = target_img.clone()
        min = float(target_img.min())
        max = float(target_img.max())
        target_img.add_(-min).div_(max - min + 1e-5)

    batch_size = input_img.size(0)
    height = input_img.size(1)
    width = input_img.size(2)

    grid_image = np.zeros((batch_size * height, 3 * width, 3), dtype=np.uint8)

    for i in range(batch_size):
        input_np = input_img[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()

        pred_np = input_img[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()

        target_np = input_img[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()

        height_begin = height * i
        height_end = height * (i + 1)

        width_begin = width * 0
        width_end = width * 1
        grid_image[height_begin:height_end, width_begin:width_end, :] = input_np

        width_begin = width * 1
        width_end = width * 2
        grid_image[height_begin:height_end, width_begin:width_end, :] = pred_np

        width_begin = width * 2
        width_end = width * 3
        grid_image[height_begin:height_end, width_begin:width_end, :] = target_np

    cv2.imwrite(file_name, grid_image)
