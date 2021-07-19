import numpy as np
import cv2
import os
import torchvision.transforms.functional as TF
import torch

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def save_pred_batch_images(input_img, pred_img, target_img, prefix, normalize=False):
    file_name = prefix + ".jpg"

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
    height = input_img.size(2)
    width = input_img.size(3)

    grid_image = np.zeros((batch_size * height, 3 * width, 3), dtype=np.uint8)

    for i in range(batch_size):
        input_np = input_img[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()

        pred_np = pred_img[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()

        target_np = target_img[i].mul(255) \
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

    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)

    imwrite(file_name, grid_image)


def save_numpy_image(config, img, prefix, normalize=False):
    file_name = prefix + ".png"
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if normalize:
        img = img.clone()
        min = float(img.min())
        max = float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
    imwrite(file_name, img)


def save_torch_image(config, img, prefix, normalize=False):
    file_name = prefix + ".png"
    if normalize:
        img = img.clone()
        min = float(img.min())
        max = float(img.max())
        img.add_(-min).div_(max - min + 1e-5)
    img = img.mul(255) \
        .clamp(0, 255) \
        .byte() \
        .permute(1, 2, 0) \
        .cpu().numpy()
    imwrite(file_name, img)
