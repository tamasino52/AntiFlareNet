import numpy as np
import cv2
import os


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


def save_pred_batch_images(prefix, *imgs):
    file_name = prefix + ".jpg"

    num_image = len(imgs)
    batch_size = imgs[0].size(0)
    height = imgs[0].size(2)
    width = imgs[0].size(3)

    grid_image = np.zeros((batch_size * height, num_image * width, 3), dtype=np.uint8)

    for j in range(num_image):
        for i in range(batch_size):
            img = imgs[j][i].mul(255) \
                .clamp(0, 255) \
                .byte() \
                .permute(1, 2, 0) \
                .cpu().numpy()

            height_begin = height * i
            height_end = height * (i + 1)

            width_begin = width * j
            width_end = width * (j + 1)

            grid_image[height_begin:height_end, width_begin:width_end, :] = img
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
