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
import os
import _init_paths
import dataset
from tqdm import tqdm
import numpy as np
import random
import cv2

from models.MPRNet import MPRNet
from core.config import config as cfg
from dataset.flare_image import FlareTestDataset
from pathlib import Path
import torch.nn.functional as F
from skimage import img_as_ubyte
from utils.utils import load_model_state
import segmentation_models_pytorch as seg

seed = 1452
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main():
    # 출력 경로 설정
    this_dir = Path(os.path.dirname(__file__))
    output_dir = (this_dir / '..' / cfg.OUTPUT_DIR).resolve()

    # 모델 생성
    print('=> Constructing models ..')
    patch_model = MPRNet()
    patch_model = load_model_state(patch_model, output_dir, 'model_best.pth.tar')

    patch_model.cuda()
    patch_model.eval()

    # 모델 생성
    print('=> Constructing merge models ..')
    merge_model = seg.DeepLabV3Plus(classes=4, activation='softmax2d')
    merge_model = load_model_state(merge_model, output_dir, 'merge_model_best.pth.tar')
    merge_model.cuda()
    merge_model.eval()

    # 데이터셋 생성
    print('=> Loading Image data ..')
    dataset = FlareTestDataset(cfg)

    # 데이터로더 적재
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        drop_last=False,
        pin_memory=True)

    ps = cfg.PATCH_SIZE
    st = cfg.STRIDE
    with torch.no_grad():
        for i, (input, file) in enumerate(tqdm(test_loader), 0):
            interim = []
            for scale in [1, 2, 4, 8]:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input_ = TF.resize(input, [int(input.shape[2] / scale), int(input.shape[3] / scale)])

                # Padding in case images are not multiples of 8
                factor = ps - st
                h, w = input_.shape[2], input_.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                padh += factor if H-h == factor else 0
                padw += factor if W-w == factor else 0
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                output = torch.zeros_like(input_)
                overlap = torch.zeros_like(input_)
                for top in range(0, H, ps-st):
                    for left in range(0, W, ps-st):
                        pred = patch_model(input_[:, :, top:top+ps, left:left+ps].cuda())
                        pred = torch.clamp(pred[0], 0, 1)
                        overlap[:, :, top:top+ps, left:left+ps] += 1.
                        output[:, :, top:top+ps, left:left+ps] += pred.cpu()

                pred = output / overlap

                # Unpad images to original dimensions
                pred = pred[:, :, :h, :w]
                pred = TF.resize(pred, [int(pred.shape[2] * scale), int(pred.shape[3] * scale)])
                interim.append(pred)

            # Merging
            input_ = TF.resize(input, [512, 512])
            pred = merge_model(input_.cuda())
            pred = pred.cpu()

            merge = 0
            pred = TF.resize(pred, [input.shape[2], input.shape[3]])
            for s in range(4):
                merge += pred[:, s].unsqueeze(1) * interim[s]

            merge = merge.permute(0, 2, 3, 1).detach().numpy()

            for batch in range(len(merge)):
                pred_img = img_as_ubyte(np.clip(merge[batch], 0, 1))
                filepath = (os.path.join(output_dir, file[batch] + '.png'))
                cv2.imwrite(filepath, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
