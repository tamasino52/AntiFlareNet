import torch.nn as nn
import torch.nn.functional as F
import torch
from models.u_net import GeneratorUNet
from models.u_net_resnet import ResNetUNet
import segmentation_models_pytorch as smp


class AntiFlareNet(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.cfg = cfg
        if cfg.BACKBONE == 'u-net':
            self.generator = GeneratorUNet()
        elif cfg.BACKBONE == 'u-net-resnet':
            self.generator = ResNetUNet()
        elif cfg.BACKBONE == 'u-net++':
            self.generator = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=3)
        elif cfg.BACKBONE == 'deeplab-v3+':
            self.generator = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=3)

        self.criterion_pixelwise = None
        self.is_train = is_train
        if is_train:
            self.criterion_pixelwise = torch.nn.L1Loss()
        self._initialize_weights()

    def forward(self, x, y=None):
        p = self.generator(x)

        # Pixel-wise Loss
        if y is not None:
            loss_pixel = self.criterion_pixelwise(p, y)
            return p, loss_pixel
        else:
            return p

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
