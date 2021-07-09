import torch.nn as nn
import torch.nn.functional as F
import torch
from models.u_net import GeneratorUNet
from models.u_net_resnet import ResNetUNet
import segmentation_models_pytorch as smp


class PatchFlareNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.criterion = torch.nn.L1Loss()

        if cfg.PATCH_MODEL == 'u-net':
            self.generator = GeneratorUNet()
        elif cfg.PATCH_MODEL == 'u-net-resnet':
            self.generator = smp.Unet(classes=3, encoder_weights='imagenet')
        elif cfg.PATCH_MODEL == 'u-net++':
            self.generator = smp.UnetPlusPlus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=3)
        elif cfg.PATCH_MODEL == 'deeplab-v3+':
            self.generator = smp.DeepLabV3Plus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                in_channels=3,
                classes=3)

        self.criterion = torch.nn.L1Loss()
        self._initialize_weights()

    def forward(self, x, y=None):
        p = self.generator(x)

        # Pixel-wise Loss
        if y is not None:
            loss = self.criterion(p, y)
            return p, loss
        return p

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.normal_(m.weight, 0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.normal_(m.weight, 0, 0.001)
