import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models

from models.hinet_arch import HINet
from core.loss import PSNRLoss


class PatchFlareNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.criterion = PSNRLoss(0.5)  # TODO : PSNR Loss weight to configs file

        self.generator = HINet()
        self._initialize_weights()

    def forward(self, x, y=None):
        p1, p2 = self.generator(x)

        # Pixel-wise Loss
        if y is not None:
            loss = - self.criterion(p1, y) - self.criterion(p2, y)
            return p2, loss
        return p2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
