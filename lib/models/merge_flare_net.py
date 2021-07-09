import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from models.u_net import GeneratorUNet
from models.u_net_resnet import ResNetUNet
import segmentation_models_pytorch as smp


class MergeFlareNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.multi_scale = cfg.MULTI_SCALE
        self.num_class = len(cfg.MULTI_SCALE)
        self.softmax = nn.Softmax2d()

        if cfg.MERGE_MODEL == 'u-net':
            self.model = smp.Unet(encoder_name="resnet34",
                                  encoder_weights="imagenet",
                                  encoder_depth=5,
                                  in_channels=3,
                                  classes=self.num_class,
                                  )
            self.input_size = (512, 512)

        self._initialize_weights()

    def forward(self, x):
        size = x.shape[2:]
        x = TF.resize(x, self.input_size)
        p = self.model(x)
        p = TF.resize(p, size)
        p = self.softmax(p)
        return p

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
