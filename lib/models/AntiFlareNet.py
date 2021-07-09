import torch.nn as nn
import torch.nn.functional as F
import torch
import scipy.signal
import numpy as np
from models.merge_flare_net import MergeFlareNet
from models.patch_flare_net import PatchFlareNet


class AntiFlareNet(nn.Module):
    def __init__(self, cfg, is_train):
        super().__init__()
        self.cfg = cfg
        self.is_train = is_train

        self.patch_model = PatchFlareNet(cfg)
        self.merge_model = MergeFlareNet(cfg)
        self.criterion = torch.nn.L1Loss()
        self.multi_scale = cfg.MULTI_SCALE
        self.num_scale = len(cfg.MULTI_SCALE)
        self.patcher = Patcher(cfg)

        self._initialize_weights()

    def forward(self, x, y=None):
        # x : b, c, h, w
        p_weight = self.merge_model(x)
        # TODO : memory 관리 필요
        p_list = []
        for layer in range(self.num_scale):
            p = self.patcher.split(x, layer)  # b*l, c, k, k
            p = self.patch_model(p)  # b*l, c, k, k
            p = self.patcher.fade(p, layer)  # b*l, c, k, k
            p = self.patcher.merge(p, layer)  # b, c, h, w
            p = p * p_weight[:, layer].unsqueeze(1)  # b, c, h, w
            p_list.append(p.unsqueeze(1))  # b, 1, c, h, w

        result = torch.stack(p_list, dim=1).sum(dim=1)  # b, s, c, h, w -> b, c, h, w
        if y is not None:
            loss = self.criterion(p, y)
            return p, loss
        return result

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)


class Patcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.multi_scale = cfg.MULTI_SCALE
        self.num_scale = len(cfg.MULTI_SCALE)
        self.patch_size = cfg.PATCH_SIZE
        self.stride = cfg.STRIDE
        self.unfold_layers = []
        self.size_list = [0] * self.num_scale
        self.window_list = []
        for scale in self.multi_scale:
            wind = get_window_2D(int(self.patch_size * scale), int(self.stride * scale))
            self.window_list.append(torch.from_numpy(wind).float().unsqueeze(0).cuda())

    def split(self, x, layer):
        scale = self.multi_scale[layer]
        k = int(self.patch_size * scale)
        s = int(self.stride * scale)
        b = x.shape[0]
        c = x.shape[1]
        self.size_list[layer] = x.shape

        p = F.unfold(input=x, kernel_size=k, stride=k-s, padding=s)
        p = p.view(b, c, k, k, -1)
        p = torch.permute(p, (0, 4, 1, 2, 3))
        p = p.view(-1, c, k, k)
        return p  # b*l, c, k k

    def fade(self, x, layer):
        wind = self.window_list[layer]
        p = x * wind
        return p

    def merge(self, x, layer):
        scale = self.multi_scale[layer]
        k = int(self.patch_size * scale)
        s = int(self.stride * scale)
        b = self.size_list[layer][0]
        c = self.size_list[layer][1]

        x = x.view(b, -1, c, k, k)  # b*l, c, k, k -> b, l, c, k, k
        x = torch.permute(x, (0, 2, 3, 4, 1))  # b,c,k,k,l
        x = x.view(b, c*k*k, -1)  # b, c*k*k, l

        p = F.fold(input=x, output_size=self.size_list[layer][2:], kernel_size=k, stride=k-s, padding=s)
        return p


def get_spline_window(window_size, power=2):

    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def get_window_2D(window_size, padding, power=2):
    wind_1d = get_spline_window(2 * padding, power)
    wind = np.ones(window_size, dtype=np.float_)
    border = int(len(wind_1d)/2)
    wind[:border] = wind_1d[:border]
    wind[-border:] = wind_1d[border:]
    wind = wind[np.newaxis, :, np.newaxis]
    wind = wind * wind.transpose(0, 2, 1)
    return wind / wind.max()
