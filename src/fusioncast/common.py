import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=bias,
            )
        ]
        if bn:
            modules.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            modules.append(act)
        super().__init__(*modules)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super().__init__()
        modules = []
        for i in range(2):
            modules.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules.append(act)

        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        modules = []
        if scale == 2:
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * 3, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(n_feat))
                if act:
                    modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if act:
                modules.append(act())
        elif scale in {5, 10, 25}:
            modules.append(conv(n_feat, scale * scale * 4, 3, bias))
            modules.append(nn.PixelShuffle(scale))
        elif scale == 1:
            modules.append(conv(n_feat, 44, 3, bias))
        else:
            raise NotImplementedError(f"Unsupported upsampling scale: {scale}")

        super().__init__(*modules)
