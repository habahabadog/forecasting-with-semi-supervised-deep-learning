"""Shared neural-network building blocks for FusionCast."""

import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """Create a padded 2D convolution that preserves spatial size."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=kernel_size // 2,
        bias=bias,
    )


class MeanShift(nn.Conv2d):
    """Fixed 1x1 normalization layer kept for compatibility with RCAN code."""

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for parameter in self.parameters():
            parameter.requires_grad = False


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
        for idx in range(2):
            modules.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if idx == 0:
                modules.append(act)

        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x).mul(self.res_scale) + x


class Upsampler(nn.Sequential):
    """Pixel-shuffle upsampler with explicit output-channel control."""

    def __init__(self, conv, scale, n_feat, out_channels, bn=False, act=False, bias=True):
        if scale < 1 or int(scale) != scale:
            raise ValueError("scale must be a positive integer")

        scale = int(scale)
        modules = []
        if scale == 1:
            modules.append(conv(n_feat, out_channels, 3, bias))
        else:
            modules.append(conv(n_feat, out_channels * scale * scale, 3, bias))
            modules.append(nn.PixelShuffle(scale))

        if bn:
            modules.append(nn.BatchNorm2d(out_channels))
        if act:
            modules.append(act())

        super().__init__(*modules)


def validate_pixel_shuffle_scale(scale):
    """Return the integer scale used by the spatial super-resolution module."""
    scale = int(scale)
    if scale < 1:
        raise ValueError("scale must be >= 1")
    if scale > 1 and not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return scale
