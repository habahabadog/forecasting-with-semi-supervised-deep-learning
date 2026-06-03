from __future__ import annotations

import torch
from torch import nn

from . import common


VARIABLE_CHANNELS = 4
INTERMEDIATE_TIMEPOINTS = 11


def make_model(args, parent=False):
    return RCAN(args)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class AverageChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_weights = nn.Conv2d(1, channels, kernel_size=1)

    def forward(self, x):
        avg_values = x.mean(dim=1, keepdim=True)
        channel_weights = self.channel_weights(avg_values)
        scale = torch.sigmoid(channel_weights)
        return x * scale


class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super().__init__()
        modules_body: list[nn.Module] = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat))
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(AverageChannelAttention(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super().__init__()
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCAN(nn.Module):
    def __init__(
        self,
        in_chan=6,
        out_chan=1,
        conv=common.default_conv,
        n_resgroups=4,
        n_resblocks=4,
        n_feats=108,
        reduction=6,
        scale=10,
    ):
        super().__init__()
        self.out_chan = out_chan
        kernel_size = 3
        act = nn.ReLU(True)

        modules_head = [conv(in_chan, n_feats, kernel_size)]

        modules_body = [
            ResidualGroup(
                conv,
                n_feats,
                kernel_size,
                reduction,
                act=act,
                n_resblocks=n_resblocks,
            )
            for _ in range(n_resgroups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return self.tail(res)


class RCANtime(RCAN):
    def __init__(
        self,
        in_chan=24,
        out_chan=36,
        conv=common.default_conv,
        n_resgroups=4,
        n_resblocks=4,
        n_feats=108,
        reduction=6,
        scale=1,
    ):
        super().__init__(
            in_chan=in_chan,
            out_chan=out_chan,
            conv=conv,
            n_resgroups=n_resgroups,
            n_resblocks=n_resblocks,
            n_feats=n_feats,
            reduction=reduction,
            scale=scale,
        )


class CascadeSRModel(nn.Module):
    def __init__(self, base_model: nn.Module | None = None, base_model2: nn.Module | None = None):
        super().__init__()
        self.base_model = base_model or RCANtime(
            in_chan=8,
            out_chan=44,
            n_resgroups=2,
            n_resblocks=2,
            n_feats=32,
            reduction=16,
            scale=1,
        )
        self.base_model2 = base_model2 or RCAN(
            in_chan=4,
            out_chan=4,
            n_resgroups=2,
            n_resblocks=2,
            n_feats=96,
            reduction=16,
            scale=5,
        )

    def dynamic_timepoint_processing(self, x1, x2, num_timepoints=INTERMEDIATE_TIMEPOINTS):
        processed_timepoints = []
        for i in range(num_timepoints):
            input_start = i * VARIABLE_CHANNELS
            input_end = input_start + VARIABLE_CHANNELS
            time_pair = torch.cat(
                [x1[:, input_start:input_end, :, :], x2[:, input_start:input_end, :, :]],
                dim=1,
            )

            output_start = VARIABLE_CHANNELS * (num_timepoints - i - 1)
            output_end = output_start + VARIABLE_CHANNELS
            processed = self.base_model(time_pair)[:, output_start:output_end, :, :]
            processed_timepoints.append(processed)

        return processed_timepoints

    def forward(self, x):
        x1 = self.base_model(x[:, :8, :, :])
        x2 = self.base_model(x[:, 4:, :, :])
        processed_timepoints = self.dynamic_timepoint_processing(x1, x2)

        x_final = torch.mean(torch.stack(processed_timepoints), dim=0)
        x_final = self.base_model2(x_final)

        return x1, x2, x_final, processed_timepoints
