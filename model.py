"""FusionCast model definition.

The public API keeps the original RCAN/CascadeSRModel names while exposing the
paper-facing FusionCast class. A cadence is represented by the number of
sub-hourly states generated between two hourly inputs:

* steps_per_hour=1 -> 30-minute guidance
* steps_per_hour=3 -> 15-minute guidance
* steps_per_hour=5 -> 10-minute guidance
"""

from __future__ import annotations

import common
import torch
import torch.nn as nn


def make_model(args, parent=False):
    """Build a FusionCast model from an argparse-style namespace."""
    del parent
    return FusionCast(
        variables=getattr(args, "variables", 4),
        steps_per_hour=getattr(args, "steps_per_hour", 3),
        spatial_scale=getattr(args, "spatial_scale", 5),
        n_resgroups=getattr(args, "n_resgroups", 2),
        n_resblocks=getattr(args, "n_resblocks", 2),
        n_feats=getattr(args, "n_feats", 96),
        reduction=getattr(args, "reduction", 16),
    )


class CALayer(nn.Module):
    """Squeeze-and-excitation channel attention used by RCAN blocks."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        reduced_channels = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, reduced_channels, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv_du(self.avg_pool(x))


class AverageChannelAttention(nn.Module):
    """Spatially varying channel gate derived from the channel mean."""

    def __init__(self, channels):
        super().__init__()
        self.channel_weights = nn.Conv2d(1, channels, kernel_size=1)

    def forward(self, x):
        avg_values = x.mean(dim=1, keepdim=True)
        return x * torch.sigmoid(self.channel_weights(avg_values))


class RCAB(nn.Module):
    """Residual channel attention block."""

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
        modules = []
        for idx in range(2):
            modules.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if idx == 0:
                modules.append(act)
        modules.append(CALayer(n_feat))

        for idx in range(2):
            modules.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if idx == 0:
                modules.append(act)
        modules.append(AverageChannelAttention(n_feat))

        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x).mul(self.res_scale) + x


class ResidualGroup(nn.Module):
    """Stack of residual channel attention blocks."""

    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super().__init__()
        modules = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=act,
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x) + x


class RCAN(nn.Module):
    """Residual Channel Attention Network for spatial super-resolution."""

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
        kernel_size = 3
        act = nn.ReLU(True)

        self.head = nn.Sequential(conv(in_chan, n_feats, kernel_size))
        body = [
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
        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, out_channels=out_chan)
        )

    def forward(self, x):
        x = self.head(x)
        return self.tail(self.body(x) + x)


class RCANtime(RCAN):
    """RCAN variant used as the temporal interpolation operator."""

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


class FusionCast(nn.Module):
    """Semi-supervised spatiotemporal downscaling model.

    Input tensors contain three consecutive hourly coarse fields stacked on the
    channel axis: [T0 variables, T60 variables, T120 variables]. The model first
    creates sub-hourly low-resolution states for the two adjacent hourly
    intervals, constrains the central hourly anchor through a second temporal
    pass, and applies spatial super-resolution to the anchor estimate.
    """

    def __init__(
        self,
        variables=4,
        steps_per_hour=3,
        spatial_scale=5,
        n_resgroups=2,
        n_resblocks=2,
        n_feats=96,
        reduction=16,
        temporal_model=None,
        spatial_model=None,
    ):
        super().__init__()
        if variables < 1:
            raise ValueError("variables must be >= 1")
        if steps_per_hour < 1:
            raise ValueError("steps_per_hour must be >= 1")

        self.variables = int(variables)
        self.steps_per_hour = int(steps_per_hour)
        self.spatial_scale = int(spatial_scale)

        temporal_out = self.steps_per_hour * self.variables
        self.temporal_model = temporal_model or RCANtime(
            in_chan=2 * self.variables,
            out_chan=temporal_out,
            n_resgroups=n_resgroups,
            n_resblocks=n_resblocks,
            n_feats=n_feats,
            reduction=reduction,
            scale=1,
        )
        self.spatial_model = spatial_model or RCAN(
            in_chan=self.variables,
            out_chan=self.variables,
            n_resgroups=n_resgroups,
            n_resblocks=n_resblocks,
            n_feats=n_feats,
            reduction=reduction,
            scale=self.spatial_scale,
        )

    @property
    def input_channels(self):
        return 3 * self.variables

    @property
    def base_model(self):
        """Backward-compatible alias for the temporal model."""
        return self.temporal_model

    @property
    def base_model2(self):
        """Backward-compatible alias for the spatial model."""
        return self.spatial_model

    def split_subhour_frames(self, tensor):
        """Split a temporal output into per-time-step variable groups."""
        expected = self.steps_per_hour * self.variables
        if tensor.shape[1] != expected:
            raise ValueError(
                f"expected {expected} channels, got {tensor.shape[1]}"
            )
        return list(torch.split(tensor, self.variables, dim=1))

    def _central_anchor_candidates(self, first_interval, second_interval):
        first_frames = self.split_subhour_frames(first_interval)
        second_frames = self.split_subhour_frames(second_interval)

        candidates = []
        for offset, (left_frame, right_frame) in enumerate(
            zip(first_frames, second_frames)
        ):
            bridge = self.temporal_model(torch.cat([left_frame, right_frame], dim=1))
            bridge_frames = self.split_subhour_frames(bridge)
            central_index = self.steps_per_hour - offset - 1
            candidates.append(bridge_frames[central_index])
        return candidates

    def interpolate_low_res(self, x):
        """Return sub-hourly low-resolution states for both hourly intervals."""
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"FusionCast expects {self.input_channels} channels, got {x.shape[1]}"
            )
        first_interval = self.temporal_model(x[:, : 2 * self.variables])
        second_interval = self.temporal_model(x[:, self.variables :])
        return first_interval, second_interval

    def predict_subhourly(self, x):
        """Generate high-resolution sub-hourly fields for deployment inference.

        Returns a tensor with shape [batch, 2 * steps_per_hour, variables,
        high_height, high_width], corresponding to the intermediate states in
        the first and second hourly intervals.
        """
        first_interval, second_interval = self.interpolate_low_res(x)
        low_res_frames = (
            self.split_subhour_frames(first_interval)
            + self.split_subhour_frames(second_interval)
        )
        high_res_frames = [self.spatial_model(frame) for frame in low_res_frames]
        return torch.stack(high_res_frames, dim=1)

    def forward(self, x):
        first_interval, second_interval = self.interpolate_low_res(x)
        anchor_candidates = self._central_anchor_candidates(
            first_interval, second_interval
        )
        anchor_low_res = torch.stack(anchor_candidates, dim=0).mean(dim=0)
        high_res_anchor = self.spatial_model(anchor_low_res)

        return (first_interval, second_interval, *anchor_candidates, high_res_anchor)


class CascadeSRModel(FusionCast):
    """Backward-compatible name used by the original release."""

    def __init__(self, base_model=None, base_model2=None, **kwargs):
        super().__init__(
            temporal_model=base_model,
            spatial_model=base_model2,
            **kwargs,
        )
