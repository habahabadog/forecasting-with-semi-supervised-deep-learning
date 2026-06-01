"""Compatibility loader for the original 15-minute split checkpoints.

The files `base_model_500.pth` and `base_model2_500.pth` were produced by the
earlier experiment code before the public API was cleaned up. They are included
for provenance and can be loaded through this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model import RCAN, RCANtime


class LegacyFusionCast15Min(nn.Module):
    """Original 15-minute cascade used by the split checkpoint files."""

    def __init__(self):
        super().__init__()
        self.base_model = RCANtime(
            in_chan=24,
            out_chan=36,
            n_resgroups=4,
            n_resblocks=4,
            n_feats=36,
            reduction=6,
            scale=1,
        )
        self.base_model2 = RCAN(
            in_chan=12,
            out_chan=3,
            n_resgroups=4,
            n_resblocks=4,
            n_feats=36,
            reduction=6,
            scale=2,
        )

    def forward(self, x):
        if x.shape[1] != 36:
            raise ValueError(f"expected 36 input channels, got {x.shape[1]}")

        x1 = self.base_model(x[:, :24])
        x2 = self.base_model(x[:, 12:])

        x_concat1 = torch.cat([x1[:, :12], x2[:, :12]], dim=1)
        x_concat2 = torch.cat([x1[:, 12:24], x2[:, 12:24]], dim=1)
        x_concat3 = torch.cat([x1[:, 24:], x2[:, 24:]], dim=1)

        x3 = self.base_model(x_concat1)[:, 24:]
        x4 = self.base_model(x_concat2)[:, 12:24]
        x5 = self.base_model(x_concat3)[:, :12]
        x6 = self.base_model2((x3 + x4 + x5) / 3)
        return x3, x4, x5, x6


def load_legacy_15min_checkpoint(
    temporal_path="checkpoints/pretrained/fusioncast_15min_temporal_legacy.pth",
    spatial_path="checkpoints/pretrained/fusioncast_15min_spatial_legacy.pth",
    map_location="cpu",
):
    """Load the original temporal and spatial state_dict checkpoints."""
    model = LegacyFusionCast15Min()
    temporal_state = torch.load(temporal_path, map_location=map_location)
    spatial_state = torch.load(spatial_path, map_location=map_location)
    model.base_model.load_state_dict(temporal_state)
    model.base_model2.load_state_dict(spatial_state)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal",
        default="checkpoints/pretrained/fusioncast_15min_temporal_legacy.pth",
    )
    parser.add_argument(
        "--spatial",
        default="checkpoints/pretrained/fusioncast_15min_spatial_legacy.pth",
    )
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_legacy_15min_checkpoint(args.temporal, args.spatial)
    model.eval()
    x = torch.randn(1, 36, args.height, args.width)
    with torch.no_grad():
        outputs = model(x)
    print("legacy 15-minute checkpoint loaded.")
    for idx, output in enumerate(outputs):
        print(f"output[{idx}] shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
