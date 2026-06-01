"""Run a no-data forward-pass check for FusionCast."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import FusionCast


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variables", type=int, default=4)
    parser.add_argument("--spatial-scale", type=int, default=5)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--paper-config", action="store_true")
    parser.add_argument(
        "--demo-data-dir",
        default=None,
        help=(
            "Optional directory of pre-stacked demo .npy tensors with shape "
            "[3 * variables, height, width]."
        ),
    )
    parser.add_argument(
        "--demo-steps-per-hour",
        type=int,
        default=3,
        help="Cadence to use when running --demo-data-dir, default 3 for 15 minutes.",
    )
    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def model_kwargs(use_paper_config):
    if use_paper_config:
        return {"n_resgroups": 2, "n_resblocks": 2, "n_feats": 96, "reduction": 16}
    return {"n_resgroups": 1, "n_resblocks": 1, "n_feats": 16, "reduction": 4}


def run_case(steps_per_hour, args, device):
    model = FusionCast(
        variables=args.variables,
        steps_per_hour=steps_per_hour,
        spatial_scale=args.spatial_scale,
        **model_kwargs(args.paper_config),
    ).to(device)
    model.eval()

    x = torch.randn(
        1,
        3 * args.variables,
        args.height,
        args.width,
        device=device,
    )
    with torch.no_grad():
        outputs = model(x)
        high_res_sequence = model.predict_subhourly(x)

    print(f"steps_per_hour={steps_per_hour}")
    for idx, output in enumerate(outputs):
        print(f"  output[{idx}] shape: {tuple(output.shape)}")
    print(f"  high-res subhourly sequence: {tuple(high_res_sequence.shape)}")


def load_demo_batch(demo_data_dir, variables):
    demo_path = Path(demo_data_dir)
    if not demo_path.exists():
        raise FileNotFoundError(f"demo data directory not found: {demo_path}")

    files = sorted(demo_path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy files found in {demo_path}")

    expected_channels = 3 * variables
    tensors = []
    for file_path in files:
        array = np.load(file_path).astype("float32")
        if array.ndim != 3 or array.shape[0] != expected_channels:
            raise ValueError(
                f"{file_path} must have shape "
                f"[{expected_channels}, height, width], got {array.shape}"
            )
        tensors.append(torch.from_numpy(array))

    return files, torch.stack(tensors, dim=0)


def run_demo_data(args, device):
    files, demo_batch = load_demo_batch(args.demo_data_dir, args.variables)
    model = FusionCast(
        variables=args.variables,
        steps_per_hour=args.demo_steps_per_hour,
        spatial_scale=args.spatial_scale,
        **model_kwargs(args.paper_config),
    ).to(device)
    model.eval()

    demo_batch = demo_batch.to(device)
    with torch.no_grad():
        high_res_sequence = model.predict_subhourly(demo_batch)

    print("demo-data smoke test")
    print(f"  files: {[file_path.name for file_path in files]}")
    print(f"  input batch: {tuple(demo_batch.shape)}")
    print(f"  high-res subhourly sequence: {tuple(high_res_sequence.shape)}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    for steps_per_hour in (1, 3, 5):
        run_case(steps_per_hour, args, device)

    if args.demo_data_dir:
        run_demo_data(args, device)


if __name__ == "__main__":
    main()
