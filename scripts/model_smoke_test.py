from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fusioncast import CascadeSRModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU shape smoke test for FusionCast.")
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--width", type=int, default=4)
    args = parser.parse_args()

    model = CascadeSRModel().eval()
    x = torch.randn(1, 12, args.height, args.width)
    with torch.no_grad():
        x1, x2, y, mid = model(x)

    expected = (1, 4, args.height * 5, args.width * 5)
    print(f"input: {tuple(x.shape)}")
    print(f"temporal pass 1: {tuple(x1.shape)}")
    print(f"temporal pass 2: {tuple(x2.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"intermediate frames: {len(mid)}")
    if tuple(y.shape) != expected:
        raise SystemExit(f"Unexpected output shape {tuple(y.shape)}; expected {expected}")


if __name__ == "__main__":
    main()
