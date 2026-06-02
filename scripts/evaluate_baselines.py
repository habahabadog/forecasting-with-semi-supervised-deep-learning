"""Evaluate interpolation baselines and optional legacy FusionCast outputs.

The script supports two data layouts:

* public: hourly files named YYYYMMDDHH.npy with shape [variables, H, W]
* legacy: hourly files named YYYY_MM_DD_HH.npy with shape [12, H, W]

For the legacy layout, channels are interpreted as three variables over four
within-hour slots. Metrics are computed on channels [0, 4, 8], matching the
legacy checkpoint target used by the archived experiments.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy_15min import load_legacy_15min_checkpoint  # noqa: E402


PUBLIC_VARIABLES = ("temperature", "u10", "v10", "precipitation")
LEGACY_VARIABLES = ("temperature", "u10", "v10")
LEGACY_TARGET_CHANNELS = (0, 4, 8)
LEGACY_MEAN = np.array(
    [
        2.93113343e02,
        2.94473137e02,
        2.95870316e02,
        2.97293362e02,
        -9.62603878e-01,
        -9.63820933e-01,
        -8.17060305e-01,
        -6.38883960e-01,
        1.00693005e00,
        6.23785707e-01,
        3.60441722e-01,
        1.87235837e-01,
    ],
    dtype=np.float32,
).reshape(12, 1, 1)
LEGACY_STD = np.array(
    [
        5.29269216,
        5.35595765,
        5.39171511,
        5.42697679,
        2.29167805,
        1.99422925,
        1.65863338,
        1.30963854,
        3.37417093,
        2.96718336,
        2.50736727,
        1.95559884,
    ],
    dtype=np.float32,
).reshape(12, 1, 1)
LEGACY_HIGH_MEAN = LEGACY_MEAN[list(LEGACY_TARGET_CHANNELS)]
LEGACY_HIGH_STD = LEGACY_STD[list(LEGACY_TARGET_CHANNELS)]


@dataclass(frozen=True)
class SampleWindow:
    left: str
    middle: str
    right: str
    target_time: datetime


class NpySource:
    def __init__(self, path: Path):
        self.path = path
        self._zip_file: zipfile.ZipFile | None = None
        if path.is_file() and path.suffix.lower() == ".zip":
            self._zip_file = zipfile.ZipFile(path)
            self.files = sorted(
                name for name in self._zip_file.namelist() if name.endswith(".npy")
            )
        else:
            self.files = sorted(str(item) for item in path.glob("*.npy"))

        if not self.files:
            raise FileNotFoundError(f"no .npy files found in {path}")

    def close(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()

    def load(self, name: str) -> np.ndarray:
        if self._zip_file is not None:
            with self._zip_file.open(name) as handle:
                return np.load(io.BytesIO(handle.read())).astype(np.float32)
        return np.load(name).astype(np.float32)


def parse_timestamp(file_name: str) -> datetime:
    stem = Path(file_name).stem
    for fmt in ("%Y%m%d%H", "%Y_%m_%d_%H"):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp format: {file_name}")


def detect_format(source: NpySource, requested: str) -> str:
    if requested != "auto":
        return requested
    channels = int(source.load(source.files[0]).shape[0])
    if channels == 12:
        return "legacy"
    return "public"


def parse_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    for fmt in ("%Y%m%d%H", "%Y_%m_%d_%H"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"unsupported date format: {value}")


def build_windows(
    files: list[str],
    start: datetime | None,
    end: datetime | None,
) -> list[SampleWindow]:
    by_time = {parse_timestamp(name): name for name in files}
    windows: list[SampleWindow] = []
    for timestamp in sorted(by_time):
        middle = timestamp + timedelta(hours=1)
        right = timestamp + timedelta(hours=2)
        if start is not None and middle < start:
            continue
        if end is not None and middle > end:
            continue
        if middle in by_time and right in by_time:
            windows.append(
                SampleWindow(by_time[timestamp], by_time[middle], by_time[right], middle)
            )
    return windows


def split_windows(files: list[str], split_timestamp: str, train_fraction: float) -> dict[str, list[SampleWindow]]:
    split_time = parse_date(split_timestamp)
    if split_time is None:
        raise ValueError("split timestamp is required")
    before = [name for name in files if parse_timestamp(name) < split_time]
    after = [name for name in files if parse_timestamp(name) >= split_time]
    train_candidates = sorted(before, key=parse_timestamp)
    train_size = max(3, int(len(train_candidates) * train_fraction))
    train_files = train_candidates[:train_size]
    val_files = train_candidates[train_size:]
    if len(val_files) < 3:
        val_files = train_candidates[-3:]
    if len(after) < 3:
        after = val_files
    return {
        "train": build_windows(train_files, None, None),
        "validation": build_windows(val_files, None, None),
        "test": build_windows(after, None, None),
    }


def crop_to_target(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return prediction[:, : target.shape[-2], : target.shape[-1]]


def upsample(coarse: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(coarse).unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(tensor, size=size, mode=mode)
    else:
        out = F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return out.squeeze(0).numpy()


def public_fields(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    spatial_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, middle, right = arrays
    target = middle[:, :-1, :-1]
    height, width = target.shape[-2:]
    low_left = left[:, :-1:spatial_scale, :-1:spatial_scale]
    low_middle = middle[:, :-1:spatial_scale, :-1:spatial_scale]
    low_right = right[:, :-1:spatial_scale, :-1:spatial_scale]
    return low_left, low_middle, low_right, target[:, :height, :width]


def legacy_fields(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    spatial_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, middle, right = arrays
    channels = list(LEGACY_TARGET_CHANNELS)
    target = middle[channels, :-1, :-1]
    low_left = left[channels, :-1:spatial_scale, :-1:spatial_scale]
    low_middle = middle[channels, :-1:spatial_scale, :-1:spatial_scale]
    low_right = right[channels, :-1:spatial_scale, :-1:spatial_scale]
    return low_left, low_middle, low_right, target


def legacy_model_input(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> torch.Tensor:
    normalized = [
        ((array[:, :-1:2, :-1:2] - LEGACY_MEAN) / LEGACY_STD).astype(np.float32)
        for array in arrays
    ]
    stacked = np.concatenate(normalized, axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)


def denormalize_legacy_output(output: torch.Tensor) -> np.ndarray:
    array = output.squeeze(0).detach().cpu().numpy()
    return array * LEGACY_HIGH_STD + LEGACY_HIGH_MEAN


def compute_metrics(error: np.ndarray, variable_names: tuple[str, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "mae_overall": float(np.mean(np.abs(error))),
        "rmse_overall": float(np.sqrt(np.mean(error**2))),
    }
    for idx, name in enumerate(variable_names):
        metrics[f"mae_{name}"] = float(np.mean(np.abs(error[idx])))
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(error[idx] ** 2)))

    if "u10" in variable_names and "v10" in variable_names:
        u_idx = variable_names.index("u10")
        v_idx = variable_names.index("v10")
        metrics["wind_vector_rmse"] = float(
            np.sqrt(np.mean(error[u_idx] ** 2 + error[v_idx] ** 2))
        )
    return metrics


def add_metrics(total: dict[str, float], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + value


def evaluate_scope(
    source: NpySource,
    windows: list[SampleWindow],
    data_format: str,
    spatial_scale: int,
    max_windows: int | None,
    legacy_model: torch.nn.Module | None,
    device: torch.device,
) -> list[dict[str, str | float | int]]:
    if max_windows is not None:
        windows = windows[:max_windows]
    if not windows:
        return []

    variable_names = LEGACY_VARIABLES if data_format == "legacy" else PUBLIC_VARIABLES
    method_totals: dict[str, dict[str, float]] = {}
    method_counts: dict[str, int] = {}

    for window in windows:
        arrays = (source.load(window.left), source.load(window.middle), source.load(window.right))
        if data_format == "legacy":
            low_left, low_middle, low_right, target = legacy_fields(arrays, spatial_scale)
        else:
            low_left, low_middle, low_right, target = public_fields(arrays, spatial_scale)

        target_size = target.shape[-2:]
        coarse_methods = {
            "central": low_middle,
            "temporal_linear": (low_left + low_right) / 2.0,
            "persistence_left": low_left,
        }
        for coarse_name, coarse in coarse_methods.items():
            for mode in ("nearest", "bilinear", "bicubic"):
                method = f"{coarse_name}_{mode}"
                prediction = crop_to_target(upsample(coarse, target_size, mode), target)
                metrics = compute_metrics(prediction - target, variable_names)
                add_metrics(method_totals.setdefault(method, {}), metrics)
                method_counts[method] = method_counts.get(method, 0) + 1

        if legacy_model is not None and data_format == "legacy":
            model_input = legacy_model_input(arrays).to(device)
            with torch.no_grad():
                prediction = denormalize_legacy_output(legacy_model(model_input)[-1])
            prediction = crop_to_target(prediction, target)
            metrics = compute_metrics(prediction - target, variable_names)
            add_metrics(method_totals.setdefault("fusioncast_legacy", {}), metrics)
            method_counts["fusioncast_legacy"] = method_counts.get("fusioncast_legacy", 0) + 1

    rows: list[dict[str, str | float | int]] = []
    for method in sorted(method_totals):
        count = method_counts[method]
        row: dict[str, str | float | int] = {"method": method, "windows": count}
        for key, value in sorted(method_totals[method].items()):
            row[key] = value / count
        rows.append(row)
    return rows


def parse_case(case_spec: str) -> tuple[str, datetime, datetime]:
    parts = [part.strip() for part in case_spec.split(",")]
    if len(parts) != 3:
        raise ValueError("--case must use name,start,end")
    start = parse_date(parts[1])
    end = parse_date(parts[2])
    if start is None or end is None:
        raise ValueError("--case requires start and end")
    return parts[0], start, end


def write_rows(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory or .zip file containing .npy data.")
    parser.add_argument("--format", choices=("auto", "public", "legacy"), default="auto")
    parser.add_argument("--spatial-scale", type=int, default=None)
    parser.add_argument("--split-timestamp", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--start", default=None, help="Optional target-time start.")
    parser.add_argument("--end", default=None, help="Optional target-time end.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case as name,start,end. Example: maon,2022082513,2022082513",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--legacy-checkpoint", action="store_true")
    parser.add_argument(
        "--legacy-temporal",
        default="checkpoints/pretrained/fusioncast_15min_temporal_legacy.pth",
    )
    parser.add_argument(
        "--legacy-spatial",
        default="checkpoints/pretrained/fusioncast_15min_spatial_legacy.pth",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="baseline_metrics.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = NpySource(Path(args.data))
    try:
        data_format = detect_format(source, args.format)
        spatial_scale = args.spatial_scale or (2 if data_format == "legacy" else 5)
        device = torch.device(args.device)

        legacy_model = None
        if args.legacy_checkpoint:
            if data_format != "legacy":
                raise ValueError("--legacy-checkpoint is only valid for legacy data")
            legacy_model = load_legacy_15min_checkpoint(
                args.legacy_temporal,
                args.legacy_spatial,
                map_location=device,
            ).to(device)
            legacy_model.eval()

        all_rows: list[dict[str, str | float | int]] = []
        if args.case:
            for case_spec in args.case:
                name, start, end = parse_case(case_spec)
                windows = build_windows(source.files, start, end)
                for row in evaluate_scope(
                    source,
                    windows,
                    data_format,
                    spatial_scale,
                    args.max_windows,
                    legacy_model,
                    device,
                ):
                    all_rows.append({"scope": name, **row})
        else:
            if args.split_timestamp is not None:
                scopes = split_windows(source.files, args.split_timestamp, args.train_fraction)
            else:
                scopes = {
                    "custom": build_windows(
                        source.files,
                        parse_date(args.start),
                        parse_date(args.end),
                    )
                }
            for scope, windows in scopes.items():
                for row in evaluate_scope(
                    source,
                    windows,
                    data_format,
                    spatial_scale,
                    args.max_windows,
                    legacy_model,
                    device,
                ):
                    all_rows.append({"scope": scope, **row})

        if not all_rows:
            raise RuntimeError("no evaluation windows were found")
        write_rows(Path(args.output), all_rows)
        print(f"wrote {len(all_rows)} rows to {args.output}")
    finally:
        source.close()


if __name__ == "__main__":
    main()
