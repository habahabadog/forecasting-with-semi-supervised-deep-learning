"""Dataset and reproducibility utilities for FusionCast."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_MEAN = np.array(
    [295.71266308, -1.35061465, -0.47545855, 24.42778042],
    dtype=np.float32,
).reshape(4, 1, 1)
DEFAULT_STD = np.array(
    [4.25024591, 2.50847865, 2.44562078, 67.33232656],
    dtype=np.float32,
).reshape(4, 1, 1)


def setup_seed(seed):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_hourly_timestamp(file_name):
    """Parse file names such as 2022100209.npy into hourly datetimes."""
    return datetime.strptime(Path(file_name).stem, "%Y%m%d%H")


def list_npy_files(data_dir):
    """Return sorted .npy file names from a directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data directory not found: {data_path}. "
            "Provide --data-dir with preprocessed hourly .npy files."
        )

    files = sorted(path.name for path in data_path.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no .npy files found in {data_path}")
    return files


def consecutive_three_hour_windows(file_list):
    """Create [T0, T60, T120] windows from consecutive hourly files."""
    file_times = [parse_hourly_timestamp(file_name) for file_name in file_list]
    windows = []
    for idx in range(len(file_times) - 2):
        if (
            file_times[idx + 1] == file_times[idx] + timedelta(hours=1)
            and file_times[idx + 2] == file_times[idx] + timedelta(hours=2)
        ):
            windows.append(
                [file_list[idx], file_list[idx + 1], file_list[idx + 2]]
            )
    return windows


class WeatherSequenceDataset(Dataset):
    """Load three-hour weather-field sequences from hourly .npy files.

    Each file is expected to contain an array with shape [variables, H, W].
    The default four channels are 2 m temperature, 10 m U wind, 10 m V wind,
    and precipitation, matching the manuscript experiments.
    """

    def __init__(
        self,
        data_dir,
        file_list,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        spatial_stride=5,
    ):
        self.data_dir = Path(data_dir)
        self.file_list = list(file_list)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.spatial_stride = int(spatial_stride)
        self.stacked_list = consecutive_three_hour_windows(self.file_list)

        if not self.stacked_list:
            raise ValueError("no consecutive three-hour windows found in file_list")

    def __len__(self):
        return len(self.stacked_list)

    def _load_field(self, file_name):
        field = np.load(self.data_dir / file_name).astype(np.float32)
        if field.ndim != 3:
            raise ValueError(f"{file_name} must have shape [variables, H, W]")
        return field

    def __getitem__(self, index):
        low_frames = []
        high_frames = []

        for file_name in self.stacked_list[index]:
            field = self._load_field(file_name)
            normalized = (field - self.mean) / self.std
            low_frame = normalized[:, :-1 : self.spatial_stride, :-1 : self.spatial_stride]
            high_frame = normalized[:, :-1, :-1]
            low_frames.append(torch.from_numpy(low_frame).float())
            high_frames.append(torch.from_numpy(high_frame).float())

        low_sequence = torch.stack(low_frames, dim=0).reshape(
            -1, *low_frames[0].shape[1:]
        )
        high_sequence = torch.stack(high_frames, dim=0).reshape(
            -1, *high_frames[0].shape[1:]
        )

        variables = low_frames[0].shape[0]
        low_anchor = low_sequence[variables : 2 * variables]
        high_anchor = high_sequence[variables : 2 * variables]
        return low_sequence, low_anchor, high_anchor, high_sequence


MyDataset = WeatherSequenceDataset


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_dataset: WeatherSequenceDataset
    val_dataset: WeatherSequenceDataset
    test_dataset: WeatherSequenceDataset


def split_files(file_list, split_timestamp="20221000", train_fraction=0.8):
    train_candidates = sorted(
        file_name for file_name in file_list if file_name < f"{split_timestamp}.npy"
    )
    test_files = sorted(
        file_name for file_name in file_list if file_name > f"{split_timestamp}.npy"
    )

    train_size = max(3, int(len(train_candidates) * train_fraction))
    train_files = train_candidates[:train_size]
    val_files = train_candidates[train_size:]
    if len(val_files) < 3:
        val_files = train_candidates[-3:]
    if len(test_files) < 3:
        test_files = val_files
    return train_files, val_files, test_files


def build_dataloaders(
    data_dir,
    batch_size=16,
    num_workers=0,
    spatial_stride=5,
    split_timestamp="20221000",
):
    files = list_npy_files(data_dir)
    train_files, val_files, test_files = split_files(files, split_timestamp)

    train_dataset = WeatherSequenceDataset(data_dir, train_files, spatial_stride=spatial_stride)
    val_dataset = WeatherSequenceDataset(data_dir, val_files, spatial_stride=spatial_stride)
    test_dataset = WeatherSequenceDataset(data_dir, test_files, spatial_stride=spatial_stride)
    pin_memory = torch.cuda.is_available()

    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
