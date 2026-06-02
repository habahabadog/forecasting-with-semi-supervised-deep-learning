from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CHANNEL_NAMES = ("t2m", "u10", "v10", "precipitation")
DEFAULT_MEAN = np.array([295.71266308, -1.35061465, -0.47545855, 24.42778042], dtype=np.float32)
DEFAULT_STD = np.array([4.25024591, 2.50847865, 2.44562078, 67.33232656], dtype=np.float32)


@dataclass(frozen=True)
class SplitConfig:
    train_validation_cutoff: str = "20221000"
    train_fraction: float = 0.9


def list_hourly_files(data_dir: str | Path) -> list[str]:
    data_path = Path(data_dir)
    return sorted(path.name for path in data_path.glob("*.npy"))


def split_files(files: list[str], config: SplitConfig = SplitConfig()) -> tuple[list[str], list[str], list[str]]:
    train_pool = sorted(name for name in files if name < config.train_validation_cutoff)
    test_files = sorted(name for name in files if name > config.train_validation_cutoff)
    train_size = int(len(train_pool) * config.train_fraction)
    return train_pool[:train_size], train_pool[train_size:], test_files


def consecutive_triplets(files: list[str]) -> list[list[str]]:
    file_times = [datetime.strptime(name[:-4], "%Y%m%d%H") for name in files]
    return [
        [files[i], files[i + 1], files[i + 2]]
        for i in range(len(file_times) - 2)
        if file_times[i + 1] == file_times[i] + timedelta(hours=1)
        and file_times[i + 2] == file_times[i] + timedelta(hours=2)
    ]


class Art1kmTripletDataset(Dataset):
    """Three-hour 2022 art1km samples using the four-variable tensor convention."""

    def __init__(
        self,
        data_dir: str | Path,
        file_list: list[str],
        mean: np.ndarray = DEFAULT_MEAN,
        std: np.ndarray = DEFAULT_STD,
        spatial_stride: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.mean = mean.reshape(4, 1, 1).astype(np.float32)
        self.std = std.reshape(4, 1, 1).astype(np.float32)
        self.spatial_stride = spatial_stride
        self.stacked_list = consecutive_triplets(file_list)

    def __len__(self) -> int:
        return len(self.stacked_list)

    def __getitem__(self, index: int):
        low_frames = []
        high_frames = []
        for file_name in self.stacked_list[index]:
            data = np.load(self.data_dir / file_name).astype(np.float32)
            low = (data[:, :-1:self.spatial_stride, :-1:self.spatial_stride] - self.mean) / self.std
            high = (data[:, :-1, :-1] - self.mean) / self.std
            low_frames.append(torch.from_numpy(low).float())
            high_frames.append(torch.from_numpy(high).float())

        low_stack = torch.stack(low_frames, dim=0).reshape(-1, *low_frames[0].shape[1:])
        high_stack = torch.stack(high_frames, dim=0).reshape(-1, *high_frames[0].shape[1:])
        return low_stack, low_stack[4:8], high_stack[4:8], high_stack
