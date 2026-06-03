from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path("../art/artnpy")
TRAIN_VALIDATION_CUTOFF = "20221000"
TRAIN_FRACTION = 0.8
BATCH_SIZE = 4

CHANNEL_MEAN = np.array(
    [295.71266308, -1.35061465, -0.47545855, 24.42778042],
    dtype=np.float32,
).reshape(4, 1, 1)
CHANNEL_STD = np.array(
    [4.25024591, 2.50847865, 2.44562078, 67.33232656],
    dtype=np.float32,
).reshape(4, 1, 1)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def consecutive_triplets(file_list: list[str]) -> list[list[str]]:
    file_times = [datetime.strptime(file_name[:-4], "%Y%m%d%H") for file_name in file_list]
    return [
        [file_list[index], file_list[index + 1], file_list[index + 2]]
        for index in range(len(file_times) - 2)
        if file_times[index + 1] == file_times[index] + timedelta(hours=1)
        and file_times[index + 2] == file_times[index] + timedelta(hours=2)
    ]


class Art1kmSnapshotDataset(Dataset):
    def __init__(self, file_dir: Path, file_list: list[str], mean: np.ndarray, std: np.ndarray) -> None:
        self.file_dir = Path(file_dir)
        self.file_list = file_list
        self.mean = mean
        self.std = std
        self.stacked_list = consecutive_triplets(file_list)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low_frames = []
        high_frames = []

        for file_name in self.stacked_list[index]:
            data = np.load(self.file_dir / file_name).astype(np.float32)
            low_data = (data[:, :-1:5, :-1:5] - self.mean) / self.std
            high_data = (data[:, :-1, :-1] - self.mean) / self.std

            low_frames.append(torch.from_numpy(low_data).float())
            high_frames.append(torch.from_numpy(high_data).float())

        low_stack = torch.stack(low_frames, dim=0).reshape(-1, *low_frames[0].shape[1:])
        high_stack = torch.stack(high_frames, dim=0).reshape(-1, *high_frames[0].shape[1:])
        return low_stack, low_stack[4:8], high_stack[4:8], high_stack

    def __len__(self) -> int:
        return len(self.stacked_list)


def split_files(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    train_validation_files = sorted(file_name for file_name in files if file_name < TRAIN_VALIDATION_CUTOFF)
    held_out_files = sorted(file_name for file_name in files if file_name > TRAIN_VALIDATION_CUTOFF)
    train_size = int(len(train_validation_files) * TRAIN_FRACTION)
    return (
        train_validation_files[:train_size],
        train_validation_files[train_size:],
        held_out_files,
    )


def build_loaders(data_dir: Path = DATA_DIR):
    common_files = sorted(path.name for path in Path(data_dir).glob("*.npy"))
    train_files, validation_files, held_out_files = split_files(common_files)

    train_dataset = Art1kmSnapshotDataset(data_dir, train_files, CHANNEL_MEAN, CHANNEL_STD)
    validation_dataset = Art1kmSnapshotDataset(data_dir, validation_files, CHANNEL_MEAN, CHANNEL_STD)
    held_out_dataset = Art1kmSnapshotDataset(data_dir, held_out_files, CHANNEL_MEAN, CHANNEL_STD)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    held_out_loader = DataLoader(held_out_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return train_dataset, validation_dataset, held_out_dataset, train_loader, validation_loader, held_out_loader


setup_seed(42)
train_dataset, test_dataset, new_dataset, train_load, test_load, new_load = build_loaders()
batch_size = BATCH_SIZE
