from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fusioncast import CascadeSRModel
from dataset_2022_snapshot import (
    batch_size,
    new_dataset,
    new_load,
    test_dataset,
    test_load,
    train_dataset,
    train_load,
)


CHANNELS_PER_FRAME = 4
TIMEPOINTS = 11
EPOCHS = 100
CHECKPOINT_DIR = Path("model")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def weighted_anchor_loss(prediction, anchor, criterion, weights):
    loss = 0.0
    for index, weight in enumerate(weights):
        start = index * CHANNELS_PER_FRAME
        end = start + CHANNELS_PER_FRAME
        loss = loss + criterion(prediction[:, start:end], anchor) * weight
    return loss


def interpolation_anchor_loss(prediction, low_frames, criterion):
    ascending = [(index + 1) / TIMEPOINTS for index in range(TIMEPOINTS)]
    descending = [(TIMEPOINTS - index) / TIMEPOINTS for index in range(TIMEPOINTS)]

    first_anchor = low_frames[:, 0:4]
    middle_anchor = low_frames[:, 4:8]
    final_anchor = low_frames[:, 8:12]

    return (
        weighted_anchor_loss(prediction, first_anchor, criterion, descending)
        + weighted_anchor_loss(prediction, middle_anchor, criterion, ascending)
        + weighted_anchor_loss(prediction, final_anchor, criterion, ascending)
        + weighted_anchor_loss(prediction, final_anchor, criterion, descending)
    )


def mean_dataset_loss(total_loss: float, dataset) -> float:
    if len(dataset) == 0:
        return 0.0
    return total_loss / (len(dataset) / batch_size)


def train_one_epoch(model, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0

    for low_da, low2, high_da, _ in train_load:
        low_da = low_da.to(DEVICE)
        low2 = low2.to(DEVICE)
        high_da = high_da.to(DEVICE)

        optimizer.zero_grad()
        x1, _x2, outputs, mid = model(low_da)

        spatial_loss = criterion(outputs, high_da)
        temporal_loss = sum(criterion(frame, low2) for frame in mid)
        anchor_loss = interpolation_anchor_loss(x1, low_da, criterion)
        loss = spatial_loss + temporal_loss + anchor_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return mean_dataset_loss(total_loss, train_dataset)


def evaluate_validation(model, criterion) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for low_da, low2, high_da, _ in test_load:
            low_da = low_da.to(DEVICE)
            low2 = low2.to(DEVICE)
            high_da = high_da.to(DEVICE)

            _x1, _x2, outputs, mid = model(low_da)
            spatial_loss = criterion(outputs, high_da)
            temporal_loss = sum(criterion(frame, low2) for frame in mid)
            total_loss += (spatial_loss + temporal_loss).item()

    return mean_dataset_loss(total_loss, test_dataset)


def evaluate_held_out(model, criterion) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for low_da, _low2, high_da, _ in new_load:
            low_da = low_da.to(DEVICE)
            high_da = high_da.to(DEVICE)

            _x1, _x2, outputs, _mid = model(low_da)
            total_loss += criterion(outputs, high_da).item()

    return mean_dataset_loss(total_loss, new_dataset)


def save_checkpoints(model, epoch_index: int) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(model.base_model.state_dict(), CHECKPOINT_DIR / f"base_model_{epoch_index}.pth")
    torch.save(model.base_model2.state_dict(), CHECKPOINT_DIR / f"base_model2_{epoch_index}.pth")


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = CascadeSRModel().to(DEVICE)
    criterion = nn.L1Loss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=10,
        min_lr=1e-8,
        eps=1e-8,
    )

    writer = SummaryWriter("logs_train")
    for epoch_index in range(1, EPOCHS + 1):
        print(f"epoch {epoch_index} started")

        train_loss = train_one_epoch(model, criterion, optimizer)
        writer.add_scalar("train_loss", train_loss, epoch_index)
        print(f"train loss: {train_loss:.6f}")

        validation_loss = evaluate_validation(model, criterion)
        scheduler.step(validation_loss)
        writer.add_scalar("validation_loss", validation_loss, epoch_index)
        print(f"validation loss: {validation_loss:.6f}")

        save_checkpoints(model, epoch_index)
        print("checkpoints saved")

        held_out_loss = evaluate_held_out(model, criterion)
        writer.add_scalar("held_out_loss", held_out_loss, epoch_index)
        print(f"held-out test loss: {held_out_loss:.6f}")

    writer.close()


if __name__ == "__main__":
    main()
