"""Train FusionCast on preprocessed hourly weather-field arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from model import FusionCast
from utils import build_dataloaders, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../art/artnpy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variables", type=int, default=4)
    parser.add_argument("--steps-per-hour", type=int, default=3)
    parser.add_argument("--spatial-scale", type=int, default=5)
    parser.add_argument("--n-resgroups", type=int, default=2)
    parser.add_argument("--n-resblocks", type=int, default=2)
    parser.add_argument("--n-feats", type=int, default=96)
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs_train")
    parser.add_argument("--split-timestamp", default="20221000")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def split_subhour_frames(tensor, variables):
    if tensor.shape[1] % variables != 0:
        raise ValueError("temporal output channels must be divisible by variables")
    return list(torch.split(tensor, variables, dim=1))


def temporal_consistency_loss(interval, left_anchor, right_anchor, criterion, variables):
    frames = split_subhour_frames(interval, variables)
    steps = len(frames)
    loss = interval.new_tensor(0.0)
    for idx, frame in enumerate(frames, start=1):
        left_weight = 1.0 - idx / (steps + 1)
        right_weight = idx / (steps + 1)
        loss = loss + left_weight * criterion(frame, left_anchor)
        loss = loss + right_weight * criterion(frame, right_anchor)
    return loss


def fusioncast_loss(outputs, low_sequence, low_anchor, high_anchor, criterion, variables):
    first_interval, second_interval = outputs[0], outputs[1]
    anchor_candidates = outputs[2:-1]
    high_res_anchor = outputs[-1]

    left_hour = low_sequence[:, :variables]
    middle_hour = low_sequence[:, variables : 2 * variables]
    right_hour = low_sequence[:, 2 * variables : 3 * variables]

    high_res_loss = criterion(high_res_anchor, high_anchor)
    anchor_loss = sum(criterion(candidate, low_anchor) for candidate in anchor_candidates)
    consistency_loss = temporal_consistency_loss(
        first_interval, left_hour, middle_hour, criterion, variables
    )
    consistency_loss = consistency_loss + temporal_consistency_loss(
        second_interval, middle_hour, right_hour, criterion, variables
    )
    total_loss = high_res_loss + anchor_loss + consistency_loss
    metrics = {
        "high_res": float(high_res_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
        "consistency": float(consistency_loss.detach().cpu()),
    }
    return total_loss, metrics


def run_epoch(model, loader, criterion, device, variables, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for low_sequence, low_anchor, high_anchor, _ in loader:
        low_sequence = low_sequence.to(device)
        low_anchor = low_anchor.to(device)
        high_anchor = high_anchor.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(low_sequence)
            loss, _ = fusioncast_loss(
                outputs, low_sequence, low_anchor, high_anchor, criterion, variables
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def evaluate_high_res_mae(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for low_sequence, _, high_anchor, _ in loader:
            low_sequence = low_sequence.to(device)
            high_anchor = high_anchor.to(device)
            high_res_anchor = model(low_sequence)[-1]
            total_loss += criterion(high_res_anchor, high_anchor).item()
    return total_loss / max(1, len(loader))


def save_checkpoint(model, checkpoint_dir, epoch, args, val_loss):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "val_loss": val_loss,
    }
    torch.save(payload, checkpoint_dir / f"fusioncast_epoch_{epoch:03d}.pth")
    torch.save(
        model.temporal_model.state_dict(),
        checkpoint_dir / f"fusioncast_temporal_epoch_{epoch:03d}.pth",
    )
    torch.save(
        model.spatial_model.state_dict(),
        checkpoint_dir / f"fusioncast_spatial_epoch_{epoch:03d}.pth",
    )


class NullSummaryWriter:
    """Drop-in writer used when TensorBoard is not installed."""

    def add_scalar(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        pass


def build_summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed; scalar logging is disabled.")
        return NullSummaryWriter()
    return SummaryWriter(log_dir)


def build_model(args):
    return FusionCast(
        variables=args.variables,
        steps_per_hour=args.steps_per_hour,
        spatial_scale=args.spatial_scale,
        n_resgroups=args.n_resgroups,
        n_resblocks=args.n_resblocks,
        n_feats=args.n_feats,
        reduction=args.reduction,
    )


def dry_run(args, device):
    model = build_model(args).to(device)
    criterion = nn.L1Loss().to(device)
    low_size = 8
    high_size = low_size * args.spatial_scale
    low_sequence = torch.randn(2, 3 * args.variables, low_size, low_size, device=device)
    low_anchor = torch.randn(2, args.variables, low_size, low_size, device=device)
    high_anchor = torch.randn(2, args.variables, high_size, high_size, device=device)
    outputs = model(low_sequence)
    loss, metrics = fusioncast_loss(
        outputs, low_sequence, low_anchor, high_anchor, criterion, args.variables
    )
    loss.backward()
    print(f"dry-run loss: {loss.item():.6f}")
    print(f"loss components: {metrics}")
    print(f"output shapes: {[tuple(output.shape) for output in outputs]}")


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.dry_run:
        dry_run(args, device)
        return

    data = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spatial_stride=args.spatial_scale,
        split_timestamp=args.split_timestamp,
    )
    model = build_model(args).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=10,
        min_lr=1e-8,
        eps=1e-8,
    )
    writer = build_summary_writer(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, data.train_loader, criterion, device, args.variables, optimizer
        )
        val_loss = run_epoch(model, data.val_loader, criterion, device, args.variables)
        test_mae = evaluate_high_res_mae(model, data.test_loader, criterion, device)
        scheduler.step(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("mae/test_high_res", test_mae, epoch)
        save_checkpoint(model, args.checkpoint_dir, epoch, args, val_loss)

        print(
            f"epoch {epoch:03d}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, test_mae={test_mae:.6f}"
        )

    writer.close()


if __name__ == "__main__":
    main()
