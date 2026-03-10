import torch

from model import CascadeSRModel


def main():
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = CascadeSRModel().to(device)
    model.eval()

    # Dummy input:
    # CascadeSRModel expects 12-channel low-resolution input (see train.py / utils.py)
    # Shape: (batch_size, channels, height, width)
    batch_size = 1
    channels = 12
    height = 16
    width = 16

    x = torch.randn(batch_size, channels, height, width, device=device)

    with torch.no_grad():
        outputs = model(x)

    print("Forward pass successful.")
    if isinstance(outputs, (list, tuple)):
        for idx, out in enumerate(outputs):
            print(f"Output[{idx}] shape: {tuple(out.shape)}")
    else:
        print(f"Output shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()

