## Project Overview

This repository contains a PyTorch implementation of a cascade super-resolution model for spatiotemporal data.  
The core idea is to reconstruct high-resolution fields from low-resolution inputs using a Residual Channel Attention Network (RCAN) backbone and a cascade architecture.

## Main Files

- **`common.py`**  
  Low-level building blocks for convolutional neural networks, including:
  - `default_conv` factory for 2D convolutions
  - `MeanShift` for simple normalization / denormalization
  - `BasicBlock`, `ResBlock`, and `Upsampler` modules used by the main models

- **`model.py`**  
  Defines the main neural network architectures:
  - Channel attention modules (`CALayer`, `AverageChannelAttention`)
  - Residual Channel Attention Block (`RCAB`) and `ResidualGroup`
  - `RCAN` and `RCANtime` super-resolution networks
  - `CascadeSRModel`, which applies `RCANtime` and `RCAN` in a cascade to produce multi-step super-resolved outputs

- **`utils.py`**  
  Data utilities and configuration:
  - Global seeding via `setup_seed`
  - Dataset definition `MyDataset` that loads `.npy` files from `../art/artnpy`, normalizes them, and builds stacked sequences
  - Construction of `train_dataset`, `test_dataset`, `new_dataset` and their corresponding `DataLoader`s (`train_load`, `test_load`, `new_load`)

- **`train.py`**  
  Training script for `CascadeSRModel`:
  - Builds model and L1 loss on GPU
  - Uses Adam optimizer with `ReduceLROnPlateau` scheduler
  - Runs training / validation loops and logs to TensorBoard (`logs_train`)
  - Saves model weights for the two base models into the `model/` directory each epoch

- **`run.sh`**  
  Convenience shell script (Linux/macOS) that:
  - Sets `CUDA_VISIBLE_DEVICES`
  - Sets `PYTORCH_CUDA_ALLOC_CONF` for CUDA memory behavior
  - Launches `python train.py`

## Requirements

- Python 3.8+ (recommended)
- PyTorch (GPU build recommended)
- NumPy
- TensorBoard (for log visualization)

Install typical dependencies with:

```bash
pip install torch torchvision tensorboard numpy
```

You also need preprocessed `.npy` files placed in `../art/artnpy` as expected by `utils.py`.

## How to Train

### Option 1: Direct Python (need data)

```bash
python train.py
```

### Option 2: Shell Script (need data)

```bash
bash run.sh
```

You can edit `run.sh` to change the GPU index or other environment variables.

## Quick Test (Sanity Check)

To quickly check that the model code and environment are set up correctly, you can run a forward pass with random data:

1. Make sure you have installed the required Python packages.
2. Run the following command in this folder:

```bash
python quick_test.py
```

This script will:

- Instantiate `CascadeSRModel`
- Move it to CUDA if available (otherwise CPU)
- Create a random input tensor with the expected shape
- Run a forward pass and print the output tensor shapes

If the script finishes without errors and prints shapes, the basic model wiring and environment are working.

