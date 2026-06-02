# FusionCast

PyTorch reference implementation for **FusionCast**, a semi-supervised
spatiotemporal downscaling method for regional NWP-like weather guidance.

FusionCast ingests three consecutive hourly coarse-resolution fields
`[T0, T60, T120]`, estimates sub-hourly low-resolution states between the
hourly anchors, and applies spatial super-resolution to produce kilometre-scale
weather fields. The public implementation supports the cadences used in the
manuscript:

| Cadence | `--steps-per-hour` | Intermediate states per hour |
| --- | ---: | ---: |
| 30 minutes | 1 | 1 |
| 15 minutes | 3 | 3 |
| 10 minutes | 5 | 5 |

The default variable order follows the manuscript experiments:

1. 2 m temperature
2. 10 m U wind
3. 10 m V wind
4. precipitation

## Repository Contents

- `model.py` defines `FusionCast`, the temporal RCAN operator, the spatial RCAN
  operator, deployment inference through `predict_subhourly()`, and the
  backward-compatible `CascadeSRModel` alias.
- `utils.py` contains deterministic setup, hourly sequence construction, and
  PyTorch dataloaders for preprocessed `.npy` fields.
- `train.py` trains FusionCast with the anchor-supervised temporal loss and the
  high-resolution spatial loss used by the manuscript.
- `quick_test.py` runs no-data forward checks for 30-, 15-, and 10-minute
  settings.
- `legacy_15min.py` loads the original split 15-minute checkpoint files from
  the pre-publication experiment code.
- `run.sh` is a Linux/macOS convenience launcher for training.
- `configs/` contains example JSON configurations for the three cadences.
- `checkpoints/README.md` documents the expected pretrained-weight layout.
- `data/demo/` contains two small pre-stacked `.npy` tensors for inference
  smoke tests.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if GPU training is
needed. CPU is sufficient for `quick_test.py`.

## Quick Test

Run a shape-only smoke test without data:

```bash
python quick_test.py
```

Expected behavior: the script instantiates small FusionCast models for
`steps_per_hour` values 1, 3, and 5, performs forward passes on random tensors,
and prints output shapes.

To test the manuscript-scale hidden width:

```bash
python quick_test.py --paper-config
```

To verify loading of the included demonstration `.npy` tensors:

```bash
python quick_test.py --demo-data-dir data/demo
```

On some Windows scientific Python stacks, PyTorch may fail before execution
with a duplicate OpenMP runtime error. Fix the Python environment if possible;
for a local smoke test only, this workaround can be used:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python quick_test.py
```

## Data Format

Training expects hourly `.npy` files named with UTC+8 local timestamps, for
example:

```text
2022100209.npy
2022100210.npy
2022100211.npy
```

Each file should store a `float32` array with shape:

```text
[variables, height, width]
```

The loader builds consecutive three-hour windows and downsamples the native
high-resolution grid by `--spatial-scale` to create coarse inputs. The default
normalization statistics in `utils.py` match the manuscript experiments.

## Split Protocol

By default, `train.py` follows the manuscript split implemented in
`utils.split_files`: files before `--split-timestamp 20221000` are sorted
chronologically and split 8:2 into training and validation subsets. The first
80% of those pre-cutoff files are used for training, and the remaining 20% are
used for validation. Files after the split timestamp are used for testing. For
the 2022 hourly archive used in the manuscript, this corresponds to January
through September split 8:2 for training/validation, with October through
December held out as the independent test period.

The full meteorological archive used in the paper is not redistributed in this
repository because the authors do not control its public release rights.

The included `data/demo/` files are small pre-stacked inference tensors with
shape `[3 * variables, height, width] = [12, 61, 121]`. They are provided only
for smoke testing `FusionCast.predict_subhourly()` and are not a substitute for
the hourly training archive expected by `train.py`.

## Training

For the default 15-minute setting:

```bash
python train.py --data-dir /path/to/artnpy --steps-per-hour 3
```

For the 10- and 30-minute settings:

```bash
python train.py --data-dir /path/to/artnpy --steps-per-hour 5
python train.py --data-dir /path/to/artnpy --steps-per-hour 1
```

Useful options:

```bash
python train.py \
  --data-dir /path/to/artnpy \
  --epochs 100 \
  --batch-size 16 \
  --device auto \
  --split-timestamp 20221000 \
  --checkpoint-dir checkpoints
```

Training saves:

- combined model checkpoints: `checkpoints/fusioncast_epoch_XXX.pth`
- temporal operator weights: `checkpoints/fusioncast_temporal_epoch_XXX.pth`
- spatial operator weights: `checkpoints/fusioncast_spatial_epoch_XXX.pth`

Use `python train.py --dry-run` to verify the training loss and backward pass
without loading data.

## Pretrained Weights

This repository includes the original split 15-minute checkpoint files from the
pre-publication experiment code:

```text
checkpoints/pretrained/fusioncast_15min_temporal_legacy.pth
checkpoints/pretrained/fusioncast_15min_spatial_legacy.pth
```

They can be loaded with:

```bash
python legacy_15min.py
```

These legacy files correspond to the original `base_model_500.pth` and
`base_model2_500.pth` temporal/spatial modules. They are kept for provenance;
new checkpoints produced by `train.py` use the cleaned public API and are saved
under `checkpoints/fusioncast_epoch_XXX.pth`.

To reproduce every manuscript figure without retraining, the following
additional artifacts should still be provided as GitHub Releases, Git LFS files,
or another archived asset:

- 10-minute and 30-minute weights used for the multi-cadence comparison
- the normalization statistics used at training time
- metadata describing the train/validation/test file split

See `checkpoints/README.md` for a suggested file naming convention.

## Manuscript Alignment

This repository implements the software components described in the manuscript:

- temporal interpolation from hourly coarse fields
- anchor-based partial supervision at the central hourly frame
- spatial super-resolution from 5 km to 1 km grids
- cadence-flexible generation for 10-, 15-, and 30-minute products
- deployment-style high-resolution sub-hourly sequence generation through
  `FusionCast.predict_subhourly()`

The repository does not contain the full restricted meteorological dataset or
the trained weights unless they are added separately.

## Citation

If you use this software, please cite the associated manuscript listed in
`CITATION.cff`.
