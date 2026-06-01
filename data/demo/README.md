# Demo Data

This directory contains two small `.npy` tensors for repository smoke tests:

```text
2019_01_01_01.npy
2019_01_01_02.npy
```

Each file stores a pre-stacked model input tensor with shape:

```text
[3 * variables, height, width] = [12, 61, 121]
```

The channel axis follows the FusionCast inference input convention:

```text
[T0 variables, T60 variables, T120 variables]
```

with the default four-variable order documented in the repository README.
These files are not the full meteorological archive and are not formatted as
hourly training files for `train.py`. They are provided only to verify that the
public model can load real `.npy` tensors and run deployment-style inference:

```bash
python quick_test.py --demo-data-dir data/demo
```
