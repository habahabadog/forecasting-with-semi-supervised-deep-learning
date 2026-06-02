# Data

The full art1km archive used in the manuscript is restricted by the original data provider and is not redistributed in this repository.

Expected full-data convention for the 2022 experiments:

- Hourly `.npy` files named `YYYYMMDDHH.npy`.
- Four variables in channel order: `t2m`, `u10`, `v10`, `precipitation`.
- Native daily `.npz` tensors recorded in the experiment notebook have shape `(4, 24, 1031, 1321)`.
- Hourly target tensors are cropped to `(4, 1030, 1320)`.
- Coarse inputs are generated from target tensors using stride-5 sampling.

A small derived or synthetic demonstration subset should be added here before submission if data-permission constraints allow.
