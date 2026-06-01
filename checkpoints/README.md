# Checkpoint Layout

Training writes checkpoints to this directory by default. Large checkpoint sets
should normally be uploaded through GitHub Releases or Git LFS instead of normal
Git history.

The repository includes two small legacy 15-minute split checkpoints:

```text
pretrained/fusioncast_15min_temporal_legacy.pth
pretrained/fusioncast_15min_spatial_legacy.pth
```

They correspond to the original files `base_model_500.pth` and
`base_model2_500.pth` from the pre-publication experiment code. Load them with
`python legacy_15min.py`.

Suggested names for manuscript artifacts:

```text
fusioncast_15min.pth
fusioncast_15min_temporal.pth
fusioncast_15min_spatial.pth
fusioncast_10min.pth
fusioncast_10min_temporal.pth
fusioncast_10min_spatial.pth
fusioncast_30min.pth
fusioncast_30min_temporal.pth
fusioncast_30min_spatial.pth
normalization_stats.json
split_metadata.json
```

For public reproducibility, upload these through GitHub Releases or Git LFS and
link them from the repository README.
