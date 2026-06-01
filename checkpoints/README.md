# Checkpoint Layout

Training writes checkpoints to this directory by default. The files are not
committed because trained weights are usually too large for normal Git history.

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
