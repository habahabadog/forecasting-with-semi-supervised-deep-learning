#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python train.py \
  --data-dir "${DATA_DIR:-../art/artnpy}" \
  --steps-per-hour "${STEPS_PER_HOUR:-3}" \
  --spatial-scale "${SPATIAL_SCALE:-5}" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --device "${DEVICE:-auto}"
