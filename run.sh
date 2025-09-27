#!/bin/bash

# 设置GPU编号
export CUDA_VISIBLE_DEVICES=7

# 设置PyTorch显存碎片优化参数
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 执行训练脚本
python train.py
