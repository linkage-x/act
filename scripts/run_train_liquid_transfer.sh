#!/bin/bash
cd ~/code/act
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 使用配置文件训练 FR3 Liquid Transfer 任务
CUDA_VISIBLE_DEVICES=7 python3 imitate_episodes.py \
  --config configs/tasks/fr3_liquid_transfer_0920_50ep_ds.yaml