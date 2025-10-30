#!/bin/bash
cd ~/code/act
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 使用配置文件的简化训练脚本
CUDA_VISIBLE_DEVICES=7 python3 imitate_episodes.py \
  --config configs/tasks/fr3_it_mix_1030_3dmouse_customcfg3.yaml


