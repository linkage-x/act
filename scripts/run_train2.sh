#!/bin/bash
cd ~/code/act
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES=5 python3 imitate_episodes.py \
  --config configs/tasks/fr3_ip_1022_49ep.yaml
