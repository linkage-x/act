#!/bin/bash
cd ~/code/act
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES=7 python3 imitate_episodes.py \
--task_name fr3_bs_0916_50ep_ds \
--ckpt_dir ckpts/fr3_bs_0916_50ep_ds \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 512 --dim_feedforward 3200 \
--num_epochs 8000  --lr 1e-4 \
--seed 0
