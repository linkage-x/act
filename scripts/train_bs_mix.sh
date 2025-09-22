#!/bin/bash
cd ~/code/act
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CUDA_VISIBLE_DEVICES=6 python3 imitate_episodes.py \
--task_name fr3_bs_mix_ds_0924 \
--ckpt_dir ckpts/fr3_bs_mix_ds_0924 \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 512 --dim_feedforward 3200 \
--num_epochs 8000  --lr 0.5e-5 \
--seed 0
