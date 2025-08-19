#!/bin/bash

python3 imitate_episodes.py \
--task_name fr3_peg_in_hole \
--ckpt_dir ckpts/fr3_peg_in_hole \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
--num_epochs 2000  --lr 1e-5 \
--seed 0