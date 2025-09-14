#!/bin/bash

#python3 imitate_episodes.py \
#--task_name fr3_block_stacking_0915_55ep \
#--ckpt_dir ckpts/fr3_block_stacking_0915_55ep \
#--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
#--num_epochs 5000  --lr 5e-5 \
#--seed 0
cd ~/code/act
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 imitate_episodes.py \
--task_name fr3_mixed_training_example \
--ckpt_dir ckpts/fr3_blockstacking_0918_mix129eps \
--policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 256 --dim_feedforward 3200 \
--num_epochs 8000  --lr 1e-4 \
--seed 0
