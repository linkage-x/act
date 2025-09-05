# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of ACT (Action Chunking with Transformers) for robot learning. The codebase supports both simulation environments and real robot control, particularly for the Franka Research 3 (FR3) robot.

## Key Commands

### Data Preparation
```bash
# Edit constants.py first to set data paths, then:
python convert_fr3_robust.py
```

### Training
```bash
# Quick train using shell script
sh ./run_train.sh

# Manual training with custom parameters
python3 imitate_episodes.py \
  --task_name <task_name> \
  --ckpt_dir <checkpoint_dir> \
  --policy_class ACT --kl_weight 10 --chunk_size 100 \
  --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
  --num_epochs 2000 --lr 3e-5 --seed 0
```

### Evaluation
```bash
# Add --eval flag to training command
python3 imitate_episodes.py --task_name <task_name> --ckpt_dir <ckpt_dir> --eval

# Enable temporal ensembling
python3 imitate_episodes.py --task_name <task_name> --ckpt_dir <ckpt_dir> --eval --temporal_agg
```

### Testing
```bash
# Run inference tests
python3 test_inference.py

# Visualize episodes from dataset
python3 visualize_episodes.py --dataset_dir <data_dir> --episode_idx 0
```

## Architecture

### Core Components
- **imitate_episodes.py**: Main training/evaluation script for ACT policy
- **policy.py**: ACT policy wrapper and interface
- **detr/**: Modified DETR transformer architecture for action prediction
- **utils.py**: Data loading, normalization, and helper functions
- **constants.py**: Task configurations and robot parameters

### Data Processing
- **convert_fr3_robust.py**: Converts raw robot data to HDF5 format
- Episodes stored as HDF5 with structure: images (camera views), qpos (joint positions), actions

### Supported Tasks
- Simulated: `sim_transfer_cube_scripted`, `sim_insertion_scripted`
- Real FR3: `fr3_peg_in_hole`, `fr3_pickup_kiwi`

### Key Parameters
- **chunk_size**: Number of future actions to predict (typically 100)
- **kl_weight**: Weight for KL divergence loss in VAE (typically 10)
- **hidden_dim**: Transformer hidden dimension (typically 512)
- **dim_feedforward**: Feed-forward network dimension (typically 3200)

## Important Notes

- FR3 robot has 8 DOF: 7 arm joints + 1 gripper
- Camera inputs: `ee_cam` (end-effector) and `third_person_cam`
- For real-world tasks, train for 5000+ epochs or 3-4x after loss plateaus
- Success improves even after loss plateaus - be patient with training
- Episode lengths vary by task (400-2000 timesteps)