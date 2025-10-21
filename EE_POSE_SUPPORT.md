# End-Effector Pose Support for ACT

本文档说明如何使用 ACT 训练和推理 end-effector (EE) pose 控制策略。

## 概述

ACT 现在支持两种控制模式：
1. **Joint Position Control** (默认): 使用关节位置作为观测和动作
2. **EE Pose Control**: 使用末端执行器位姿 (position + quaternion) 作为观测和动作

## 数据转换

### 1. 准备数据

确保你的原始数据包含 `ee_states` 字段，格式如下：

```json
{
  "data": [
    {
      "joint_states": {
        "single": {
          "position": [j1, j2, j3, j4, j5, j6, j7],
          "velocity": [...]
        }
      },
      "ee_states": {
        "single": {
          "pose": [x, y, z, qx, qy, qz, qw]  // 位置 + 四元数
        }
      },
      "tools": {
        "single": {
          "position": gripper_value
        }
      },
      ...
    }
  ]
}
```

### 2. 转换数据为 HDF5 格式

使用 `--store_ee_pose` 标志来存储 EE pose 数据：

```bash
python scripts/convert_fr3_preprocess.py \
  --input_dir /path/to/raw/episodes \
  --output_dir /path/to/output/hdf5 \
  --store_ee_pose \
  --downsample_rate 2 \
  --segment_mode none \
  --target_length 500
```

这会在 HDF5 文件中创建以下数据集：
- `/observations/qpos`: 关节位置 (用于 joint control)
- `/observations/ee_pose`: EE 位姿 (用于 EE pose control)
- `/action`: 关节位置动作
- `/ee_action`: EE 位姿动作

## 训练

### 1. 配置任务

在 `constants.py` 或配置文件中添加 `control_mode` 参数：

```python
# constants.py 示例
SIM_TASK_CONFIGS = {
    "fr3_peg_in_hole_ee_pose": {
        "dataset_dir": "/path/to/hdf5/data",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["ee_cam", "third_person_cam"],
        "state_dim": 8,  # 对于 EE pose: [x, y, z, qx, qy, qz, qw, gripper]
        "control_mode": "ee_pose",  # 设置为 'ee_pose' 或 'joint'
    },
}
```

### 2. 运行训练

```bash
python imitate_episodes.py \
  --config configs/tasks/your_task_config.yaml \
  --task_name fr3_peg_in_hole_ee_pose \
  --ckpt_dir checkpoints/ee_pose_exp \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 32 \
  --num_epochs 2000 \
  --lr 1e-5
```

训练时会自动：
- 检测数据集中是否包含 EE pose 数据
- 根据 `control_mode` 选择使用 joint positions 或 EE poses
- 使用相应的归一化统计信息

## 推理

推理时，`factory/components/gym_interface.py` 中的 `step` 函数已经处理了不同观测类型的转换：

```python
# gym_interface.py 已经实现了以下逻辑：
if observation_type == ObservationType.END_EFFECTOR_POSE:
    # 使用 EE pose 作为观测
    observation = ee_pose
elif observation_type == ObservationType.JOINT_POSITION_ONLY:
    # 使用 joint positions 作为观测
    observation = joint_positions
```

因此，你只需要在推理配置中指定正确的 `observation_type` 和 `action_type`。

## 注意事项

### EE Pose 格式

- **观测/动作格式**: `[x, y, z, qx, qy, qz, qw, gripper]` (8 维)
  - `x, y, z`: 位置 (米)
  - `qx, qy, qz, qw`: 四元数方向
  - `gripper`: 夹爪开合度

### 状态维度

- **Joint control**: `state_dim = n_joints + 1` (FR3: 7 关节 + 1 夹爪 = 8)
- **EE pose control**: `state_dim = 8` ([x,y,z,qx,qy,qz,qw,gripper])

### 归一化

系统会自动为每种控制模式计算和应用独立的归一化统计：
- Joint mode: 使用 `qpos_mean`, `qpos_std`, `action_mean`, `action_std`
- EE pose mode: 使用 `ee_pose_mean`, `ee_pose_std`, `ee_action_mean`, `ee_action_std`

### 兼容性

- 如果数据集不包含 EE pose 数据，但指定了 `control_mode='ee_pose'`，系统会自动回退到 joint control
- 现有的 joint control 数据集完全兼容，无需修改

## 示例工作流

### 完整的 EE Pose 控制流程

```bash
# 1. 转换数据（存储 EE pose）
python scripts/convert_fr3_preprocess.py \
  --input_dir /data/raw/peg_in_hole \
  --output_dir /data/processed/peg_in_hole_hdf5 \
  --store_ee_pose \
  --downsample_rate 2

# 2. 训练 (在 constants.py 中设置 control_mode='ee_pose')
python imitate_episodes.py \
  --task_name fr3_peg_in_hole_ee_pose \
  --ckpt_dir checkpoints/ee_pose_model \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --num_epochs 2000

# 3. 推理（使用 factory 的推理系统，配置相应的 observation_type）
# 在推理配置中设置：
# observation_type: END_EFFECTOR_POSE
# action_type: END_EFFECTOR_POSE
```

## 故障排除

### 1. "WARNING: EE pose control mode requested but no EE pose data found"

**原因**: 数据集不包含 EE pose 数据

**解决方案**:
- 确保原始数据包含 `ee_states` 字段
- 使用 `--store_ee_pose` 标志重新转换数据

### 2. State dimension mismatch

**原因**: `state_dim` 配置与实际数据不匹配

**解决方案**:
- Joint control: 设置 `state_dim = n_joints + n_grippers`
- EE pose control: 设置 `state_dim = 8`

### 3. 训练时 loss 不收敛

**可能原因**:
- EE pose 的归一化范围可能与 joint positions 不同
- 需要调整学习率或其他超参数

**建议**:
- 检查归一化统计信息是否合理
- 尝试调整学习率 (EE pose 可能需要更小的学习率)
- 检查数据质量

## 技术细节

### 数据存储结构 (HDF5)

```
episode_0.hdf5
├── /observations
│   ├── /qpos          # [T, n_joints+gripper] - Joint positions
│   ├── /qvel          # [T, n_joints+gripper] - Joint velocities
│   ├── /ee_pose       # [T, 8] - EE pose (新增)
│   └── /images
│       ├── /ee_cam    # [T, H, W, 3]
│       └── /third_person_cam
├── /action            # [T, n_joints+gripper] - Joint actions
├── /ee_action         # [T, 8] - EE pose actions (新增)
└── attrs
    ├── sim: False
    ├── episode_length: T
    └── has_ee_pose: True (新增)
```

### 代码修改位置

1. **scripts/convert_fr3_preprocess.py**: 数据转换，存储 EE pose
2. **utils.py**:
   - `EpisodicDataset`: 加载和处理 EE pose 数据
   - `get_norm_stats`: 计算 EE pose 归一化统计
   - `load_data`: 支持 control_mode 参数
3. **imitate_episodes.py**:
   - 从配置读取 `control_mode`
   - 自动调整 `state_dim`
4. **constants.py**: 任务配置中添加 `control_mode` 字段

## 参考

- lerobot/reader.py: 参考了 ActionType 和 ObservationType 的定义
- factory/components/gym_interface.py: 推理时的观测类型处理
