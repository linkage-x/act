# Monte01 双臂机器人支持

本文档说明如何使用更新后的ACT代码库来支持Monte01双臂机器人。

## 主要更新

### 1. 机器人配置系统 (`robot_config.py`)
- 解耦机器人特定配置
- 支持FR3 (8 DOF) 和 Monte01 (16 DOF)
- 自动检测机器人类型

### 2. 数据转换 (`convert_fr3_robust.py`)
- 自动检测单臂/双臂配置
- 支持不同相机命名 (`left_ee_cam`, `right_ee_cam`)
- 处理Monte01夹爪范围 (0-0.074m)

### 3. 训练支持 (`constants.py`, `imitate_episodes.py`)
- 添加Monte01任务配置
- 支持16 DOF状态空间
- 三相机输入支持

### 4. 统一推理系统 (`robot_inference.py`, `policy_inference.py`)
- 机器人无关的推理框架
- 自动创建合适的推理器
- 双臂动作分离/组合功能

## 使用方法

### 数据转换
```bash
# 自动检测机器人类型并转换
python convert_fr3_robust.py \
  --input_dir /path/to/raw_data \
  --output_dir /path/to/hdf5_data
```

### 训练Monte01
```bash
# 使用便利脚本
python train_monte01.py train

# 或直接使用训练脚本
python imitate_episodes.py \
  --task_name monte01_peg_in_hole \
  --ckpt_dir ckpts/monte01_peg_in_hole \
  --policy_class ACT --kl_weight 10 --chunk_size 100 \
  --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 \
  --num_epochs 3000 --lr 3e-5 --seed 0
```

### 推理测试
```bash
# 自动检测机器人类型
python robot_inference.py \
  --ckpt_dir ckpts/monte01_peg_in_hole \
  --episode /path/to/test_episode.hdf5

# 或使用便利脚本
python train_monte01.py test
```

### 代码中使用
```python
from robot_inference import create_robot_inference

# 自动创建合适的推理器
robot_inference = create_robot_inference(
    ckpt_dir='ckpts/monte01_peg_in_hole',
    # task_name 和 robot_type 会自动检测
)

# 16 DOF 状态输入
qpos = np.array([...])  # 16维

# 三相机输入
images = {
    'ee_cam': left_ee_image,
    'right_ee_cam': right_ee_image, 
    'third_person_cam': third_person_image
}

# 推理
actions = robot_inference.predict(qpos, images)

# 分离双臂动作
left_actions, right_actions = robot_inference.split_prediction(actions[0])
```

## 配置文件更新

### constants.py
添加了Monte01任务配置：
- `monte01_peg_in_hole`: 16 DOF, 三相机
- `monte01_bimanual_insertion`: 16 DOF, 三相机

### 自动检测规则
- 以 `monte01_` 开头的任务 → Monte01双臂
- 以 `fr3_` 开头的任务 → FR3单臂  
- 检查点中action维度16 → Monte01双臂
- 检查点中action维度8 → FR3单臂

## 注意事项

1. **批大小**: Monte01由于状态空间更大，可能需要减小batch_size
2. **训练时间**: 双臂任务通常需要更多训练轮次 (3000+)
3. **相机支持**: 缺失的相机会自动用零图像填充
4. **夹爪范围**: Monte01使用0-0.074m，FR3使用0-0.08m

## 测试
```bash
# 转换数据测试
python train_monte01.py convert

# 推理测试  
python train_monte01.py test

# 训练测试
python train_monte01.py train

# 评估测试
python train_monte01.py eval
```