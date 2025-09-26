# ACT训练配置文件系统

## 目录结构

```
configs/
├── base/                  # 基础配置模板
│   ├── default.yaml      # 默认训练参数
│   ├── fr3.yaml          # FR3机器人配置
│   └── monte01.yaml      # Monte01双臂机器人配置
├── tasks/                 # 任务特定配置
│   ├── fr3_*.yaml        # FR3相关任务
│   ├── monte01_*.yaml    # Monte01相关任务
│   └── template.yaml     # 新任务模板
└── README.md             # 本文档
```

## 使用方法

### 1. 使用现有配置训练

```bash
# 使用配置文件训练
python imitate_episodes.py --config configs/tasks/fr3_bs_0916_50ep_ds.yaml

# 评估模式
python imitate_episodes.py --config configs/tasks/fr3_bs_0916_50ep_ds.yaml --eval
```

### 2. 创建新任务配置

1. 复制模板文件：
```bash
cp configs/tasks/template.yaml configs/tasks/my_new_task.yaml
```

2. 编辑配置文件，修改以下关键参数：
   - `task.name`: 任务名称
   - `data.dataset_dir`: 数据路径（可使用 `auto` 自动推断）
   - `training.ckpt_dir`: 检查点保存路径

3. 运行训练：
```bash
python imitate_episodes.py --config configs/tasks/my_new_task.yaml
```

### 3. 配置继承

配置文件支持继承机制，通过 `extends` 字段指定父配置：

```yaml
# 继承FR3基础配置
extends: base/fr3

# 覆盖特定参数
task:
  name: my_fr3_task

data:
  dataset_dir: auto  # 自动推断为 /boot/common_data/my_fr3_task_hdf5
```

### 4. 自动路径推断

设置 `dataset_dir: auto` 会自动根据任务名生成数据路径：
- 任务名：`fr3_new_experiment`
- 自动路径：`/boot/common_data/fr3_new_experiment_hdf5`

可在基础配置中修改 `data.data_dir_base` 和 `data.data_dir_suffix` 来改变默认路径格式。

### 5. 混合数据集训练

```yaml
data:
  dataset_dirs:
    - /boot/common_data/dataset1_hdf5
    - /boot/common_data/dataset2_hdf5
    - /boot/common_data/dataset3_hdf5
```

## 配置优先级

1. 子配置文件中的设置（最高优先级）
2. 父配置文件中的设置
3. 默认配置中的设置（最低优先级）

## 常用配置参数

### 任务配置 (task)
- `name`: 任务名称（必需）

### 数据配置 (data)
- `dataset_dir`: 单个数据集路径或 `auto`
- `dataset_dirs`: 多个数据集路径列表
- `num_episodes`: episode数量（可选，自动检测）
- `episode_len`: episode长度
- `camera_names`: 相机名称列表

### 机器人配置 (robot)
- `name`: 机器人类型 (fr3, monte01)
- `state_dim`: 状态维度

### 训练配置 (training)
- `ckpt_dir`: 检查点保存目录（必需）
- `policy_class`: 策略类型（通常为ACT）
- `batch_size`: 批大小
- `num_epochs`: 训练轮数
- `lr`: 学习率
- `kl_weight`: KL散度权重（ACT特有）
- `chunk_size`: 动作块大小（ACT特有）

## 示例配置

查看 `configs/tasks/` 目录下的示例配置文件，了解不同任务的配置方法。