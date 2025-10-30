#!/usr/bin/env python3
"""
ACT训练任务配置管理器
支持YAML配置文件、配置继承、自动路径推断
"""

import yaml
import os
import copy
from typing import Dict, Any, Optional, Union, List
from pathlib import Path


class TaskConfigManager:
    """任务配置管理器"""

    def __init__(self, config_root: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_root: 配置文件根目录，默认为当前脚本目录下的configs
        """
        if config_root is None:
            # 首先尝试当前工作目录下的configs
            cwd_configs = Path.cwd() / 'configs'
            if cwd_configs.exists():
                self.config_root = cwd_configs
            else:
                # 回退到脚本目录下的configs
                self.config_root = Path(__file__).parent / 'configs'
        else:
            self.config_root = Path(config_root)

        self.base_configs_dir = self.config_root / 'base'
        self.task_configs_dir = self.config_root / 'tasks'

        # 确保目录存在
        self.base_configs_dir.mkdir(parents=True, exist_ok=True)
        self.task_configs_dir.mkdir(parents=True, exist_ok=True)

    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载YAML文件

        Args:
            file_path: YAML文件路径

        Returns:
            Dict: 解析后的配置字典
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"YAML文件格式错误 {file_path}: {e}")

    def resolve_config_path(self, config_path: str) -> Path:
        """
        解析配置文件路径

        Args:
            config_path: 配置文件路径（可以是相对路径或绝对路径）

        Returns:
            Path: 解析后的绝对路径
        """
        path = Path(config_path)

        # 如果是绝对路径，直接使用
        if path.is_absolute():
            if path.exists():
                return path
            raise FileNotFoundError(f"找不到配置文件: {config_path}")

        # 候选路径列表，按优先级排序
        candidates = []

        # 1. 相对于config_root的路径
        candidates.append(self.config_root / config_path)

        # 2. 如果没有.yaml后缀，添加后缀再试
        if not config_path.endswith('.yaml'):
            candidates.append(self.config_root / (config_path + '.yaml'))

        # 3. 相对于当前工作目录的路径
        candidates.append(Path.cwd() / config_path)
        if not config_path.endswith('.yaml'):
            candidates.append(Path.cwd() / (config_path + '.yaml'))

        # 4. 如果config_path不包含"configs/"前缀，尝试添加
        if not config_path.startswith('configs/'):
            candidates.append(self.config_root.parent / config_path)
            if not config_path.endswith('.yaml'):
                candidates.append(self.config_root.parent / (config_path + '.yaml'))

        # 按顺序检查所有候选路径
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # 如果都找不到，抛出错误并显示尝试过的路径
        tried_paths = [str(c) for c in candidates]
        raise FileNotFoundError(f"找不到配置文件: {config_path}\n尝试过的路径: {tried_paths}")

    def load_config_with_inheritance(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置并处理继承关系

        Args:
            config_path: 配置文件路径

        Returns:
            Dict: 合并后的完整配置
        """
        # 解析配置文件路径
        full_config_path = self.resolve_config_path(config_path)

        # 加载配置
        config = self.load_yaml(full_config_path)

        # 处理继承
        if 'extends' in config:
            base_config_path = config['extends']

            # 递归加载基础配置
            base_config = self.load_config_with_inheritance(base_config_path)

            # 深度合并配置
            merged_config = self.deep_merge(base_config, config)

            # 移除extends字段
            if 'extends' in merged_config:
                del merged_config['extends']

            return merged_config

        return config

    def deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典

        Args:
            base: 基础字典
            override: 覆盖字典

        Returns:
            Dict: 合并后的字典
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def resolve_auto_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析自动路径推断

        Args:
            config: 配置字典

        Returns:
            Dict: 处理后的配置字典
        """
        config = copy.deepcopy(config)

        # 获取基础配置
        data_config = config.get('data', {})
        task_config = config.get('task', {})

        data_dir_base = data_config.get('data_dir_base', '/boot/common_data')
        data_dir_suffix = data_config.get('data_dir_suffix', '_hdf5')
        task_name = task_config.get('name', 'unknown_task')

        # 处理单个dataset_dir
        if 'dataset_dir' in data_config and data_config['dataset_dir'] == 'auto':
            auto_path = f"{data_dir_base}/{task_name}{data_dir_suffix}"
            data_config['dataset_dir'] = auto_path

        # 处理dataset_dirs列表中的auto
        if 'dataset_dirs' in data_config:
            dirs = data_config['dataset_dirs']
            for i, dir_path in enumerate(dirs):
                if dir_path == 'auto':
                    auto_path = f"{data_dir_base}/{task_name}{data_dir_suffix}"
                    dirs[i] = auto_path

        return config

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        验证配置文件的完整性和正确性

        Args:
            config: 配置字典

        Raises:
            ValueError: 配置验证失败时抛出异常
        """
        # 检查必需的顶级字段
        required_top_fields = ['task', 'data', 'training', 'robot']
        for field in required_top_fields:
            if field not in config:
                raise ValueError(f"缺少必需的配置字段: {field}")

        # 检查任务配置
        task_config = config['task']
        if 'name' not in task_config:
            raise ValueError("task.name 字段是必需的")

        # 检查数据配置
        data_config = config['data']
        required_data_fields = ['camera_names', 'episode_len']
        for field in required_data_fields:
            if field not in data_config:
                raise ValueError(f"缺少必需的数据配置字段: data.{field}")

        # 检查机器人配置
        robot_config = config['robot']
        required_robot_fields = ['name', 'state_dim']
        for field in required_robot_fields:
            if field not in robot_config:
                raise ValueError(f"缺少必需的机器人配置字段: robot.{field}")

        # 检查训练配置
        training_config = config['training']
        required_training_fields = ['policy_class', 'ckpt_dir']
        for field in required_training_fields:
            if field not in training_config:
                raise ValueError(f"缺少必需的训练配置字段: training.{field}")

    def convert_to_legacy_format(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将新格式配置转换为与现有代码兼容的格式

        Args:
            config: 新格式配置

        Returns:
            Dict: 兼容格式的配置
        """
        # 提取各部分配置
        task_config = config.get('task', {})
        data_config = config.get('data', {})
        robot_config = config.get('robot', {})
        training_config = config.get('training', {})

        # 构造task_config（兼容现有SIM_TASK_CONFIGS格式）
        legacy_task_config = {
            'dataset_dir': data_config.get('dataset_dir'),
            'num_episodes': data_config.get('num_episodes'),
            'episode_len': data_config.get('episode_len'),
            'camera_names': data_config.get('camera_names'),
            'state_dim': robot_config.get('state_dim'),
        }

        # 处理多数据集情况
        if 'dataset_dirs' in data_config:
            legacy_task_config['dataset_dir'] = data_config['dataset_dirs']

        # 辅助函数：转换字符串形式的科学记数法为浮点数
        def parse_numeric(value):
            if isinstance(value, str):
                try:
                    # 尝试转换为浮点数（支持科学记数法）
                    return float(value)
                except ValueError:
                    return value
            return value

        # 构造args字典（兼容现有参数格式）
        legacy_args = {
            'task_name': task_config.get('name'),
            'ckpt_dir': training_config.get('ckpt_dir'),
            'policy_class': training_config.get('policy_class', 'ACT'),
            'batch_size': training_config.get('batch_size', 32),
            'num_epochs': training_config.get('num_epochs', 2000),
            'lr': parse_numeric(training_config.get('lr', 3e-5)),
            'seed': training_config.get('seed', 0),
            'eval': False,  # 默认为训练模式
            'onscreen_render': training_config.get('onscreen_render', False),
            'temporal_agg': training_config.get('temporal_agg', False),
            # ACT特定参数
            'kl_weight': training_config.get('kl_weight', 10),
            'chunk_size': training_config.get('chunk_size', 100),
            'hidden_dim': training_config.get('hidden_dim', 512),
            'dim_feedforward': training_config.get('dim_feedforward', 3200),
            'dropout': training_config.get('dropout', 0.1),
            'weight_decay': parse_numeric(training_config.get('weight_decay', 1e-4)),
        }

        return {
            'task_config': legacy_task_config,
            'args': legacy_args,
            'config': config  # 保留原始完整配置
        }

    def load_task_config(self, config_path: str, eval_mode: bool = False) -> Dict[str, Any]:
        """
        加载任务配置的主入口函数

        Args:
            config_path: 配置文件路径
            eval_mode: 是否为评估模式

        Returns:
            Dict: 包含task_config和args的字典，与现有代码兼容
        """
        # 加载并处理继承
        config = self.load_config_with_inheritance(config_path)

        # 解析自动路径
        config = self.resolve_auto_paths(config)

        # 验证配置
        self.validate_config(config)

        # 转换为兼容格式
        legacy_format = self.convert_to_legacy_format(config)

        # 设置评估模式
        if eval_mode:
            legacy_format['args']['eval'] = True

        return legacy_format


def load_task_config(config_path: str, eval_mode: bool = False) -> Dict[str, Any]:
    """
    便捷函数：加载任务配置

    Args:
        config_path: 配置文件路径
        eval_mode: 是否为评估模式

    Returns:
        Dict: 包含task_config和args的字典
    """
    manager = TaskConfigManager()
    return manager.load_task_config(config_path, eval_mode)


# 示例使用
if __name__ == "__main__":
    # 创建配置管理器
    manager = TaskConfigManager()

    # 加载配置
    try:
        config = manager.load_task_config('tasks/example.yaml')
        print("配置加载成功:")
        print(f"任务名: {config['args']['task_name']}")
        print(f"数据目录: {config['task_config']['dataset_dir']}")
        print(f"相机: {config['task_config']['camera_names']}")
    except Exception as e:
        print(f"配置加载失败: {e}")
