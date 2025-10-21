#!/usr/bin/env python3

import torch
import numpy as np
import pickle
import os
from abc import ABC, abstractmethod


class PolicyInference(ABC):
    """通用推理算法基类，抽象出推理算法接口"""
    
    def __init__(self, ckpt_dir, policy_config):
        """
        初始化推理器
        
        Args:
            ckpt_dir: 检查点目录路径
            policy_config: 策略配置字典
        """
        self.ckpt_dir = ckpt_dir
        self.policy_config = policy_config
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载数据集统计信息
        self.load_dataset_stats()
        
        # 初始化策略
        self.init_policy()
        
        # 加载模型
        self.load_policy()
        
    def load_dataset_stats(self):
        """加载数据集统计信息用于归一化"""
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"找不到数据集统计文件: {stats_path}")
            
        with open(stats_path, 'rb') as f:
            self.dataset_stats = pickle.load(f)
            
        print(f"✅ 加载数据集统计信息: {stats_path}")
        print(f"   - 统计信息键: {list(self.dataset_stats.keys())}")
        
    @abstractmethod
    def init_policy(self):
        """初始化策略模型（抽象方法，子类需要实现）"""
        pass
        
    @abstractmethod
    def load_policy(self):
        """加载训练好的策略模型（抽象方法，子类需要实现）"""
        pass
        
    @abstractmethod
    def forward_policy(self, state, images):
        """
        策略前向推理（抽象方法，子类需要实现）
        
        Args:
            state: 状态张量
            images: 图像张量
            
        Returns:
            actions: 预测的动作
        """
        pass
        
    def normalize_data(self, data, stats, key):
        """数据归一化
        支持 stats 中为 torch.Tensor 或 numpy.ndarray，两者与 data 类型对齐
        """
        mean = stats[f'{key}_mean']
        std = stats[f'{key}_std']
        # Align types with input data (numpy path)
        if isinstance(data, np.ndarray):
            if hasattr(mean, 'detach'):
                mean = mean.detach().cpu().numpy()
            if hasattr(std, 'detach'):
                std = std.detach().cpu().numpy()
            return (data - mean) / std
        # Torch path
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std)
        return (data - mean) / std
        
    def denormalize_data(self, data, stats, key):
        """数据反归一化
        支持 stats 中为 torch.Tensor 或 numpy.ndarray，两者与 data 类型对齐
        """
        mean = stats[f'{key}_mean']
        std = stats[f'{key}_std']
        # Align types with input data
        if isinstance(data, np.ndarray):
            if hasattr(mean, 'detach'):
                mean = mean.detach().cpu().numpy()
            if hasattr(std, 'detach'):
                std = std.detach().cpu().numpy()
            return data * std + mean
        # Torch path
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.as_tensor(std)
        return data * std + mean
        
    def predict(self, state, images):
        """
        进行动作预测

        Args:
            state: 状态向量
            images: 图像数据

        Returns:
            predicted_actions: 预测的动作序列
        """
        with torch.no_grad():
            # 预处理状态
            state = np.array(state)

            # 🔥 关键修复: 根据控制模式选择正确的归一化键
            # 检查是否为EE pose控制模式
            if self.dataset_stats.get('has_ee_pose', False):
                # EE控制模式: 使用ee_pose统计信息
                state_key = 'ee_pose'
                action_key = 'ee_action'
                print(f"🎯 Using EE pose control mode for normalization")
            else:
                # 关节控制模式: 使用qpos统计信息
                state_key = 'qpos'
                action_key = 'action'
                print(f"🎯 Using joint control mode for normalization")

            state_normalized = self.normalize_data(state, self.dataset_stats, state_key)
            state_tensor = torch.from_numpy(state_normalized).float().to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # 添加batch维度

            # 策略推理
            actions = self.forward_policy(state_tensor, images)

            # 反归一化动作 - 使用对应的action key
            actions_np = actions.cpu().numpy().squeeze(0)  # 移除batch维度
            actions_denorm = self.denormalize_data(actions_np, self.dataset_stats, action_key)

            return actions_denorm


class ACTInference(PolicyInference):
    """ACT算法特定的推理器"""
    
    def __init__(self, ckpt_dir, state_dim, camera_names, **kwargs):
        """
        初始化ACT推理器
        
        Args:
            ckpt_dir: 检查点目录路径
            state_dim: 状态维度
            camera_names: 相机名称列表
            **kwargs: 其他ACT参数
        """
        # ACT策略配置 - 确保数值参数为正确类型
        policy_config = {
            'lr': float(kwargs.get('lr', 1e-5)),  # 确保 lr 为浮点数
            'num_queries': int(kwargs.get('num_queries', 100)),  # chunk_size
            'kl_weight': int(kwargs.get('kl_weight', 10)),
            'hidden_dim': int(kwargs.get('hidden_dim', 512)),
            'dim_feedforward': int(kwargs.get('dim_feedforward', 3200)),
            'lr_backbone': float(kwargs.get('lr_backbone', 1e-5)),  # 确保 lr_backbone 为浮点数
            'backbone': str(kwargs.get('backbone', 'resnet18')),
            'enc_layers': int(kwargs.get('enc_layers', 4)),
            'dec_layers': int(kwargs.get('dec_layers', 7)),
            'nheads': int(kwargs.get('nheads', 8)),
            'camera_names': camera_names,
            'vq': bool(kwargs.get('vq', False)),
            'vq_class': kwargs.get('vq_class', None),
            'vq_dim': kwargs.get('vq_dim', None) if kwargs.get('vq_dim') is None else int(kwargs.get('vq_dim')),
            'action_dim': int(state_dim),
            'no_encoder': bool(kwargs.get('no_encoder', False)),
        }
        
        super().__init__(ckpt_dir, policy_config)
        
    def init_policy(self):
        """初始化ACT策略"""
        from .policy import ACTPolicy
        
        self.policy = ACTPolicy(self.policy_config)
        self.policy.to(self.device)
        self.policy.eval()
        
        print(f"✅ ACT策略初始化完成")
        print(f"   - State维度: {self.policy_config['action_dim']}")
        print(f"   - Action块大小: {self.policy_config['num_queries']}")
        print(f"   - 相机: {self.policy_config['camera_names']}")
        
    def load_policy(self):
        """加载ACT策略模型"""
        policy_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"找不到策略文件: {policy_path}")
            
        checkpoint = torch.load(policy_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint)
        
        print(f"✅ 加载ACT策略模型: {policy_path}")
        
    def forward_policy(self, state, images):
        """ACT策略前向推理"""
        return self.policy(state, images)
