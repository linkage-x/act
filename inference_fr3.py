#!/usr/bin/env python3

import torch
import numpy as np
import pickle
import os
import argparse
from policy import ACTPolicy
from constants import SIM_TASK_CONFIGS
import h5py

class FR3ACTInference:
    def __init__(self, ckpt_dir, task_name='fr3_peg_in_hole_extended'):
        self.task_name = task_name
        self.ckpt_dir = ckpt_dir
        self.task_config = SIM_TASK_CONFIGS[task_name]
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载数据集统计信息
        self.load_dataset_stats()
        
        # 初始化策略
        self.init_policy()
        
        # 加载最佳模型
        self.load_best_policy()
        
    def load_dataset_stats(self):
        """加载数据集统计信息用于归一化"""
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        
        # 如果没有找到dataset_stats.pkl，尝试找dataset_stats_bac.pkl
        if not os.path.exists(stats_path):
            stats_path = os.path.join(self.ckpt_dir, 'dataset_stats_bac.pkl')
            
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"找不到数据集统计文件: {stats_path}")
            
        with open(stats_path, 'rb') as f:
            self.dataset_stats = pickle.load(f)
            
        print(f"✅ 加载数据集统计信息: {stats_path}")
        print(f"   - 统计信息键: {list(self.dataset_stats.keys())}")
        
    def init_policy(self):
        """初始化ACT策略"""
        state_dim = self.task_config['state_dim']
        
        # ACT策略参数 (与训练时保持一致)
        policy_config = {
            'lr': 1e-5,
            'num_queries': 100,  # chunk_size
            'kl_weight': 10,
            'hidden_dim': 512,
            'dim_feedforward': 3200,
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': self.task_config['camera_names'],
            'vq': False,
            'vq_class': None,
            'vq_dim': None,
            'action_dim': state_dim,
            'no_encoder': False,
        }
        
        self.policy = ACTPolicy(policy_config)
        self.policy.to(self.device)
        self.policy.eval()
        
        print(f"✅ ACT策略初始化完成")
        print(f"   - State维度: {state_dim}")
        print(f"   - Action块大小: {policy_config['num_queries']}")
        print(f"   - 相机: {policy_config['camera_names']}")
        
    def load_best_policy(self):
        """加载最佳训练模型"""
        policy_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        
        # 如果没有找到policy_best.ckpt，尝试找policy_best_bac.ckpt
        if not os.path.exists(policy_path):
            policy_path = os.path.join(self.ckpt_dir, 'policy_best_bac.ckpt')
            
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"找不到策略文件: {policy_path}")
            
        checkpoint = torch.load(policy_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint)
        
        print(f"✅ 加载最佳策略模型: {policy_path}")
        
    def normalize_data(self, data, stats, key):
        """数据归一化"""
        mean = stats[f'{key}_mean']
        std = stats[f'{key}_std']
        return (data - mean) / std
        
    def denormalize_data(self, data, stats, key):
        """数据反归一化"""
        mean = stats[f'{key}_mean']
        std = stats[f'{key}_std']
        return data * std + mean
        
    def preprocess_images(self, images_dict):
        """预处理图像数据"""
        processed_images = {}
        
        for cam_name, image in images_dict.items():
            if cam_name not in self.task_config['camera_names']:
                continue
                
            # 确保图像格式正确 (H, W, C) -> (C, H, W)
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
                
            # 归一化到 [0, 1]
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
                
            # 转换为tensor并添加batch维度
            image_tensor = torch.from_numpy(image).float().to(self.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
                
            processed_images[cam_name] = image_tensor
            
        return processed_images
        
    def predict(self, qpos, images_dict):
        """
        进行动作预测
        
        Args:
            qpos: 当前关节位置 (8维度)
            images_dict: 相机图像字典 {'ee_cam': img, 'third_person_cam': img}
            
        Returns:
            predicted_actions: 预测的动作序列 (chunk_size, 8)
        """
        with torch.no_grad():
            # 预处理输入数据
            qpos = np.array(qpos)
            if qpos.shape[0] != self.task_config['state_dim']:
                raise ValueError(f"QPos维度不匹配: 期望{self.task_config['state_dim']}, 得到{qpos.shape[0]}")
                
            # 归一化qpos
            qpos_normalized = self.normalize_data(qpos, self.dataset_stats, 'qpos')
            qpos_tensor = torch.from_numpy(qpos_normalized).float().to(self.device)
            qpos_tensor = qpos_tensor.unsqueeze(0)  # 添加batch维度
            
            # 预处理图像
            image_tensors = self.preprocess_images(images_dict)
            
            # 构建输入数据
            if len(image_tensors) != len(self.task_config['camera_names']):
                raise ValueError(f"相机数量不匹配: 期望{self.task_config['camera_names']}, 得到{list(image_tensors.keys())}")
                
            # 策略推理
            actions = self.policy(qpos_tensor, image_tensors)
            
            # 反归一化动作
            actions_np = actions.cpu().numpy().squeeze(0)  # 移除batch维度
            actions_denorm = self.denormalize_data(actions_np, self.dataset_stats, 'action')
            
            return actions_denorm
            
    def predict_from_episode(self, episode_path, timestep=0):
        """
        从episode文件中读取数据并进行预测
        
        Args:
            episode_path: HDF5 episode文件路径
            timestep: 要预测的时间步
            
        Returns:
            predicted_actions: 预测的动作序列
            ground_truth: 真实动作 (用于对比)
        """
        with h5py.File(episode_path, 'r') as f:
            # 读取数据
            qpos = f['observations/qpos'][timestep]
            ee_cam = f['observations/images/ee_cam'][timestep]
            third_person_cam = f['observations/images/third_person_cam'][timestep]
            
            # 真实动作 (用于对比)
            if timestep < f['action'].shape[0]:
                ground_truth = f['action'][timestep]
            else:
                ground_truth = None
                
        # 构建图像字典
        images_dict = {
            'ee_cam': ee_cam,
            'third_person_cam': third_person_cam
        }
        
        # 进行预测
        predicted_actions = self.predict(qpos, images_dict)
        
        return predicted_actions, ground_truth, qpos
        
    def evaluate_episode(self, episode_path, num_steps=10):
        """
        评估整个episode的预测性能
        
        Args:
            episode_path: HDF5 episode文件路径
            num_steps: 要评估的步数
        """
        print(f"\n📊 评估Episode: {episode_path}")
        
        mse_errors = []
        
        with h5py.File(episode_path, 'r') as f:
            episode_len = min(f['action'].shape[0], num_steps)
            
        for t in range(episode_len):
            pred_actions, gt_action, qpos = self.predict_from_episode(episode_path, t)
            
            if gt_action is not None:
                # 计算MSE误差 (只比较第一个动作，因为chunk_size=100)
                mse = np.mean((pred_actions[0] - gt_action) ** 2)
                mse_errors.append(mse)
                
                if t % 5 == 0:  # 每5步打印一次
                    print(f"  步骤 {t:3d}: MSE={mse:.6f}")
                    print(f"    预测: {pred_actions[0]}")
                    print(f"    真实: {gt_action}")
                    
        avg_mse = np.mean(mse_errors)
        print(f"\n📈 平均MSE误差: {avg_mse:.6f}")
        return avg_mse

def main():
    parser = argparse.ArgumentParser(description='FR3 ACT模型推理')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                       help='检查点目录路径')
    parser.add_argument('--episode_path', type=str, default=None,
                       help='测试episode的HDF5文件路径')
    parser.add_argument('--task_name', type=str, default='fr3_peg_in_hole_extended',
                       help='任务名称')
    parser.add_argument('--num_steps', type=int, default=20,
                       help='评估的步数')
    parser.add_argument('--timestep', type=int, default=0,
                       help='单步预测的时间步')
    
    args = parser.parse_args()
    
    try:
        # 初始化推理器
        print("🚀 初始化FR3 ACT推理器...")
        inferencer = FR3ACTInference(args.ckpt_dir, args.task_name)
        
        if args.episode_path:
            if os.path.exists(args.episode_path):
                # 评估指定episode
                inferencer.evaluate_episode(args.episode_path, args.num_steps)
            else:
                print(f"❌ Episode文件不存在: {args.episode_path}")
        else:
            # 如果没有指定episode，尝试找一个测试文件
            dataset_dir = inferencer.task_config['dataset_dir']
            test_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
            
            if test_files:
                test_file = os.path.join(dataset_dir, test_files[0])
                print(f"🎯 使用测试文件: {test_file}")
                inferencer.evaluate_episode(test_file, args.num_steps)
            else:
                print("❌ 没有找到测试文件")
                
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()