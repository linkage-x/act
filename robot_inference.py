#!/usr/bin/env python3

import torch
import numpy as np
import h5py
import os
from abc import ABC, abstractmethod
from policy_inference import PolicyInference


class RobotInference(ABC):
    """机器人推理抽象类，定义机器人特定的接口"""
    
    def __init__(self, policy_inference, task_config):
        """
        初始化机器人推理器
        
        Args:
            policy_inference: 策略推理器实例 (ACTInference, PPOInference等)
            task_config: 任务配置字典
        """
        self.policy_inference = policy_inference
        self.task_config = task_config
        
    @abstractmethod
    def preprocess_images(self, images_dict):
        """
        预处理图像数据（抽象方法，子类需要实现）
        
        Args:
            images_dict: 原始图像字典
            
        Returns:
            processed_images: 预处理后的图像字典
        """
        pass
        
    @abstractmethod
    def validate_input_data(self, state, images_dict):
        """
        验证输入数据格式（抽象方法，子类需要实现）
        
        Args:
            state: 机器人状态
            images_dict: 图像字典
            
        Raises:
            ValueError: 当输入数据格式不正确时
        """
        pass
        
    def predict(self, state, images_dict):
        """
        进行动作预测
        
        Args:
            state: 机器人状态
            images_dict: 相机图像字典
            
        Returns:
            predicted_actions: 预测的动作序列
        """
        # 验证输入数据
        self.validate_input_data(state, images_dict)
        
        # 预处理图像
        processed_images = self.preprocess_images(images_dict)
        
        # 调用策略推理
        return self.policy_inference.predict(state, processed_images)


class FR3Inference(RobotInference):
    """FR3机器人特定的推理器"""
    
    def __init__(self, policy_inference, task_name='fr3_peg_in_hole_extended'):
        """
        初始化FR3推理器
        
        Args:
            policy_inference: 策略推理器实例
            task_name: 任务名称
        """
        from constants import SIM_TASK_CONFIGS
        
        self.task_name = task_name
        
        if task_name not in SIM_TASK_CONFIGS:
            raise ValueError(f"未知任务名称: {task_name}")
            
        task_config = SIM_TASK_CONFIGS[task_name]
        super().__init__(policy_inference, task_config)
        
    def preprocess_images(self, images_dict):
        """预处理FR3图像数据"""
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
            image_tensor = torch.from_numpy(image).float().to(self.policy_inference.device)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
                
            processed_images[cam_name] = image_tensor
            
        return processed_images
        
    def validate_input_data(self, qpos, images_dict):
        """验证FR3输入数据格式"""
        # 验证qpos维度
        qpos = np.array(qpos)
        if qpos.shape[0] != self.task_config['state_dim']:
            raise ValueError(f"QPos维度不匹配: 期望{self.task_config['state_dim']}, 得到{qpos.shape[0]}")
            
        # 验证相机数量
        if len(images_dict) != len(self.task_config['camera_names']):
            raise ValueError(f"相机数量不匹配: 期望{self.task_config['camera_names']}, 得到{list(images_dict.keys())}")
            
        # 验证相机名称
        for cam_name in images_dict.keys():
            if cam_name not in self.task_config['camera_names']:
                raise ValueError(f"未知相机名称: {cam_name}")
                
    def predict_from_episode(self, episode_path, timestep=0):
        """
        从episode文件中读取数据并进行预测
        
        Args:
            episode_path: HDF5 episode文件路径
            timestep: 要预测的时间步
            
        Returns:
            predicted_actions: 预测的动作序列
            ground_truth: 真实动作 (用于对比)
            qpos: 关节位置
        """
        with h5py.File(episode_path, 'r') as f:
            # 读取数据
            qpos = f['observations/qpos'][timestep]
            
            # 构建图像字典 - 使用配置中的相机名称
            images_dict = {}
            for cam_name in self.task_config['camera_names']:
                if f'observations/images/{cam_name}' in f:
                    images_dict[cam_name] = f[f'observations/images/{cam_name}'][timestep]
                else:
                    raise KeyError(f"相机数据不存在: observations/images/{cam_name}")
            
            # 真实动作 (用于对比)
            if timestep < f['action'].shape[0]:
                ground_truth = f['action'][timestep]
            else:
                ground_truth = None
                
        # 进行预测
        predicted_actions = self.predict(qpos, images_dict)
        
        return predicted_actions, ground_truth, qpos
        
    def evaluate_episode(self, episode_path, num_steps=10):
        """
        评估整个episode的预测性能
        
        Args:
            episode_path: HDF5 episode文件路径
            num_steps: 要评估的步数
            
        Returns:
            avg_mse: 平均MSE误差
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


class RealtimeController:
    """通用实时控制器"""
    
    def __init__(self, robot_inference):
        """
        初始化实时控制器
        
        Args:
            robot_inference: 机器人推理器实例
        """
        self.robot_inference = robot_inference
        self.action_buffer = None  # 动作缓冲区
        self.buffer_index = 0      # 当前动作索引
        
        print("🤖 实时控制器初始化完成")
        
    def get_robot_state(self):
        """
        获取机器人当前状态 (需要根据实际机器人接口实现)
        
        Returns:
            state: 机器人状态
        """
        raise NotImplementedError("需要子类实现get_robot_state方法")
        
    def get_camera_images(self):
        """
        获取相机图像 (需要根据实际机器人接口实现)
        
        Returns:
            images_dict: 图像字典
        """
        raise NotImplementedError("需要子类实现get_camera_images方法")
        
    def send_action_to_robot(self, action):
        """
        发送动作到机器人 (需要根据实际机器人接口实现)
        
        Args:
            action: 动作向量
        """
        raise NotImplementedError("需要子类实现send_action_to_robot方法")
        
    def step(self):
        """执行一步控制循环"""
        # 如果动作缓冲区为空或已用完，重新预测
        if self.action_buffer is None or self.buffer_index >= self.action_buffer.shape[0]:
            state = self.get_robot_state()
            images = self.get_camera_images()
            
            # 预测动作序列
            self.action_buffer = self.robot_inference.predict(state, images)
            self.buffer_index = 0
            
            print(f"🔄 重新预测动作序列，长度: {self.action_buffer.shape[0]}")
            
        # 执行当前动作
        current_action = self.action_buffer[self.buffer_index]
        self.send_action_to_robot(current_action)
        
        self.buffer_index += 1
        
        return current_action
        
    def run_realtime_control(self, max_steps=1000, freq_hz=10):
        """
        运行实时控制循环
        
        Args:
            max_steps: 最大步数
            freq_hz: 控制频率 (Hz)
        """
        import time
        
        dt = 1.0 / freq_hz
        
        print(f"🚀 开始实时控制，频率: {freq_hz}Hz")
        
        for step in range(max_steps):
            start_time = time.time()
            
            # 执行一步控制
            action = self.step()
            
            # 控制频率
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
            if step % 50 == 0:
                print(f"步骤 {step}: 动作 = {action}")
                
        print("✅ 实时控制完成")


class FR3RealtimeController(RealtimeController):
    """FR3特定的实时控制器"""
    
    def get_robot_state(self):
        """获取FR3机器人状态"""
        # TODO: 实现实际的FR3状态读取
        # 这里返回模拟数据
        qpos = np.array([0.0, -0.5, 0.3, 0.0, -0.2, 0.1, 0.0, 0.05])
        return qpos
        
    def get_camera_images(self):
        """获取FR3相机图像"""
        # TODO: 实现实际的FR3相机图像获取
        # 这里返回模拟数据
        ee_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        third_person_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        
        return {
            'ee_cam': ee_img,
            'third_person_cam': third_person_img
        }
        
    def send_action_to_robot(self, action):
        """发送动作到FR3机器人"""
        # TODO: 实现实际的FR3机器人控制
        print(f"发送动作到FR3机器人: {action}")