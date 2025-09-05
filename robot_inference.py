#!/usr/bin/env python3

import torch
import numpy as np
import h5py
import os
from abc import ABC, abstractmethod
from policy_inference import PolicyInference
from robot_config import RobotConfig, get_robot_config, detect_robot_from_task, detect_robot_from_checkpoint


class RobotInference(ABC):
    """机器人推理抽象类，定义机器人特定的接口"""
    
    def __init__(self, policy_inference, task_config, robot_config):
        """
        初始化机器人推理器
        
        Args:
            policy_inference: 策略推理器实例 (ACTInference, PPOInference等)
            task_config: 任务配置字典
            robot_config: 机器人配置 (RobotConfig实例)
        """
        self.policy_inference = policy_inference
        self.task_config = task_config
        self.robot_config = robot_config
        
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
        from robot_config import get_robot_config
        
        self.task_name = task_name
        
        if task_name not in SIM_TASK_CONFIGS:
            raise ValueError(f"未知任务名称: {task_name}")
            
        task_config = SIM_TASK_CONFIGS[task_name]
        robot_config = get_robot_config('fr3')
        super().__init__(policy_inference, task_config, robot_config)
        
    def preprocess_images(self, images_dict):
        """预处理FR3图像数据"""
        processed_images = {}
        
        for cam_name, image in images_dict.items():
            if cam_name not in self.robot_config.camera_names:
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
        if qpos.shape[0] != self.robot_config.dof:
            raise ValueError(f"QPos维度不匹配: 期望{self.robot_config.dof}, 得到{qpos.shape[0]}")
            
        # 验证相机数量
        if len(images_dict) != len(self.robot_config.camera_names):
            raise ValueError(f"相机数量不匹配: 期望{self.robot_config.camera_names}, 得到{list(images_dict.keys())}")
            
        # 验证相机名称
        for cam_name in images_dict.keys():
            if cam_name not in self.robot_config.camera_names:
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
            for cam_name in self.robot_config.camera_names:
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


class Monte01Inference(RobotInference):
    """Monte01双臂机器人特定的推理器"""
    
    def __init__(self, policy_inference, task_name='monte01_peg_in_hole'):
        """
        初始化Monte01推理器
        
        Args:
            policy_inference: 策略推理器实例
            task_name: 任务名称
        """
        from constants import SIM_TASK_CONFIGS
        from robot_config import get_robot_config
        
        self.task_name = task_name
        
        if task_name not in SIM_TASK_CONFIGS:
            raise ValueError(f"未知任务名称: {task_name}")
            
        task_config = SIM_TASK_CONFIGS[task_name]
        robot_config = get_robot_config('monte01')
        super().__init__(policy_inference, task_config, robot_config)
        
    def preprocess_images(self, images_dict):
        """预处理Monte01图像数据 (支持三个相机)"""
        processed_images = {}
        
        for cam_name, image in images_dict.items():
            if cam_name not in self.robot_config.camera_names:
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
        """验证Monte01输入数据格式"""
        # 验证qpos维度 (16 DOF)
        qpos = np.array(qpos)
        if qpos.shape[0] != self.robot_config.dof:
            raise ValueError(f"QPos维度不匹配: 期望{self.robot_config.dof}, 得到{qpos.shape[0]}")
            
        # 验证相机数量 (支持缺失相机)
        required_cameras = set(self.robot_config.camera_names)
        available_cameras = set(images_dict.keys())
        
        if not available_cameras.issubset(required_cameras):
            unknown_cams = available_cameras - required_cameras
            raise ValueError(f"未知相机名称: {unknown_cams}")
                
    def predict_from_episode(self, episode_path, timestep=0):
        """
        从episode文件中读取数据并进行预测 (双臂版本)
        
        Args:
            episode_path: HDF5 episode文件路径
            timestep: 要预测的时间步
            
        Returns:
            predicted_actions: 预测的动作序列 (16 DOF)
            ground_truth: 真实动作 (用于对比)
            qpos: 关节位置 (16 DOF)
        """
        with h5py.File(episode_path, 'r') as f:
            # 读取数据
            qpos = f['observations/qpos'][timestep]
            
            # 构建图像字典 - 使用配置中的相机名称，支持缺失相机
            images_dict = {}
            for cam_name in self.robot_config.camera_names:
                if f'observations/images/{cam_name}' in f:
                    images_dict[cam_name] = f[f'observations/images/{cam_name}'][timestep]
                else:
                    print(f"⚠️ 相机数据缺失: {cam_name}, 将使用零图像")
                    # 创建零图像
                    images_dict[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 真实动作 (用于对比)
            if timestep < f['action'].shape[0]:
                ground_truth = f['action'][timestep]
            else:
                ground_truth = None
                
        # 进行预测
        predicted_actions = self.predict(qpos, images_dict)
        
        return predicted_actions, ground_truth, qpos
        
    def split_prediction(self, actions):
        """
        将双臂动作分离为左臂和右臂
        
        Args:
            actions: 组合动作 (16 DOF)
            
        Returns:
            left_actions: 左臂动作 (8 DOF)
            right_actions: 右臂动作 (8 DOF)
        """
        return self.robot_config.split_state(actions)
        
    def combine_actions(self, left_actions, right_actions):
        """
        组合左臂和右臂动作
        
        Args:
            left_actions: 左臂动作 (8 DOF)
            right_actions: 右臂动作 (8 DOF)
            
        Returns:
            combined_actions: 组合动作 (16 DOF)
        """
        return self.robot_config.combine_state(left_actions, right_actions)
        
    def evaluate_episode(self, episode_path, num_steps=10):
        """
        评估整个episode的预测性能 (双臂版本)
        
        Args:
            episode_path: HDF5 episode文件路径
            num_steps: 要评估的步数
            
        Returns:
            avg_mse: 平均MSE误差
        """
        print(f"\n📊 评估Monte01 Episode: {episode_path}")
        
        mse_errors = []
        left_mse_errors = []
        right_mse_errors = []
        
        with h5py.File(episode_path, 'r') as f:
            episode_len = min(f['action'].shape[0], num_steps)
            
        for t in range(episode_len):
            pred_actions, gt_action, qpos = self.predict_from_episode(episode_path, t)
            
            if gt_action is not None:
                # 计算整体MSE误差
                mse = np.mean((pred_actions[0] - gt_action) ** 2)
                mse_errors.append(mse)
                
                # 分别计算左右臂MSE
                pred_left, pred_right = self.split_prediction(pred_actions[0])
                gt_left, gt_right = self.split_prediction(gt_action)
                
                left_mse = np.mean((pred_left - gt_left) ** 2)
                right_mse = np.mean((pred_right - gt_right) ** 2)
                
                left_mse_errors.append(left_mse)
                right_mse_errors.append(right_mse)
                
                if t % 5 == 0:  # 每5步打印一次
                    print(f"  步骤 {t:3d}: 总MSE={mse:.6f}, 左臂={left_mse:.6f}, 右臂={right_mse:.6f}")
                    
        avg_mse = np.mean(mse_errors)
        avg_left_mse = np.mean(left_mse_errors) if left_mse_errors else 0.0
        avg_right_mse = np.mean(right_mse_errors) if right_mse_errors else 0.0
        
        print(f"\n📈 平均MSE误差:")
        print(f"   - 总体: {avg_mse:.6f}")
        print(f"   - 左臂: {avg_left_mse:.6f}")
        print(f"   - 右臂: {avg_right_mse:.6f}")
        
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


class Monte01RealtimeController(RealtimeController):
    """Monte01双臂机器人特定的实时控制器"""
    
    def get_robot_state(self):
        """获取Monte01机器人状态 (16 DOF)"""
        # TODO: 实现实际的Monte01状态读取
        # 这里返回模拟数据
        left_arm = np.array([0.0, -0.5, 0.3, 0.0, -0.2, 0.1, 0.0, 0.037])  # 8 DOF
        right_arm = np.array([0.0, -0.5, 0.3, 0.0, -0.2, 0.1, 0.0, 0.037])  # 8 DOF
        qpos = np.concatenate([left_arm, right_arm])  # 16 DOF
        return qpos
        
    def get_camera_images(self):
        """获取Monte01相机图像 (三个相机)"""
        # TODO: 实现实际的Monte01相机图像获取
        # 这里返回模拟数据
        ee_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        right_ee_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        third_person_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        
        return {
            'ee_cam': ee_img,
            'right_ee_cam': right_ee_img,
            'third_person_cam': third_person_img
        }
        
    def send_action_to_robot(self, action):
        """发送动作到Monte01机器人 (16 DOF)"""
        # TODO: 实现实际的Monte01机器人控制
        left_action = action[:8]
        right_action = action[8:16]
        print(f"发送动作到Monte01机器人:")
        print(f"  左臂: {left_action}")
        print(f"  右臂: {right_action}")


def create_robot_inference(ckpt_dir, task_name=None, robot_type=None):
    """
    工厂函数：自动创建合适的机器人推理器
    
    Args:
        ckpt_dir: 检查点目录路径
        task_name: 任务名称 (可选，用于自动检测)
        robot_type: 机器人类型 (可选，'fr3' 或 'monte01')
        
    Returns:
        robot_inference: 机器人推理器实例
    """
    from policy_inference import ACTInference
    from robot_config import detect_robot_from_checkpoint, detect_robot_from_task
    
    # 自动检测机器人类型
    if robot_type is None:
        if task_name:
            robot_type = detect_robot_from_task(task_name)
        else:
            robot_type = detect_robot_from_checkpoint(ckpt_dir)
            
    print(f"🤖 检测到机器人类型: {robot_type}")
    
    # 获取机器人配置
    robot_config = get_robot_config(robot_type)
    
    # 创建策略推理器
    policy_inference = ACTInference(
        ckpt_dir=ckpt_dir,
        state_dim=robot_config.dof,
        camera_names=robot_config.camera_names
    )
    
    # 创建机器人推理器
    if robot_type == 'fr3':
        if not task_name:
            task_name = 'fr3_peg_in_hole'
        robot_inference = FR3Inference(policy_inference, task_name)
    elif robot_type == 'monte01':
        if not task_name:
            task_name = 'monte01_peg_in_hole'
        robot_inference = Monte01Inference(policy_inference, task_name)
    else:
        raise ValueError(f"不支持的机器人类型: {robot_type}")
        
    print(f"✅ {robot_type.upper()} 推理器创建完成")
    print(f"   - 任务: {task_name}")
    print(f"   - DOF: {robot_config.dof}")
    print(f"   - 相机: {robot_config.camera_names}")
    
    return robot_inference


def create_realtime_controller(robot_inference):
    """
    工厂函数：创建合适的实时控制器
    
    Args:
        robot_inference: 机器人推理器实例
        
    Returns:
        realtime_controller: 实时控制器实例
    """
    if isinstance(robot_inference, FR3Inference):
        return FR3RealtimeController(robot_inference)
    elif isinstance(robot_inference, Monte01Inference):
        return Monte01RealtimeController(robot_inference)
    else:
        raise ValueError(f"不支持的推理器类型: {type(robot_inference)}")


def main():
    """测试脚本"""
    import argparse
    
    parser = argparse.ArgumentParser(description='机器人推理测试')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                       help='检查点目录路径')
    parser.add_argument('--task_name', type=str, default=None,
                       help='任务名称')
    parser.add_argument('--robot_type', type=str, default=None,
                       choices=['fr3', 'monte01'], help='机器人类型')
    parser.add_argument('--episode', type=str, default=None,
                       help='测试episode文件路径')
    parser.add_argument('--num_steps', type=int, default=20,
                       help='评估步数')
    
    args = parser.parse_args()
    
    try:
        # 创建机器人推理器
        robot_inference = create_robot_inference(
            args.ckpt_dir, 
            args.task_name, 
            args.robot_type
        )
        
        if args.episode:
            # 测试episode文件
            print(f"\n🎯 测试episode文件: {args.episode}")
            avg_mse = robot_inference.evaluate_episode(args.episode, args.num_steps)
            print(f"\n✅ 测试完成! 平均MSE: {avg_mse:.6f}")
        else:
            # 创建实时控制器测试
            print(f"\n🎮 创建实时控制器测试")
            controller = create_realtime_controller(robot_inference)
            print(f"✅ 实时控制器创建成功: {type(controller).__name__}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()