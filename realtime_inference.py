#!/usr/bin/env python3

import numpy as np
import cv2
import time
from inference_fr3 import FR3ACTInference

class FR3RealtimeController:
    """FR3实时控制器"""
    
    def __init__(self, ckpt_dir, task_name='fr3_peg_in_hole_extended'):
        self.inferencer = FR3ACTInference(ckpt_dir, task_name)
        self.action_buffer = None  # 动作缓冲区
        self.buffer_index = 0      # 当前动作索引
        
        print("🤖 FR3实时控制器初始化完成")
        
    def get_robot_state(self):
        """
        获取机器人当前状态 (需要根据实际机器人接口实现)
        
        Returns:
            qpos: 8维关节位置 [7 arm joints + 1 gripper]
        """
        # TODO: 实现实际的机器人状态读取
        # 这里返回模拟数据
        qpos = np.array([0.0, -0.5, 0.3, 0.0, -0.2, 0.1, 0.0, 0.05])
        return qpos
        
    def get_camera_images(self):
        """
        获取相机图像 (需要根据实际相机接口实现)
        
        Returns:
            images_dict: {'ee_cam': img, 'third_person_cam': img}
        """
        # TODO: 实现实际的相机图像获取
        # 这里返回模拟数据
        ee_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        third_person_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        
        return {
            'ee_cam': ee_img,
            'third_person_cam': third_person_img
        }
        
    def send_action_to_robot(self, action):
        """
        发送动作到机器人 (需要根据实际机器人接口实现)
        
        Args:
            action: 8维动作 [7 arm joints + 1 gripper]
        """
        # TODO: 实现实际的机器人控制
        print(f"发送动作: {action}")
        
    def predict_action_sequence(self, qpos, images_dict):
        """预测动作序列并更新缓冲区"""
        action_sequence = self.inferencer.predict(qpos, images_dict)
        self.action_buffer = action_sequence
        self.buffer_index = 0
        return action_sequence
        
    def get_next_action(self):
        """从缓冲区获取下一个动作"""
        if self.action_buffer is None or self.buffer_index >= len(self.action_buffer):
            return None
            
        action = self.action_buffer[self.buffer_index]
        self.buffer_index += 1
        return action
        
    def run_control_loop(self, duration=30, control_frequency=10):
        """
        运行控制循环
        
        Args:
            duration: 运行时长 (秒)
            control_frequency: 控制频率 (Hz)
        """
        print(f"🚀 开始控制循环 - 时长:{duration}s, 频率:{control_frequency}Hz")
        
        dt = 1.0 / control_frequency
        start_time = time.time()
        prediction_interval = 1.0  # 每秒重新预测一次
        last_prediction_time = 0
        
        step_count = 0
        
        try:
            while time.time() - start_time < duration:
                step_start = time.time()
                
                # 获取机器人状态和图像
                qpos = self.get_robot_state()
                images_dict = self.get_camera_images()
                
                # 检查是否需要重新预测
                current_time = time.time()
                if (current_time - last_prediction_time > prediction_interval or 
                    self.action_buffer is None):
                    
                    print(f"\n🔄 步骤 {step_count}: 重新预测动作序列...")
                    prediction_start = time.time()
                    
                    try:
                        self.predict_action_sequence(qpos, images_dict)
                        prediction_time = time.time() - prediction_start
                        print(f"   预测耗时: {prediction_time*1000:.1f}ms")
                        print(f"   预测序列长度: {len(self.action_buffer)}")
                        last_prediction_time = current_time
                    except Exception as e:
                        print(f"   ❌ 预测失败: {e}")
                        continue
                
                # 获取当前动作
                action = self.get_next_action()
                if action is not None:
                    # 发送动作到机器人
                    self.send_action_to_robot(action)
                    
                    if step_count % 10 == 0:  # 每10步打印一次
                        print(f"   步骤 {step_count}: 执行动作 {action[:3]:.3f}...")
                else:
                    print(f"   ⚠️  步骤 {step_count}: 没有可用动作")
                
                # 控制循环频率
                step_time = time.time() - step_start
                sleep_time = max(0, dt - step_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif step_time > dt:
                    print(f"   ⚠️  控制循环超时: {step_time*1000:.1f}ms > {dt*1000:.1f}ms")
                
                step_count += 1
                
        except KeyboardInterrupt:
            print(f"\n⏹️  用户中断控制循环")
        except Exception as e:
            print(f"\n❌ 控制循环错误: {e}")
            import traceback
            traceback.print_exc()
        
        total_time = time.time() - start_time
        avg_frequency = step_count / total_time
        
        print(f"\n📊 控制循环完成:")
        print(f"   总步数: {step_count}")
        print(f"   总时间: {total_time:.1f}s")
        print(f"   平均频率: {avg_frequency:.1f}Hz")
        
    def test_single_prediction(self):
        """测试单次预测性能"""
        print("🧪 测试单次预测性能...")
        
        qpos = self.get_robot_state()
        images_dict = self.get_camera_images()
        
        # 预热
        for i in range(3):
            _ = self.inferencer.predict(qpos, images_dict)
        
        # 测试性能
        times = []
        for i in range(10):
            start_time = time.time()
            actions = self.inferencer.predict(qpos, images_dict)
            end_time = time.time()
            times.append(end_time - start_time)
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"📈 预测性能统计:")
        print(f"   平均耗时: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        print(f"   动作序列长度: {actions.shape[0]}")
        print(f"   理论最大频率: {1/avg_time:.1f} Hz")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FR3实时控制')
    parser.add_argument('--ckpt_dir', type=str, 
                       default='ckpts/fr3_peg_in_hole_act',
                       help='模型检查点目录')
    parser.add_argument('--duration', type=int, default=30,
                       help='控制时长(秒)')
    parser.add_argument('--frequency', type=int, default=10,
                       help='控制频率(Hz)')
    parser.add_argument('--test_only', action='store_true',
                       help='仅测试性能，不运行控制循环')
    
    args = parser.parse_args()
    
    try:
        controller = FR3RealtimeController(args.ckpt_dir)
        
        if args.test_only:
            controller.test_single_prediction()
        else:
            controller.run_control_loop(args.duration, args.frequency)
            
    except Exception as e:
        print(f"❌ 控制器启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()