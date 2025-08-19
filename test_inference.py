#!/usr/bin/env python3

import sys
import numpy as np
import os

# Set up argv before any imports to avoid argparse conflicts
def setup_argv():
    if len(sys.argv) == 1:  # Only if run directly without arguments
        sys.argv = ['test_inference.py', '--ckpt_dir', 'ckpts/fr3_peg_in_hole_act', '--policy_class', 'ACT', '--task_name', 'fr3_peg_in_hole_extended', '--seed', '0', '--num_epochs', '1']

setup_argv()
from inference_fr3 import FR3ACTInference

def quick_test():
    """快速测试推理功能"""
    
    # 配置路径
    ckpt_dir = "ckpts/fr3_peg_in_hole_act"  # 之前训练的检查点目录
    dataset_dir = "/media/hanyu/ubuntu/act_project/peg_in_hole_hdf5_extended"
    
    print("🧪 开始快速推理测试...")
    
    try:
        # 初始化推理器
        inferencer = FR3ACTInference(ckpt_dir)
        print("✅ 推理器初始化成功")
        
        # 找到第一个测试文件
        test_files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
        if not test_files:
            print("❌ 没有找到测试文件")
            return
            
        test_file = os.path.join(dataset_dir, test_files[0])
        print(f"📂 使用测试文件: {test_files[0]}")
        
        # 进行单步预测测试
        print("\n🎯 单步预测测试:")
        pred_actions, gt_action, qpos = inferencer.predict_from_episode(test_file, timestep=0)
        
        print(f"输入QPos: {qpos}")
        print(f"预测动作序列形状: {pred_actions.shape}")
        print(f"预测第一个动作: {pred_actions[0]}")
        if gt_action is not None:
            print(f"真实动作: {gt_action}")
            mse = np.mean((pred_actions[0] - gt_action) ** 2)
            print(f"MSE误差: {mse:.6f}")
        
        # 进行多步评估测试
        print(f"\n📊 多步评估测试 (前10步):")
        avg_mse = inferencer.evaluate_episode(test_file, num_steps=10)
        
        print(f"\n🎉 测试完成! 平均MSE: {avg_mse:.6f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def manual_test():
    """手动输入数据测试"""
    
    ckpt_dir = "ckpts/fr3_peg_in_hole_act"
    
    print("🔧 手动数据测试...")
    
    try:
        inferencer = FR3ACTInference(ckpt_dir)
        
        # 模拟输入数据
        # FR3: 7个arm joints + 1个gripper = 8 DOF
        qpos = np.array([0.1, -0.5, 0.3, 0.0, -0.2, 0.1, 0.0, 0.05])  # 8维关节位置
        
        # 模拟图像 (480x640x3)
        ee_cam_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        third_person_img = np.random.rand(480, 640, 3).astype(np.uint8) * 255
        
        images_dict = {
            'ee_cam': ee_cam_img,
            'third_person_cam': third_person_img
        }
        
        # 进行推理
        predicted_actions = inferencer.predict(qpos, images_dict)
        
        print(f"✅ 推理成功!")
        print(f"输入QPos: {qpos}")
        print(f"预测动作序列形状: {predicted_actions.shape}")
        print(f"预测第一个动作: {predicted_actions[0]}")
        
    except Exception as e:
        print(f"❌ 手动测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 FR3 ACT推理测试")
    print("=" * 50)
    
    # 检查模型文件是否存在
    ckpt_dir = "ckpts/fr3_peg_in_hole_act"
    
    if os.path.exists(os.path.join(ckpt_dir, "dataset_stats.pkl")) or os.path.exists(os.path.join(ckpt_dir, "dataset_stats_bac.pkl")):
        if os.path.exists(os.path.join(ckpt_dir, "policy_best.ckpt")) or os.path.exists(os.path.join(ckpt_dir, "policy_best_bac.ckpt")):
            print("✅ 找到模型文件，开始测试...")
            quick_test()
        else:
            print("❌ 没有找到policy_best.ckpt文件")
            print("💡 尝试手动数据测试...")
            manual_test()
    else:
        print("❌ 没有找到dataset_stats.pkl文件")
        print("请确保训练完成并生成了相应的文件")