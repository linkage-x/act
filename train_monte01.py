#!/usr/bin/env python3
"""
Monte01双臂机器人ACT训练脚本示例
"""

import os
import sys
import argparse


def train_monte01():
    """训练Monte01双臂机器人的ACT策略"""
    
    print("🤖 Monte01 双臂机器人 ACT 训练")
    print("=" * 50)
    print(f"任务: monte01_peg_in_hole")
    print(f"检查点目录: ckpts/monte01_peg_in_hole")
    print(f"DOF: 8 (当前配置为单臂，需要更新constants.py)")
    print(f"批大小: 16")
    print(f"训练轮数: 3000")
    print("=" * 50)
    
    # 构造命令行参数
    import sys
    original_argv = sys.argv.copy()
    
    sys.argv = [
        'train_monte01.py',
        '--task_name', 'monte01_peg_in_hole',
        '--ckpt_dir', 'ckpts/monte01_peg_in_hole',
        '--policy_class', 'ACT',
        '--kl_weight', '10',
        '--chunk_size', '100',
        '--hidden_dim', '512',
        '--batch_size', '16',
        '--dim_feedforward', '3200',
        '--num_epochs', '3000',
        '--lr', '2e-5',
        '--seed', '0'
    ]
    
    try:
        # 导入训练模块并解析参数
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--onscreen_render', action='store_true')
        parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
        parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
        parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
        parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
        parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
        parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
        parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
        parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
        parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
        parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
        parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
        parser.add_argument('--temporal_agg', action='store_true')
        
        args = parser.parse_args()
        
        from imitate_episodes import main
        main(vars(args))
        
    finally:
        # 恢复原始argv
        sys.argv = original_argv


def evaluate_monte01():
    """评估Monte01双臂机器人的策略"""
    
    config = {
        'task_name': 'monte01_peg_in_hole',
        'ckpt_dir': 'ckpts/monte01_peg_in_hole',
        'policy_class': 'ACT',
        'eval': True,
        'temporal_agg': True,  # 评估时启用时间聚合
        'onscreen_render': False,
        'seed': 0,
    }
    
    print("📊 Monte01 双臂机器人 ACT 评估")
    print("=" * 50)
    
    from imitate_episodes import main
    main(config)


def test_inference():
    """测试推理器"""
    
    print("🎯 测试 Monte01 推理器")
    print("=" * 40)
    
    try:
        from robot_inference import create_robot_inference
        
        ckpt_dir = 'ckpts/monte01_peg_in_hole'
        
        # 自动创建推理器
        robot_inference = create_robot_inference(
            ckpt_dir=ckpt_dir,
            task_name='monte01_peg_in_hole'
        )
        
        print(f"✅ 推理器创建成功: {type(robot_inference).__name__}")
        
        # 测试双臂功能
        import numpy as np
        
        # 创建16 DOF测试数据
        qpos = np.random.randn(16) * 0.1
        images = {
            'ee_cam': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'right_ee_cam': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            'third_person_cam': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        
        # 进行推理
        predicted_actions = robot_inference.predict(qpos, images)
        
        print(f"✅ 推理测试成功!")
        print(f"   - 输入状态维度: {qpos.shape}")
        print(f"   - 预测动作形状: {predicted_actions.shape}")
        
        # 测试双臂分离功能
        if hasattr(robot_inference, 'split_prediction'):
            left_actions, right_actions = robot_inference.split_prediction(predicted_actions[0])
            print(f"   - 左臂动作: {left_actions.shape}")
            print(f"   - 右臂动作: {right_actions.shape}")
            print(f"   - 左臂夹爪: {left_actions[7]:.4f}m")
            print(f"   - 右臂夹爪: {right_actions[7]:.4f}m")
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()


def convert_data():
    """转换数据到HDF5格式"""
    
    print("📁 转换数据到 Monte01 格式")
    print("=" * 40)
    
    # 示例命令
    cmd = """
    python convert_fr3_robust.py \\
        --input_dir /boot/common_data/peg_in_hole_merged \\
        --output_dir /boot/common_data/peg_in_hole_hdf5 \\
        --episodes all
    """
    
    print("使用以下命令转换数据:")
    print(cmd)
    print("\n该脚本会自动检测双臂机器人配置 (16 DOF)")
    
    # 可以选择直接运行转换
    import subprocess
    
    response = input("\n是否立即运行数据转换? (y/n): ")
    if response.lower() == 'y':
        cmd_list = [
            'python', 'convert_fr3_robust.py',
            '--input_dir', '/boot/common_data/peg_in_hole_merged',
            '--output_dir', '/boot/common_data/peg_in_hole_hdf5',
            '--episodes', 'all'
        ]
        
        try:
            subprocess.run(cmd_list, check=True)
            print("✅ 数据转换完成!")
        except subprocess.CalledProcessError as e:
            print(f"❌ 数据转换失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='Monte01双臂机器人ACT训练工具')
    parser.add_argument('mode', choices=['train', 'eval', 'test', 'convert'],
                       help='操作模式')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts/monte01_peg_in_hole',
                       help='检查点目录')
    parser.add_argument('--episode', type=str, default=None,
                       help='测试episode文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_monte01()
    elif args.mode == 'eval':
        evaluate_monte01()
    elif args.mode == 'test':
        test_inference()
    elif args.mode == 'convert':
        convert_data()
    else:
        print(f"未知模式: {args.mode}")
        
    print("\n🎉 操作完成!")


if __name__ == "__main__":
    main()
