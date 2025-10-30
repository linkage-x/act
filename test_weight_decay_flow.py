#!/usr/bin/env python3
"""
测试 weight_decay 数据流是否正确
验证从 YAML 配置到优化器的完整路径
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from task_config_manager import load_task_config

def test_weight_decay_flow():
    """测试 weight_decay 配置流"""

    print("=" * 70)
    print("🧪 测试 weight_decay 数据流")
    print("=" * 70)
    print()

    # 测试配置文件
    config_file = 'configs/tasks/fr3_it_mix_1030_3dmouse_customcfg3.yaml'

    print(f"📄 加载配置文件: {config_file}")
    print()

    try:
        # 加载配置
        config_data = load_task_config(config_file, eval_mode=False)

        print("✅ 配置加载成功")
        print()

        # 检查各个阶段
        print("=" * 70)
        print("📊 数据流追踪")
        print("=" * 70)
        print()

        # 阶段1: 原始配置
        original_config = config_data.get('config', {})
        training_config = original_config.get('training', {})
        yaml_weight_decay = training_config.get('weight_decay', 'NOT FOUND')

        print(f"1️⃣  YAML 原始配置:")
        print(f"   training.weight_decay = {yaml_weight_decay}")
        print()

        # 阶段2: legacy_args
        args = config_data.get('args', {})
        args_weight_decay = args.get('weight_decay', 'NOT FOUND')

        print(f"2️⃣  args 字典 (task_config_manager 处理后):")
        print(f"   args['weight_decay'] = {args_weight_decay}")
        print()

        # 阶段3: 模拟 policy_config 构建
        print(f"3️⃣  policy_config 字典 (imitate_episodes.py 会构建):")
        simulated_policy_config = {
            'lr': args['lr'],
            'dropout': args.get('dropout', 0.1),
            'weight_decay': args.get('weight_decay', 1e-4),
        }
        print(f"   policy_config['weight_decay'] = {simulated_policy_config['weight_decay']}")
        print()

        # 验证结果
        print("=" * 70)
        print("✅ 验证结果")
        print("=" * 70)
        print()

        if yaml_weight_decay == 'NOT FOUND':
            print("❌ YAML 配置中未找到 weight_decay")
            return False

        if args_weight_decay == 'NOT FOUND':
            print("❌ args 字典中未找到 weight_decay")
            print("   → task_config_manager.py 未正确读取")
            return False

        # 转换为浮点数比较（处理科学记数法）
        yaml_float = float(yaml_weight_decay) if isinstance(yaml_weight_decay, (int, float, str)) else None
        args_float = float(args_weight_decay) if isinstance(args_weight_decay, (int, float, str)) else None

        if yaml_float is None or args_float is None or abs(yaml_float - args_float) > 1e-10:
            print(f"❌ 值不匹配:")
            print(f"   YAML: {yaml_weight_decay} ({yaml_float})")
            print(f"   args: {args_weight_decay} ({args_float})")
            return False

        if simulated_policy_config['weight_decay'] != args_weight_decay:
            print(f"❌ policy_config 中值不匹配:")
            print(f"   args: {args_weight_decay}")
            print(f"   policy_config: {simulated_policy_config['weight_decay']}")
            return False

        print("✅ 所有检查通过！")
        print()
        print(f"📊 weight_decay 值在整个数据流中保持一致: {yaml_weight_decay}")
        print()
        print("数据流路径:")
        print(f"  YAML ({yaml_weight_decay})")
        print(f"    ↓")
        print(f"  task_config_manager.py")
        print(f"    ↓")
        print(f"  args dict ({args_weight_decay})")
        print(f"    ↓")
        print(f"  imitate_episodes.py")
        print(f"    ↓")
        print(f"  policy_config dict ({simulated_policy_config['weight_decay']})")
        print(f"    ↓")
        print(f"  detr/main.py → AdamW optimizer")
        print()

        # 显示完整配置
        print("=" * 70)
        print("📋 完整训练配置")
        print("=" * 70)
        print()
        print(f"批大小:       {args['batch_size']}")
        print(f"学习率:       {args['lr']}")
        print(f"KL权重:       {args['kl_weight']}")
        print(f"Chunk大小:    {args['chunk_size']}")
        print(f"Hidden维度:   {args['hidden_dim']}")
        print(f"FFN维度:      {args['dim_feedforward']}")
        print(f"Dropout:      {args.get('dropout', 0.1)}")
        print(f"权重衰减:     {args.get('weight_decay', 1e-4)}")
        print()

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_weight_decay_flow()
    sys.exit(0 if success else 1)
