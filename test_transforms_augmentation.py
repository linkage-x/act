#!/usr/bin/env python3

import torch
import numpy as np
import yaml
import os
import cv2
import matplotlib.pyplot as plt
from data_augmentation import create_training_augmentation

def test_transforms_augmentation():
    """测试PyTorch transforms光照增强"""

    print("🧪 测试PyTorch transforms光照增强...")

    # 1. 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'lighting_augmentation.yaml')
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"✅ 加载配置: {config}")

    # 2. 创建增强函数
    augmentation_fn = create_training_augmentation(config)

    if augmentation_fn is None:
        print("⚠️  增强功能已禁用")
        return True

    # 3. 创建测试图像 (模拟相机输入)
    # 创建一个简单的彩色测试图像
    test_image = create_test_image()

    # 转换为PyTorch tensor格式 (C, H, W)
    image_tensor = torch.from_numpy(test_image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW

    print(f"📊 测试图像形状: {image_tensor.shape}")
    print(f"📊 像素值范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

    # 4. 应用增强
    print("\n🎨 应用光照增强...")

    # 测试单张图像
    augmented_single = augmentation_fn(image_tensor)
    print(f"✅ 单张图像增强: {image_tensor.shape} -> {augmented_single.shape}")

    # 测试批次图像 (模拟多相机输入)
    batch_images = torch.stack([image_tensor, image_tensor, image_tensor], dim=0)  # (3, C, H, W)
    augmented_batch = augmentation_fn(batch_images)
    print(f"✅ 批次图像增强: {batch_images.shape} -> {augmented_batch.shape}")

    # 5. 验证输出
    print("\n🔍 验证增强效果...")

    # 检查输出范围
    assert augmented_single.min() >= 0.0 and augmented_single.max() <= 1.0, "单张图像像素值超出范围"
    assert augmented_batch.min() >= 0.0 and augmented_batch.max() <= 1.0, "批次图像像素值超出范围"

    # 检查形状保持
    assert augmented_single.shape == image_tensor.shape, "单张图像形状改变"
    assert augmented_batch.shape == batch_images.shape, "批次图像形状改变"

    print("✅ 所有验证通过")

    # 6. 可视化效果 (可选)
    if True:  # 设为True查看效果
        visualize_augmentation_effects(image_tensor, augmented_single, augmentation_fn)

    return True

def create_test_image(height=480, width=640):
    """创建彩色测试图像"""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建渐变背景
    for i in range(height):
        for j in range(width):
            image[i, j, 0] = int(255 * i / height)  # 红色渐变
            image[i, j, 1] = int(255 * j / width)   # 绿色渐变
            image[i, j, 2] = 128                     # 蓝色常量

    # 添加一些几何形状
    cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(image, (400, 300), 50, (0, 0, 255), -1)
    cv2.line(image, (0, 0), (width, height), (255, 0, 0), 3)

    return image

def visualize_augmentation_effects(original, augmented, augmentation_fn, num_samples=4):
    """可视化增强效果"""
    print("\n🖼️  生成可视化效果...")

    # 生成多个增强样本
    samples = [original]
    for i in range(num_samples):
        sample = augmentation_fn(original)
        samples.append(sample)

    # 转换为numpy格式用于显示
    samples_np = []
    for sample in samples:
        # CHW -> HWC, [0,1] -> [0,255]
        img_np = (sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        samples_np.append(img_np)

    # 创建图表
    fig, axes = plt.subplots(1, len(samples_np), figsize=(15, 3))

    titles = ['原始'] + [f'增强{i+1}' for i in range(num_samples)]

    for i, (img, title) in enumerate(zip(samples_np, titles)):
        if len(samples_np) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()

    # 保存到临时文件
    output_path = '/tmp/lighting_augmentation_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 可视化结果已保存: {output_path}")
    plt.close()

def performance_test(augmentation_fn, batch_size=8, num_iterations=100):
    """性能测试"""
    print(f"\n⚡ 性能测试 (批次大小: {batch_size}, 迭代次数: {num_iterations})...")

    # 创建测试批次
    test_images = torch.randn(batch_size, 3, 480, 640)  # 随机图像
    test_images = torch.clamp(test_images * 0.3 + 0.5, 0, 1)  # 规范化到[0,1]

    import time

    # 预热
    for _ in range(10):
        _ = augmentation_fn(test_images)

    # 计时
    start_time = time.time()
    for _ in range(num_iterations):
        _ = augmentation_fn(test_images)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = batch_size / avg_time

    print(f"📊 平均处理时间: {avg_time*1000:.2f}ms/批次")
    print(f"📊 处理速度: {fps:.1f} 图像/秒")
    print(f"📊 总时间: {total_time:.2f}秒")

if __name__ == '__main__':
    # 运行测试
    success = test_transforms_augmentation()

    if success:
        print("\n🎉 PyTorch transforms光照增强测试成功!")

        # 可选：运行性能测试
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'configs', 'lighting_augmentation.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            augmentation_fn = create_training_augmentation(config)
            if augmentation_fn:
                performance_test(augmentation_fn)
        except Exception as e:
            print(f"⚠️  性能测试跳过: {e}")
    else:
        print("\n❌ 测试失败")
        exit(1)