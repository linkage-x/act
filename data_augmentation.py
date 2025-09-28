#!/usr/bin/env python3
"""
数据增强模块
为ACT训练添加光照鲁棒性增强
"""

import torch
import torchvision.transforms as transforms
import numpy as np


class LightingAugmentation:
    """光照数据增强类"""

    def __init__(self, enabled=True, brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1):
        """
        初始化光照增强

        Args:
            enabled: 是否启用增强
            brightness: 亮度变化范围 (默认±30%)
            contrast: 对比度变化范围 (默认±20%)
            saturation: 饱和度变化范围 (默认±20%)
            hue: 色调变化范围 (默认±10%)
        """
        self.enabled = enabled

        if self.enabled:
            self.color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
            self.grayscale = transforms.RandomGrayscale(p=0.1)  # 10%概率转灰度
            print(f"✅ 光照数据增强已启用: brightness={brightness}, contrast={contrast}")
        else:
            print("ℹ️ 光照数据增强已禁用")

    def __call__(self, image_tensor):
        """
        应用光照增强

        Args:
            image_tensor: 图像张量 (C, H, W) 或 (N, C, H, W)

        Returns:
            增强后的图像张量
        """
        if not self.enabled:
            return image_tensor

        # 确保输入范围在[0, 1]
        image_tensor = torch.clamp(image_tensor, 0, 1)

        # 应用颜色抖动
        augmented = self.color_jitter(image_tensor)

        # 应用随机灰度
        augmented = self.grayscale(augmented)

        return augmented


class AdaptiveLightingNormalization:
    """自适应光照标准化"""

    def __init__(self, enabled=True, target_mean=0.5, adaptation_strength=0.3):
        """
        初始化自适应光照标准化

        Args:
            enabled: 是否启用
            target_mean: 目标亮度均值
            adaptation_strength: 适应强度 (0-1)
        """
        self.enabled = enabled
        self.target_mean = target_mean
        self.adaptation_strength = adaptation_strength

        if self.enabled:
            print(f"✅ 自适应光照标准化已启用: target_mean={target_mean}")
        else:
            print("ℹ️ 自适应光照标准化已禁用")

    def __call__(self, image_tensor):
        """
        应用自适应光照标准化

        Args:
            image_tensor: 图像张量 (C, H, W) 或 (N, C, H, W)

        Returns:
            标准化后的图像张量
        """
        if not self.enabled:
            return image_tensor

        # 计算当前亮度
        if len(image_tensor.shape) == 4:  # (N, C, H, W)
            current_mean = torch.mean(image_tensor, dim=(1, 2, 3), keepdim=True)
        else:  # (C, H, W)
            current_mean = torch.mean(image_tensor)

        # 计算调整因子
        adjustment = self.target_mean / (current_mean + 1e-6)
        adjustment = torch.clamp(adjustment, 0.7, 1.4)  # 限制调整范围

        # 应用调整
        adjusted = image_tensor * adjustment
        adjusted = torch.clamp(adjusted, 0, 1)

        return adjusted


def create_training_augmentation(config):
    """
    创建训练时的数据增强管道

    Args:
        config: 配置字典

    Returns:
        数据增强函数
    """
    lighting_config = config.get('lighting_augmentation', {})

    lighting_aug = LightingAugmentation(
        enabled=lighting_config.get('enabled', True),
        brightness=lighting_config.get('brightness', 0.3),
        contrast=lighting_config.get('contrast', 0.2),
        saturation=lighting_config.get('saturation', 0.2),
        hue=lighting_config.get('hue', 0.1)
    )

    adaptive_norm = AdaptiveLightingNormalization(
        enabled=lighting_config.get('adaptive_normalization', True),
        target_mean=lighting_config.get('target_mean', 0.5)
    )

    def augment_fn(image_tensor):
        # 先应用光照增强
        augmented = lighting_aug(image_tensor)

        # 再应用自适应标准化
        normalized = adaptive_norm(augmented)

        return normalized

    return augment_fn if lighting_config.get('enabled', True) else None