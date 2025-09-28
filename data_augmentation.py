import torch
import numpy as np
from torchvision import transforms
import random

class LightingAugmentationTransform:
    """基于PyTorch transforms的光照增强

    使用torchvision的ColorJitter和自定义变换实现高效的光照增强
    """

    def __init__(self, config):
        """
        Args:
            config: 增强配置字典
        """
        self.enabled = config.get('enabled', True)

        if not self.enabled:
            self.transform = None
            print("ℹ️ 训练数据增强已禁用")
            return

        # 颜色抖动参数
        brightness = config.get('brightness', 0.3)
        contrast = config.get('contrast', 0.2)
        saturation = config.get('saturation', 0.2)
        hue = config.get('hue', 0.1)

        # 自适应光照标准化
        self.adaptive_normalization = config.get('adaptive_normalization', True)
        self.target_mean = config.get('target_mean', 0.5)

        # 构建变换管道
        transform_list = []

        # 核心颜色抖动（使用torchvision的优化实现）
        if any([brightness > 0, contrast > 0, saturation > 0, hue > 0]):
            color_jitter = transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            )
            transform_list.append(color_jitter)

        # 自适应光照标准化
        if self.adaptive_normalization:
            transform_list.append(AdaptiveLightingNorm(self.target_mean))

        # 组合所有变换
        if transform_list:
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = None

        print(f"✅ 训练数据增强已启用: 亮度±{brightness*100:.0f}%, 对比度±{contrast*100:.0f}%, 饱和度±{saturation*100:.0f}%, 色调±{hue*100:.0f}%")

    def __call__(self, images):
        """
        对图像应用增强

        Args:
            images: Tensor of shape (N, C, H, W) or (C, H, W), values in [0, 1]

        Returns:
            增强后的图像，相同形状
        """
        if not self.enabled or self.transform is None:
            return images

        # 处理不同的输入形状
        if images.dim() == 3:
            # 单张图像 (C, H, W)
            images = images.unsqueeze(0)
            single_image = True
        else:
            single_image = False

        # 对批次中的每张图像独立应用变换
        batch_size = images.shape[0]
        augmented_images = []

        for i in range(batch_size):
            img = images[i]  # (C, H, W)
            augmented_img = self.transform(img)
            augmented_images.append(augmented_img)

        result = torch.stack(augmented_images)

        # 恢复原始形状
        if single_image:
            result = result.squeeze(0)

        return result


class AdaptiveLightingNorm:
    """自适应光照标准化变换"""

    def __init__(self, target_mean=0.5):
        self.target_mean = target_mean

    def __call__(self, image):
        """
        Args:
            image: Tensor of shape (C, H, W), values in [0, 1]
        """
        # 计算当前平均亮度
        current_mean = torch.mean(image).item()

        if current_mean <= 0:
            return image

        # 计算标准化因子
        normalization_factor = self.target_mean / current_mean

        # 限制调整幅度避免过度曝光
        normalization_factor = np.clip(normalization_factor, 0.5, 2.0)

        # 应用标准化
        normalized = image * normalization_factor

        return torch.clamp(normalized, 0.0, 1.0)


class RandomLightingAugmentation:
    """随机应用光照增强（用于控制增强概率）"""

    def __init__(self, base_transform, probability=0.8):
        self.base_transform = base_transform
        self.probability = probability

    def __call__(self, images):
        if random.random() < self.probability:
            return self.base_transform(images)
        else:
            return images


def create_training_augmentation(config):
    """创建训练时的光照增强管道

    Args:
        config: 增强配置字典

    Returns:
        增强函数，接受图像tensor并返回增强后的tensor，如果禁用则返回None
    """
    lighting_config = config.get('lighting_augmentation', {})

    if not lighting_config.get('enabled', True):
        print("ℹ️ 训练数据增强已禁用")
        return None

    # 创建基础增强
    base_augmentation = LightingAugmentationTransform(lighting_config)

    # 可选：添加随机性控制
    augmentation_probability = lighting_config.get('probability', 1.0)
    if augmentation_probability < 1.0:
        augmentation = RandomLightingAugmentation(base_augmentation, augmentation_probability)
    else:
        augmentation = base_augmentation

    return augmentation