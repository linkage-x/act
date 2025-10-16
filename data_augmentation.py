import torch
import numpy as np
from torchvision import transforms
import random

class RotationAugmentationTransform:
    """相机图片角度增强（旋转）

    对指定的相机视图应用随机小角度旋转，以增强模型对相机角度变化的鲁棒性
    """

    def __init__(self, config, camera_names):
        """
        Args:
            config: 旋转增强配置字典
            camera_names: 所有相机名称列表
        """
        self.enabled = config.get('enabled', True)

        if not self.enabled:
            self.rotation_angles = None
            print("ℹ️ 旋转数据增强已禁用")
            return

        # 获取旋转角度范围（单位：度）
        self.max_rotation_degrees = config.get('max_rotation_degrees', 5.0)

        # 获取需要应用旋转的相机列表
        target_cameras = config.get('target_cameras', [])

        # 创建相机索引映射
        self.camera_mask = []
        for cam_name in camera_names:
            should_rotate = cam_name in target_cameras
            self.camera_mask.append(should_rotate)

        self.camera_names = camera_names
        self.target_cameras = target_cameras

        if not any(self.camera_mask):
            print("⚠️ 未指定有效的旋转相机，旋转增强将被禁用")
            self.enabled = False
            return

        enabled_cameras = [cam_name for cam_name, mask in zip(camera_names, self.camera_mask) if mask]
        print(f"✅ 旋转数据增强已启用: ±{self.max_rotation_degrees}° for cameras: {enabled_cameras}")

    def __call__(self, images):
        """
        对指定相机的图像应用随机旋转

        Args:
            images: Tensor of shape (N, C, H, W), values in [0, 1]
                    N 对应不同的相机视图

        Returns:
            增强后的图像，相同形状
        """
        if not self.enabled or images.dim() != 4:
            return images

        batch_size = images.shape[0]
        result_images = []

        for i in range(batch_size):
            img = images[i]  # (C, H, W)

            # 检查该相机是否需要旋转
            if i < len(self.camera_mask) and self.camera_mask[i]:
                # 随机生成旋转角度
                angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)

                # 使用torchvision的旋转函数
                # 注意：需要先转为PIL兼容格式，或使用functional API
                rotated_img = transforms.functional.rotate(img, angle,
                                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                                           expand=False,
                                                           fill=0)
                result_images.append(rotated_img)
            else:
                # 不旋转的相机直接添加
                result_images.append(img)

        return torch.stack(result_images)


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


class ComposedAugmentation:
    """组合多个增强变换"""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, images):
        for transform in self.transforms:
            if transform is not None:
                images = transform(images)
        return images


def create_training_augmentation(config, camera_names=None):
    """创建训练时的数据增强管道

    Args:
        config: 增强配置字典
        camera_names: 相机名称列表（旋转增强需要）

    Returns:
        增强函数，接受图像tensor并返回增强后的tensor，如果全部禁用则返回None
    """
    augmentations = []

    # 1. 旋转增强（应用在最前面，在归一化之前）
    rotation_config = config.get('rotation_augmentation', {})
    if rotation_config.get('enabled', False):
        if camera_names is None:
            print("⚠️ 旋转增强需要camera_names参数，已跳过")
        else:
            rotation_aug = RotationAugmentationTransform(rotation_config, camera_names)
            if rotation_aug.enabled:
                augmentations.append(rotation_aug)

    # 2. 光照增强
    lighting_config = config.get('lighting_augmentation', {})
    if lighting_config.get('enabled', True):
        lighting_aug = LightingAugmentationTransform(lighting_config)
        if lighting_aug.transform is not None:
            # 可选：添加随机性控制
            augmentation_probability = lighting_config.get('probability', 1.0)
            if augmentation_probability < 1.0:
                lighting_aug = RandomLightingAugmentation(lighting_aug, augmentation_probability)
            augmentations.append(lighting_aug)

    # 如果没有任何增强，返回None
    if not augmentations:
        print("ℹ️ 训练数据增强已禁用")
        return None

    # 组合所有增强
    return ComposedAugmentation(augmentations)