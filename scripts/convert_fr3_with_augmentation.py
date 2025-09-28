#!/usr/bin/env python3

import os
import json
import numpy as np
import h5py
import cv2
import argparse
from pathlib import Path
import time
import yaml
from collections import deque
import copy

# 导入原始脚本的所有函数
from convert_fr3_preprocess import (
    robot_to_act_joint_mapping,
    detect_gripper_state_changes,
    remove_static_frames,
    downsample_episode,
    resample_episode,
    interpolate_data_points,
    segment_episode,
    segment_with_overlap,
    segment_by_action_clusters,
    detect_action_clusters,
    segment_adaptive,
    calculate_action_density
)

class LightingAugmentation:
    """光照数据增强类 - 用于离线预处理"""

    def __init__(self, config):
        """
        Args:
            config: 增强配置字典
        """
        self.enabled = config.get('enabled', True)
        if not self.enabled:
            return

        # 颜色抖动参数
        self.brightness = config.get('brightness', 0.3)
        self.contrast = config.get('contrast', 0.2)
        self.saturation = config.get('saturation', 0.2)
        self.hue = config.get('hue', 0.1)

        # 自适应光照标准化
        self.adaptive_normalization = config.get('adaptive_normalization', True)
        self.target_mean = config.get('target_mean', 0.5)

        print(f"✅ 光照增强已启用: 亮度±{self.brightness*100:.0f}%, 对比度±{self.contrast*100:.0f}%, 饱和度±{self.saturation*100:.0f}%, 色调±{self.hue*100:.0f}%")

    def augment_image(self, image):
        """
        对单张图像应用光照增强

        Args:
            image: 输入图像 (H, W, C) RGB格式, uint8

        Returns:
            增强后的图像 (H, W, C) RGB格式, uint8
        """
        if not self.enabled:
            return image

        # 转换为浮点数进行处理
        img = image.astype(np.float32) / 255.0

        # 1. 亮度调整
        if self.brightness > 0:
            brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            img = img * brightness_factor

        # 2. 对比度调整
        if self.contrast > 0:
            contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean_val = np.mean(img)
            img = (img - mean_val) * contrast_factor + mean_val

        # 3. 饱和度调整 (RGB -> HSV -> RGB)
        if self.saturation > 0:
            # 转换到HSV
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

            # 调整饱和度
            saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * saturation_factor
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)

            # 转换回RGB
            img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 4. 色调调整
        if self.hue > 0:
            # 转换到HSV
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

            # 调整色调 (H通道是0-180)
            hue_shift = np.random.uniform(-self.hue, self.hue) * 180
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + hue_shift
            img_hsv[:, :, 0] = np.mod(img_hsv[:, :, 0], 180)

            # 转换回RGB
            img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 5. 自适应光照标准化
        if self.adaptive_normalization:
            current_mean = np.mean(img)
            if current_mean > 0:
                normalization_factor = self.target_mean / current_mean
                # 限制调整幅度避免过度曝光
                normalization_factor = np.clip(normalization_factor, 0.5, 2.0)
                img = img * normalization_factor

        # 确保像素值在有效范围内
        img = np.clip(img, 0.0, 1.0)

        return (img * 255).astype(np.uint8)

def load_and_resize_image_with_augmentation(image_path, target_size=(480, 640),
                                         fallback_color=(128, 128, 128),
                                         augmentation=None, num_variants=1):
    """加载图像并应用增强，支持生成多个变体"""

    # 加载原始图像
    if image_path is None:
        base_img = np.full((target_size[0], target_size[1], 3), fallback_color, dtype=np.uint8)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"    ⚠️  Warning: Could not load image {image_path}, using fallback")
            base_img = np.full((target_size[0], target_size[1], 3), fallback_color, dtype=np.uint8)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize if necessary
            if img.shape[:2] != target_size:
                img = cv2.resize(img, (target_size[1], target_size[0]))

            base_img = img

    # 生成增强变体
    variants = []

    # 第一个变体：原始图像（无增强）
    variants.append(base_img.copy())

    # 生成增强变体
    if augmentation is not None and augmentation.enabled:
        for i in range(num_variants - 1):
            augmented_img = augmentation.augment_image(base_img)
            variants.append(augmented_img)
    else:
        # 如果没有增强，复制原始图像
        for i in range(num_variants - 1):
            variants.append(base_img.copy())

    return variants

def convert_episode_with_augmentation(episode_dir, output_dir, episode_name,
                                   global_counter, downsample_rate=4, segment_mode="gripper",
                                   target_length=300, remove_static=True,
                                   min_action_threshold=0.001, image_size=(480, 640),
                                   resample_mode="none", resample_length=None,
                                   augmentation_config=None, num_augmented_variants=3):
    """
    带光照增强的episode转换函数

    Args:
        augmentation_config: 光照增强配置
        num_augmented_variants: 每个原始episode生成的增强变体数量（包含原始）
    """

    print(f"🔄 Processing {episode_name} (生成 {num_augmented_variants} 个变体)...")

    # 初始化光照增强
    augmentation = None
    if augmentation_config:
        augmentation = LightingAugmentation(augmentation_config)

    # 加载episode数据
    data_file = os.path.join(episode_dir, 'data.json')
    if not os.path.exists(data_file):
        print(f"   ❌ No data.json found in {episode_dir}")
        return 0

    with open(data_file, 'r') as f:
        episode_data = json.load(f)

    data_points = episode_data.get('data', [])
    if not data_points:
        print(f"   ❌ No data points found in {episode_name}")
        return 0

    print(f"   📊 Original length: {len(data_points)} steps")

    # 预处理步骤（与原脚本相同）
    if remove_static:
        data_points = remove_static_frames(data_points, min_action_threshold)
        print(f"   🚀 After removing static frames: {len(data_points)} steps")

    if resample_mode == "exact" and resample_length:
        data_points = resample_episode(data_points, resample_length)
        print(f"   🎯 Resampled to exact length: {len(data_points)} steps")
    elif downsample_rate > 1 and resample_mode != "exact":
        data_points = downsample_episode(data_points, downsample_rate)
        print(f"   📉 After {downsample_rate}x downsampling: {len(data_points)} steps")

    if resample_mode == "exact":
        segments = [data_points]
        print(f"   📦 Using as single segment after exact resampling")
    else:
        segments = segment_episode(data_points, segment_mode, target_length)
        print(f"   ✂️  Generated {len(segments)} segments")

    converted_episodes = 0

    # 对每个segment生成多个增强变体
    for seg_idx, segment in enumerate(segments):
        if len(segment) < 10:
            continue

        # 为每个变体创建HDF5文件
        for variant_idx in range(num_augmented_variants):
            output_name = f"episode_{global_counter[0]}.hdf5"
            global_counter[0] += 1

            output_path = os.path.join(output_dir, output_name)

            print(f"     🔄 Converting segment {seg_idx + 1}/{len(segments)}, variant {variant_idx + 1}/{num_augmented_variants} ({len(segment)} steps)...")

            # 转换segment到HDF5，传入增强配置
            success = convert_segment_to_hdf5_augmented(
                segment, episode_dir, output_path, image_size,
                augmentation, variant_idx == 0  # 第一个变体是原始数据
            )

            if success:
                converted_episodes += 1
                print(f"     ✅ Saved to {output_name}")
            else:
                print(f"     ❌ Failed to convert segment {seg_idx}, variant {variant_idx}")

    return converted_episodes

def convert_segment_to_hdf5_augmented(data_points, episode_dir, output_path,
                                    image_size=(480, 640), augmentation=None,
                                    is_original=False):
    """
    将数据段转换为HDF5格式，支持光照增强

    Args:
        is_original: 是否为原始数据（不应用增强）
    """

    episode_len = len(data_points)

    # 准备数组
    qpos_array = []
    qvel_array = []
    action_array = []
    image_arrays = {'ee_cam': [], 'third_person_cam': [], 'side_cam': []}

    # 处理每个数据点
    for i, point in enumerate(data_points):
        # 提取关节状态（与原脚本相同）
        joint_states = point.get('joint_states', {})
        if 'single' not in joint_states:
            print(f"     ⚠️  No joint states at step {i}")
            return False

        positions = joint_states['single'].get('position', [])
        velocities = joint_states['single'].get('velocity', [])

        if len(positions) < 7 or len(velocities) < 7:
            print(f"     ⚠️  Insufficient joint data at step {i}")
            return False

        # 提取夹爪位置
        tools = point.get('tools', {})
        gripper_pos = 0.04  # Default
        if 'single' in tools and 'position' in tools['single']:
            gripper_pos = tools['single']['position']

        # 转换为ACT格式
        qpos, qvel = robot_to_act_joint_mapping(positions, velocities, gripper_pos)
        qpos_array.append(qpos)
        qvel_array.append(qvel)

        # 动作（使用下一个状态作为动作）
        if i < episode_len - 1:
            next_point = data_points[i + 1]
            next_joint_states = next_point.get('joint_states', {})
            if 'single' in next_joint_states:
                next_positions = next_joint_states['single'].get('position', positions)
                next_tools = next_point.get('tools', {})
                next_gripper_pos = gripper_pos
                if 'single' in next_tools and 'position' in next_tools['single']:
                    next_gripper_pos = next_tools['single']['position']

                action_qpos, _ = robot_to_act_joint_mapping(next_positions, velocities, next_gripper_pos)
                action_array.append(action_qpos)
            else:
                action_array.append(qpos)
        else:
            action_array.append(qpos)

        # 处理图像 - 关键改动：应用光照增强
        colors = point.get('colors', {})
        for cam_name in ['ee_cam_color', 'third_person_cam_color', 'side_cam_color']:
            act_cam_name = cam_name.replace('_color', '').replace('_cam', '_cam')
            if act_cam_name == 'ee_cam_cam':
                act_cam_name = 'ee_cam'
            elif act_cam_name == 'third_person_cam':
                act_cam_name = 'third_person_cam'
            elif act_cam_name == 'side_cam':
                act_cam_name = 'side_cam'

            if cam_name in colors and colors[cam_name] and 'path' in colors[cam_name]:
                img_path = os.path.join(episode_dir, colors[cam_name]['path'])

                # 加载图像
                img = cv2.imread(img_path)
                if img is None:
                    img = np.full((image_size[0], image_size[1], 3), 128, dtype=np.uint8)
                else:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize if necessary
                    if img.shape[:2] != image_size:
                        img = cv2.resize(img, (image_size[1], image_size[0]))

                # 应用光照增强（除非是原始变体）
                if not is_original and augmentation and augmentation.enabled:
                    img = augmentation.augment_image(img)

            else:
                img = np.full((image_size[0], image_size[1], 3), 128, dtype=np.uint8)

            image_arrays[act_cam_name].append(img)

    # 转换为numpy数组
    qpos_array = np.array(qpos_array, dtype=np.float32)
    qvel_array = np.array(qvel_array, dtype=np.float32)
    action_array = np.array(action_array, dtype=np.float32)

    for cam_name in image_arrays:
        image_arrays[cam_name] = np.array(image_arrays[cam_name])

    print(f"     📊 Converted arrays: qpos{qpos_array.shape}, actions{action_array.shape}")

    # 保存到HDF5
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('/observations/qpos', data=qpos_array)
        f.create_dataset('/observations/qvel', data=qvel_array)
        f.create_dataset('/action', data=action_array)

        for cam_name, images in image_arrays.items():
            f.create_dataset(f'/observations/images/{cam_name}',
                           data=images,
                           compression=None)

        # 元数据
        f.attrs['sim'] = False
        f.attrs['episode_length'] = episode_len

    return True

def load_augmentation_config(config_path):
    """加载光照增强配置文件"""
    if not os.path.exists(config_path):
        print(f"⚠️  配置文件不存在: {config_path}")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config.get('lighting_augmentation', {})

def main():
    parser = argparse.ArgumentParser(description='Preprocess and convert FR3 episodes to ACT format with lighting augmentation')

    # 原有参数
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing episode folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--downsample_rate', type=int, default=3,
                        help='Downsampling rate (default: 3, ignored if resample_mode is exact)')
    parser.add_argument('--segment_mode', type=str, default='none',
                        choices=['fixed', 'overlap', 'gripper', 'adaptive', 'none'],
                        help='Episode segmentation mode (overlap recommended)')
    parser.add_argument('--target_length', type=int, default=500,
                        help='Target length for segments (default: 500)')
    parser.add_argument('--remove_static_frames', action='store_true',
                        help='Remove frames where robot is stationary')
    parser.add_argument('--min_action_threshold', type=float, default=0.001,
                        help='Minimum action threshold for static frame detection')
    parser.add_argument('--image_size', type=str, default='480,640',
                        help='Image size as height,width (default: 480,640)')
    parser.add_argument('--resample_mode', type=str, default='exact',
                        choices=['none', 'exact'],
                        help='Resampling mode: none (no resampling), exact (resample entire episode)')
    parser.add_argument('--resample_length', type=int, default=None,
                        help='Target length for resampling (if not specified, uses target_length)')

    # 新增光照增强参数
    parser.add_argument('--augmentation_config', type=str,
                        default='/workspace/dependencies/act/configs/lighting_augmentation.yaml',
                        help='Path to lighting augmentation config file')
    parser.add_argument('--num_variants', type=int, default=3,
                        help='Number of augmented variants to generate per episode (including original)')
    parser.add_argument('--disable_augmentation', action='store_true',
                        help='Disable lighting augmentation')

    args = parser.parse_args()

    # 解析图像尺寸
    image_size = tuple(map(int, args.image_size.split(',')))

    # 确定重采样长度
    resample_length = args.resample_length if args.resample_length else args.target_length

    # 加载增强配置
    augmentation_config = None
    if not args.disable_augmentation:
        augmentation_config = load_augmentation_config(args.augmentation_config)
        if augmentation_config is None:
            print("⚠️  无法加载增强配置，将禁用光照增强")

    print(f"🚀 Starting FR3 episode preprocessing with lighting augmentation...")
    print(f"   📁 Input: {args.input_dir}")
    print(f"   📁 Output: {args.output_dir}")
    if args.resample_mode != 'exact':
        print(f"   📉 Downsample rate: {args.downsample_rate}")
    print(f"   ✂️  Segment mode: {args.segment_mode}")
    print(f"   📏 Target length: {args.target_length}")
    print(f"   🔄 Resample mode: {args.resample_mode}")
    if args.resample_mode != 'none':
        print(f"   🎯 Resample length: {resample_length}")
    print(f"   🚀 Remove static: {args.remove_static_frames}")
    print(f"   🖼️  Image size: {image_size}")
    print(f"   🌈 Augmentation: {'Enabled' if augmentation_config else 'Disabled'}")
    print(f"   🔢 Variants per episode: {args.num_variants}")
    print()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取episode目录
    episode_dirs = [d for d in os.listdir(args.input_dir)
                    if d.startswith('episode_') and
                    os.path.isdir(os.path.join(args.input_dir, d))]
    episode_dirs.sort()

    if not episode_dirs:
        print("❌ No episode directories found!")
        return

    print(f"📊 Found {len(episode_dirs)} episodes to process")
    if augmentation_config:
        expected_output = len(episode_dirs) * args.num_variants
        print(f"📊 Expected output episodes: {expected_output} (每个原始episode生成{args.num_variants}个变体)")
    print()

    total_converted = 0
    global_counter = [0]
    start_time = time.time()

    for episode_name in episode_dirs:
        episode_dir = os.path.join(args.input_dir, episode_name)

        converted = convert_episode_with_augmentation(
            episode_dir, args.output_dir, episode_name,
            global_counter,
            downsample_rate=args.downsample_rate,
            segment_mode=args.segment_mode,
            target_length=args.target_length,
            remove_static=args.remove_static_frames,
            min_action_threshold=args.min_action_threshold,
            image_size=image_size,
            resample_mode=args.resample_mode,
            resample_length=resample_length,
            augmentation_config=augmentation_config,
            num_augmented_variants=args.num_variants
        )

        total_converted += converted
        print()

    elapsed_time = time.time() - start_time
    print(f"✅ Preprocessing with augmentation completed!")
    print(f"   📊 Original episodes: {len(episode_dirs)}")
    print(f"   📊 Generated HDF5 files: {total_converted}")
    if augmentation_config:
        print(f"   🌈 Augmentation ratio: {total_converted / len(episode_dirs):.1f}x")
    print(f"   ⏱️  Total time: {elapsed_time:.1f}s")
    print(f"   📁 Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()