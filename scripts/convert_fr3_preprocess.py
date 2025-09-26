#!/usr/bin/env python3

import os
import json
import numpy as np
import h5py
import cv2
import argparse
from pathlib import Path
import time
from collections import deque
import copy

def load_and_resize_image_robust(image_path, target_size=(480, 640), fallback_color=(128, 128, 128)):
    """Load image and resize, with robust error handling"""
    # Handle None path by returning fallback
    if image_path is None:
        return False, None

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"    ⚠️  Warning: Could not load image {image_path}")
            return False, None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize if necessary
        if img.shape[:2] != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))

        return True, img
    except Exception as e:
        print(f"    ⚠️  Exception loading {image_path}: {e}")
        # Create a fallback image
        return False, None

def robot_to_act_joint_mapping(joint_pos, joint_vel, gripper_pos, is_dual_arm=False,
                            right_joint_pos=None, right_joint_vel=None, right_gripper_pos=None):
    """Map robot joints to ACT action space - supports both single-arm and dual-arm
    Single-arm (FR3): 8 DOF with gripper original data
    Dual-arm (Monte01): 16 DOF with gripper original data
    """

    if is_dual_arm and right_joint_pos is not None:
        # Dual-arm Monte01: 16 DOF total (7+1 left, 7+1 right)
        act_qpos = np.zeros(16)
        act_qvel = np.zeros(16)

        # Left arm (7 DOF) + gripper
        act_qpos[:7] = joint_pos[:7]
        act_qvel[:7] = joint_vel[:7]
        act_qpos[7] = gripper_pos / 0.074  # Normalize gripper to 0-1
        act_qvel[7] = 0.0

        # Right arm (7 DOF) + gripper
        act_qpos[8:15] = right_joint_pos[:7]
        act_qvel[8:15] = right_joint_vel[:7]
        act_qpos[15] = right_gripper_pos / 0.074  # Normalize gripper to 0-1
        act_qvel[15] = 0.0

    else:
        # Single-arm FR3: 8 DOF (7 joints + 1 gripper)
        act_qpos = np.zeros(8)
        act_qvel = np.zeros(8)

        # Arm joints (7 DOF)
        act_qpos[:7] = joint_pos[:7]
        act_qvel[:7] = joint_vel[:7]

        act_qpos[7] = gripper_pos
        act_qvel[7] = 0.0

    return act_qpos, act_qvel

def detect_gripper_state_changes(gripper_positions, threshold=0.005):
    """Detect significant gripper state changes for episode segmentation"""
    changes = []
    if len(gripper_positions) < 10:
        return changes

    # Smooth gripper positions to reduce noise
    smoothed = np.convolve(gripper_positions, np.ones(5)/5, mode='same')

    # Find significant changes
    diff = np.abs(np.diff(smoothed))
    change_indices = np.where(diff > threshold)[0]

    # Group nearby changes
    if len(change_indices) > 0:
        grouped_changes = [change_indices[0]]
        for idx in change_indices[1:]:
            if idx - grouped_changes[-1] > 20:  # Minimum gap between changes
                grouped_changes.append(idx)
        changes = grouped_changes

    return changes

def remove_static_frames(data_points, min_action_threshold=0.001):
    """Remove frames where the robot is mostly stationary"""
    if len(data_points) < 2:
        return data_points

    dynamic_indices = [0]  # Always keep first frame

    prev_joints = None
    for i, point in enumerate(data_points):
        try:
            # Get current joint positions
            joint_states = point.get('joint_states', {})
            if 'single' in joint_states and 'position' in joint_states['single']:
                current_joints = np.array(joint_states['single']['position'][:7])

                if prev_joints is not None:
                    # Calculate joint movement
                    movement = np.linalg.norm(current_joints - prev_joints)
                    if movement > min_action_threshold:
                        dynamic_indices.append(i)
                else:
                    dynamic_indices.append(i)

                prev_joints = current_joints
            else:
                dynamic_indices.append(i)  # Keep frame if we can't analyze it

        except Exception as e:
            dynamic_indices.append(i)  # Keep frame on error

    # Always keep last frame
    if len(data_points) - 1 not in dynamic_indices:
        dynamic_indices.append(len(data_points) - 1)

    return [data_points[i] for i in sorted(dynamic_indices)]

def downsample_episode(data_points, downsample_rate=4):
    """Downsample episode by taking every nth frame"""
    if downsample_rate <= 1:
        return data_points

    return data_points[::downsample_rate]

def resample_episode(data_points, target_length):
    """Resample episode to exact target length using linear interpolation

    Args:
        data_points: List of data points from original episode
        target_length: Target number of frames

    Returns:
        List of resampled data points with exact target_length
    """
    if not data_points or target_length <= 0:
        return []

    current_length = len(data_points)

    # If already at target length, return as is
    if current_length == target_length:
        return data_points

    # If only one frame, duplicate it
    if current_length == 1:
        return [data_points[0]] * target_length

    resampled = []

    # Calculate indices for resampling
    for i in range(target_length):
        # Map target index to source index (floating point)
        src_idx = i * (current_length - 1) / (target_length - 1)

        # Get integer indices for interpolation
        idx_low = int(np.floor(src_idx))
        idx_high = min(int(np.ceil(src_idx)), current_length - 1)

        # If indices are same (exact match), no interpolation needed
        if idx_low == idx_high:
            resampled.append(copy.deepcopy(data_points[idx_low]))
        else:
            # Linear interpolation weight
            weight = src_idx - idx_low

            # Interpolate data point
            interpolated_point = interpolate_data_points(
                data_points[idx_low],
                data_points[idx_high],
                weight
            )
            resampled.append(interpolated_point)

    return resampled

def interpolate_data_points(point1, point2, weight):
    """Linearly interpolate between two data points

    Args:
        point1: First data point (weight = 0)
        point2: Second data point (weight = 1)
        weight: Interpolation weight (0 to 1)

    Returns:
        Interpolated data point
    """
    interpolated = {}

    # Interpolate joint states
    if 'joint_states' in point1 and 'joint_states' in point2:
        interpolated['joint_states'] = {}
        for arm_key in point1['joint_states']:
            if arm_key in point2['joint_states']:
                interpolated['joint_states'][arm_key] = {}

                # Interpolate positions
                if 'position' in point1['joint_states'][arm_key] and 'position' in point2['joint_states'][arm_key]:
                    pos1 = np.array(point1['joint_states'][arm_key]['position'])
                    pos2 = np.array(point2['joint_states'][arm_key]['position'])
                    interpolated['joint_states'][arm_key]['position'] = (
                        (1 - weight) * pos1 + weight * pos2
                    ).tolist()

                # Interpolate velocities
                if 'velocity' in point1['joint_states'][arm_key] and 'velocity' in point2['joint_states'][arm_key]:
                    vel1 = np.array(point1['joint_states'][arm_key]['velocity'])
                    vel2 = np.array(point2['joint_states'][arm_key]['velocity'])
                    interpolated['joint_states'][arm_key]['velocity'] = (
                        (1 - weight) * vel1 + weight * vel2
                    ).tolist()

    # Interpolate tool states (gripper)
    if 'tools' in point1 and 'tools' in point2:
        interpolated['tools'] = {}
        for tool_key in point1['tools']:
            if tool_key in point2['tools']:
                interpolated['tools'][tool_key] = {}

                # Interpolate gripper position
                if 'position' in point1['tools'][tool_key] and 'position' in point2['tools'][tool_key]:
                    gripper1 = point1['tools'][tool_key]['position']
                    gripper2 = point2['tools'][tool_key]['position']
                    interpolated['tools'][tool_key]['position'] = (
                        (1 - weight) * gripper1 + weight * gripper2
                    )

    # For images, use nearest neighbor (don't interpolate pixel values)
    # Take the closest frame's images
    if 'colors' in point1 and 'colors' in point2:
        if weight < 0.5:
            interpolated['colors'] = point1['colors']
        else:
            interpolated['colors'] = point2['colors']

    # Copy metadata from nearest point
    if weight < 0.5:
        base_point = point1
    else:
        base_point = point2

    # Copy other fields that we don't interpolate
    for key in base_point:
        if key not in ['joint_states', 'tools', 'colors']:
            interpolated[key] = base_point[key]

    return interpolated

def segment_episode(data_points, segment_mode="fixed", target_length=300):
    """Segment long episodes into shorter ones"""
    if segment_mode == "fixed":
        # Split into fixed-length segments
        segments = []
        for start in range(0, len(data_points), target_length):
            end = min(start + target_length, len(data_points))
            if end - start >= target_length // 2:  # Keep segments with at least half target length
                segments.append(data_points[start:end])
        return segments

    elif segment_mode == "overlap":
        # 使用重叠窗口分割，保证动作连续性
        return segment_with_overlap(data_points, window_size=target_length, overlap_ratio=0.2)

    elif segment_mode == "gripper":
        # DEPRECATED: 基于夹爪变化点分割会破坏动作完整性
        # 改为使用动作簇分割
        return segment_by_action_clusters(data_points, min_segment_length=target_length//2, max_segment_length=target_length*2)

    elif segment_mode == "adaptive":
        # 基于动作密度的自适应分割
        return segment_adaptive(data_points, preferred_length=target_length)

    else:  # "none" or invalid mode
        return [data_points]


def segment_with_overlap(data_points, window_size=400, overlap_ratio=0.2):
    """重叠窗口分割，确保动作连续性"""
    if len(data_points) <= window_size:
        return [data_points] if len(data_points) >= window_size // 2 else []

    segments = []
    overlap_frames = int(window_size * overlap_ratio)
    step_size = window_size - overlap_frames

    for start in range(0, len(data_points) - window_size // 2, step_size):
        end = min(start + window_size, len(data_points))
        segments.append(data_points[start:end])
        if end == len(data_points):
            break

    return segments


def segment_by_action_clusters(data_points, min_segment_length=200, max_segment_length=800):
    """基于动作簇分割，保留完整的pick-and-place周期"""
    gripper_positions = []
    for point in data_points:
        try:
            tools = point.get('tools', {})
            if 'single' in tools and 'position' in tools['single']:
                gripper_positions.append(tools['single']['position'])
            else:
                gripper_positions.append(0.04)
        except:
            gripper_positions.append(0.04)

    # 检测动作簇（完整的开-闭-开周期）
    clusters = detect_action_clusters(gripper_positions)

    segments = []
    for cluster_start, cluster_end in clusters:
        # 在簇前后添加缓冲
        buffer = 30  # 前后各30帧缓冲
        start = max(0, cluster_start - buffer)
        end = min(len(data_points), cluster_end + buffer)

        if end - start > max_segment_length:
            # 太长则使用重叠分割
            sub_segments = segment_with_overlap(data_points[start:end], window_size=max_segment_length)
            segments.extend(sub_segments)
        elif end - start >= min_segment_length:
            segments.append(data_points[start:end])

    if not segments:  # 如果没有检测到有效簇，回退到固定分割
        return segment_with_overlap(data_points, window_size=(min_segment_length + max_segment_length) // 2)

    return segments


def detect_action_clusters(gripper_positions, min_cluster_gap=50):
    """检测夹爪动作簇"""
    if len(gripper_positions) < 10:
        return [(0, len(gripper_positions) - 1)]

    # 计算夹爪变化速度
    gripper_velocity = np.abs(np.diff(gripper_positions))

    # 找出活动帧
    activity_threshold = 0.002
    active_frames = np.where(gripper_velocity > activity_threshold)[0]

    if len(active_frames) == 0:
        return [(0, len(gripper_positions) - 1)]

    # 将活动帧聚类
    clusters = []
    cluster_start = active_frames[0]

    for i in range(1, len(active_frames)):
        if active_frames[i] - active_frames[i-1] > min_cluster_gap:
            clusters.append((cluster_start, active_frames[i-1]))
            cluster_start = active_frames[i]

    clusters.append((cluster_start, active_frames[-1]))
    return clusters


def segment_adaptive(data_points, preferred_length=400, variance=0.3):
    """基于动作密度的自适应分割"""
    if len(data_points) <= preferred_length:
        return [data_points]

    # 计算动作密度
    density = calculate_action_density(data_points)

    segments = []
    current_start = 0

    while current_start < len(data_points):
        min_len = int(preferred_length * (1 - variance))
        max_len = int(preferred_length * (1 + variance))

        search_start = min(current_start + min_len, len(data_points) - 1)
        search_end = min(current_start + max_len, len(data_points))

        if search_start >= search_end:
            segments.append(data_points[current_start:])
            break

        # 在低密度点分割
        window_density = density[search_start:search_end]
        if len(window_density) > 0:
            best_split = search_start + np.argmin(window_density)
            segments.append(data_points[current_start:best_split])
            current_start = best_split
        else:
            segments.append(data_points[current_start:])
            break

    return segments


def calculate_action_density(data_points, window=10):
    """计算动作密度"""
    density = np.zeros(len(data_points))

    for i in range(len(data_points)):
        window_start = max(0, i - window // 2)
        window_end = min(len(data_points), i + window // 2)

        total_movement = 0
        for j in range(window_start + 1, window_end):
            try:
                prev = np.array(data_points[j-1]['joint_states']['single']['position'][:7])
                curr = np.array(data_points[j]['joint_states']['single']['position'][:7])
                total_movement += np.linalg.norm(curr - prev)
            except:
                continue

        density[i] = total_movement / max(1, window_end - window_start)

    return density

def convert_episode_preprocessing(episode_dir, output_dir, episode_name,
                               global_counter, downsample_rate=4, segment_mode="gripper",
                               target_length=300, remove_static=True,
                               min_action_threshold=0.001, image_size=(480, 640),
                               resample_mode="none", resample_length=None):
    """Convert a single episode with preprocessing options

    Args:
        episode_dir: Directory containing episode data
        output_dir: Output directory for HDF5 files
        episode_name: Name of the episode
        global_counter: Global counter for naming output files
        downsample_rate: Rate for downsampling (default: 4)
        segment_mode: Mode for segmenting episodes
        target_length: Target length for segments
        remove_static: Whether to remove static frames
        min_action_threshold: Threshold for static frame detection
        image_size: Target image size tuple
        resample_mode: Mode for resampling ('none', 'exact')
        resample_length: Target length for resampling (if None, uses target_length)
    """

    print(f"🔄 Processing {episode_name}...")

    # Load episode data
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

    epilen = len(data_points)
    print(f"   📊 Original length: {epilen} steps")
    if epilen < 200 or epilen > 1000:
        print(f"   ⚠️  Warning: Unusual episode length ({epilen} steps), PASS...")
        return 0

    # Step 1: Remove static frames if requested
    if remove_static:
        data_points = remove_static_frames(data_points, min_action_threshold)
        print(f"   🚀 After removing static frames: {len(data_points)} steps")

    # Step 2: Apply resampling if mode is 'exact' (before segmentation)
    if resample_mode == "exact" and resample_length:
        data_points = resample_episode(data_points, resample_length)
        print(f"   🎯 Resampled to exact length: {len(data_points)} steps")
    # Step 2b: Otherwise apply downsampling
    elif downsample_rate > 1 and resample_mode != "exact":
        data_points = downsample_episode(data_points, downsample_rate)
        print(f"   📉 After {downsample_rate}x downsampling: {len(data_points)} steps")

    # Step 3: Segment into shorter episodes
    if resample_mode == "exact":
        # If exact resampling was done, treat as single segment
        segments = [data_points]
        print(f"   📦 Using as single segment after exact resampling")
    else:
        segments = segment_episode(data_points, segment_mode, target_length)
        print(f"   ✂️  Generated {len(segments)} segments")

    converted_episodes = 0

    for seg_idx, segment in enumerate(segments):
        if len(segment) < 10:  # Skip very short segments
            continue

        # Create output filename using global counter
        output_name = f"episode_{global_counter[0]}.hdf5"
        global_counter[0] += 1
        
        output_path = os.path.join(output_dir, output_name)

        print(f"     🔄 Converting segment {seg_idx + 1}/{len(segments)} ({len(segment)} steps)...")

        try:
            # Convert segment to HDF5
            success = convert_segment_to_hdf5(segment, episode_dir, output_path, image_size)
            if success:
                converted_episodes += 1
                print(f"     ✅ Saved to {output_name}")
            else:
                print(f"     ❌ Failed to convert segment {seg_idx}")

        except Exception as e:
            print(f"     ❌ Error converting segment {seg_idx}: {e}")

    return converted_episodes

def convert_segment_to_hdf5(data_points, episode_dir, output_path, image_size=(480, 640)):
    """Convert a data segment to HDF5 format"""

    episode_len = len(data_points)

    # Prepare arrays
    qpos_array = []
    qvel_array = []
    action_array = []
    image_arrays = {'ee_cam': [], 'third_person_cam': [], 'side_cam': []}

    # Process each data point
    for i, point in enumerate(data_points):
        try:
            # Extract joint states
            joint_states = point.get('joint_states', {})
            if 'single' not in joint_states:
                print(f"     ⚠️  No joint states at step {i}")
                return False

            positions = joint_states['single'].get('position', [])
            velocities = joint_states['single'].get('velocity', [])

            if len(positions) < 7 or len(velocities) < 7:
                print(f"     ⚠️  Insufficient joint data at step {i}")
                return False

            # Extract gripper position
            tools = point.get('tools', {})
            gripper_pos = 0.04  # Default
            if 'single' in tools and 'position' in tools['single']:
                gripper_pos = tools['single']['position']

            # Convert to ACT format
            qpos, qvel = robot_to_act_joint_mapping(positions, velocities, gripper_pos)
            qpos_array.append(qpos)
            qvel_array.append(qvel)

            # Actions (use next state as action, or current for last step)
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
                    action_array.append(qpos)  # Fallback
            else:
                action_array.append(qpos)  # Last action = current state

            # Process images
            colors = point.get('colors', {})
            for cam_name in ['ee_cam_color', 'third_person_cam_color', 'side_cam_color']:
                act_cam_name = cam_name.replace('_color', '').replace('_cam', '_cam')
                if act_cam_name == 'ee_cam_cam':
                    act_cam_name = 'ee_cam'
                elif act_cam_name == 'third_person_cam':
                    act_cam_name = 'third_person_cam'
                elif act_cam_name == 'side_cam':
                    act_cam_name = 'side_cam'

                success = False
                if cam_name in colors and colors[cam_name] and 'path' in colors[cam_name]:
                    img_path = os.path.join(episode_dir, colors[cam_name]['path'])
                    success, img = load_and_resize_image_robust(img_path, image_size)

                if success:
                    image_arrays[act_cam_name].append(img)

        except Exception as e:
            print(f"     ⚠️  Error processing step {i}: {e}")
            return False

    # Convert to numpy arrays
    qpos_array = np.array(qpos_array, dtype=np.float32)
    qvel_array = np.array(qvel_array, dtype=np.float32)
    action_array = np.array(action_array, dtype=np.float32)

    for cam_name in image_arrays:
        image_arrays[cam_name] = np.array(image_arrays[cam_name])

    print(f"     📊 Converted arrays: qpos{qpos_array.shape}, actions{action_array.shape}")

    # Save to HDF5
    try:
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('/observations/qpos', data=qpos_array)
            f.create_dataset('/observations/qvel', data=qvel_array)
            f.create_dataset('/action', data=action_array)

            for cam_name, images in image_arrays.items():
                f.create_dataset(f'/observations/images/{cam_name}',
                               data=images,
                               compression=None)

            # Metadata
            f.attrs['sim'] = False
            f.attrs['episode_length'] = episode_len

        return True

    except Exception as e:
        print(f"     ❌ HDF5 save error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Preprocess and convert FR3 episodes to ACT format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing episode folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--downsample_rate', type=int, default=2,
                        help='Downsampling rate (default: 2, ignored if resample_mode is exact)')
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
    parser.add_argument('--resample_mode', type=str, default='none',
                        choices=['none', 'exact'],
                        help='Resampling mode: none (no resampling), exact (resample entire episode),')
    parser.add_argument('--resample_length', type=int, default=None,
                        help='Target length for resampling (if not specified, uses target_length)')

    args = parser.parse_args()

    # Parse image size
    image_size = tuple(map(int, args.image_size.split(',')))

    # Determine resample length
    resample_length = args.resample_length if args.resample_length else args.target_length

    print(f"🚀 Starting FR3 episode preprocessing...")
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
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get episode directories
    episode_dirs = [d for d in os.listdir(args.input_dir)
                    if d.startswith('episode_') and
                    os.path.isdir(os.path.join(args.input_dir, d))]
    episode_dirs.sort()

    if not episode_dirs:
        print("❌ No episode directories found!")
        return

    print(f"📊 Found {len(episode_dirs)} episodes to process")
    print()

    total_converted = 0
    global_counter = [0]  # Use list to allow modification in function
    start_time = time.time()

    for episode_name in episode_dirs:
        episode_dir = os.path.join(args.input_dir, episode_name)

        converted = convert_episode_preprocessing(
            episode_dir, args.output_dir, episode_name,
            global_counter,  # Pass global counter
            downsample_rate=args.downsample_rate,
            segment_mode=args.segment_mode,
            target_length=args.target_length,
            remove_static=args.remove_static_frames,
            min_action_threshold=args.min_action_threshold,
            image_size=image_size,
            resample_mode=args.resample_mode,
            resample_length=resample_length
        )

        total_converted += converted
        print()

    elapsed_time = time.time() - start_time
    print(f"✅ Preprocessing completed!")
    print(f"   📊 Original episodes: {len(episode_dirs)}")
    print(f"   📊 Generated HDF5 files: {total_converted}")
    print(f"   ⏱️  Total time: {elapsed_time:.1f}s")
    print(f"   📁 Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
