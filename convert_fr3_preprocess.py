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

def load_and_resize_image_robust(image_path, target_size=(480, 640), fallback_color=(128, 128, 128)):
    """Load image and resize, with robust error handling"""
    # Handle None path by returning fallback
    if image_path is None:
        img = np.full((target_size[0], target_size[1], 3), fallback_color, dtype=np.uint8)
        return img

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"    ⚠️  Warning: Could not load image {image_path}, using fallback")
            # Create a gray fallback image
            img = np.full((target_size[0], target_size[1], 3), fallback_color, dtype=np.uint8)
            return img

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize if necessary
        if img.shape[:2] != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))

        return img
    except Exception as e:
        print(f"    ⚠️  Exception loading {image_path}: {e}, using fallback")
        # Create a fallback image
        img = np.full((target_size[0], target_size[1], 3), fallback_color, dtype=np.uint8)
        return img

def robot_to_act_joint_mapping(joint_pos, joint_vel, gripper_pos, is_dual_arm=False,
                            right_joint_pos=None, right_joint_vel=None, right_gripper_pos=None):
    """Map robot joints to ACT action space - supports both single-arm and dual-arm
    Single-arm (FR3): 8 DOF with gripper range 0-0.08m
    Dual-arm (Monte01): 16 DOF with gripper range 0-0.074m
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

        # Gripper (1 DOF) - normalize from 0-0.08m to 0-1
        act_qpos[7] = gripper_pos / 0.08
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

    elif segment_mode == "gripper":
        # Segment based on gripper state changes
        gripper_positions = []
        for point in data_points:
            try:
                tools = point.get('tools', {})
                if 'single' in tools and 'position' in tools['single']:
                    gripper_positions.append(tools['single']['position'])
                else:
                    gripper_positions.append(0.04)  # Default position
            except:
                gripper_positions.append(0.04)

        change_points = detect_gripper_state_changes(gripper_positions)

        if not change_points:
            # No significant changes, use fixed segmentation
            return segment_episode(data_points, "fixed", target_length)

        segments = []
        start = 0
        for change_point in change_points:
            if change_point - start >= target_length // 3:  # Minimum segment length
                segments.append(data_points[start:change_point + 1])
                start = change_point

        # Add final segment
        if len(data_points) - start >= target_length // 3:
            segments.append(data_points[start:])

        return segments

    else:  # "none" or invalid mode
        return [data_points]

def convert_episode_preprocessing(episode_dir, output_dir, episode_name,
                               downsample_rate=4, segment_mode="gripper",
                               target_length=300, remove_static=True,
                               min_action_threshold=0.001, image_size=(480, 640)):
    """Convert a single episode with preprocessing options"""

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

    print(f"   📊 Original length: {len(data_points)} steps")

    # Step 1: Remove static frames if requested
    if remove_static:
        data_points = remove_static_frames(data_points, min_action_threshold)
        print(f"   🚀 After removing static frames: {len(data_points)} steps")

    # Step 2: Downsample
    if downsample_rate > 1:
        data_points = downsample_episode(data_points, downsample_rate)
        print(f"   📉 After {downsample_rate}x downsampling: {len(data_points)} steps")

    # Step 3: Segment into shorter episodes
    segments = segment_episode(data_points, segment_mode, target_length)
    print(f"   ✂️  Generated {len(segments)} segments")

    converted_episodes = 0

    for seg_idx, segment in enumerate(segments):
        if len(segment) < 10:  # Skip very short segments
            continue

        # Create output filename
        if len(segments) > 1:
            output_name = f"{episode_name}_seg{seg_idx:02d}.hdf5"
        else:
            output_name = f"{episode_name}.hdf5"

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

                if cam_name in colors and colors[cam_name] and 'path' in colors[cam_name]:
                    img_path = os.path.join(episode_dir, colors[cam_name]['path'])
                    img = load_and_resize_image_robust(img_path, image_size)
                else:
                    img = np.full((image_size[0], image_size[1], 3), 128, dtype=np.uint8)

                image_arrays[act_cam_name].append(img)

        except Exception as e:
            print(f"     ⚠️  Error processing step {i}: {e}")
            return False

    # Convert to numpy arrays
    qpos_array = np.array(qpos_array)
    qvel_array = np.array(qvel_array)
    action_array = np.array(action_array)

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
                f.create_dataset(f'/observations/images/{cam_name}', data=images, compression='gzip')

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
    parser.add_argument('--downsample_rate', type=int, default=4,
                        help='Downsampling rate (default: 4)')
    parser.add_argument('--segment_mode', type=str, default='gripper',
                        choices=['fixed', 'gripper', 'none'],
                        help='Episode segmentation mode')
    parser.add_argument('--target_length', type=int, default=300,
                        help='Target length for segments (default: 300)')
    parser.add_argument('--remove_static_frames', action='store_true',
                        help='Remove frames where robot is stationary')
    parser.add_argument('--min_action_threshold', type=float, default=0.001,
                        help='Minimum action threshold for static frame detection')
    parser.add_argument('--image_size', type=str, default='480,640',
                        help='Image size as height,width (default: 480,640)')

    args = parser.parse_args()

    # Parse image size
    image_size = tuple(map(int, args.image_size.split(',')))

    print(f"🚀 Starting FR3 episode preprocessing...")
    print(f"   📁 Input: {args.input_dir}")
    print(f"   📁 Output: {args.output_dir}")
    print(f"   📉 Downsample rate: {args.downsample_rate}")
    print(f"   ✂️  Segment mode: {args.segment_mode}")
    print(f"   📏 Target length: {args.target_length}")
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
    start_time = time.time()

    for episode_name in episode_dirs:
        episode_dir = os.path.join(args.input_dir, episode_name)

        converted = convert_episode_preprocessing(
            episode_dir, args.output_dir, episode_name,
            downsample_rate=args.downsample_rate,
            segment_mode=args.segment_mode,
            target_length=args.target_length,
            remove_static=args.remove_static_frames,
            min_action_threshold=args.min_action_threshold,
            image_size=image_size
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