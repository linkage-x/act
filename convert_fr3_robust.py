#!/usr/bin/env python3

import os
import json
import numpy as np
import h5py
import cv2
import argparse
from pathlib import Path
import time

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
        
        # Left arm mapping (first 8 DOF)
        left_arm_joints = np.array(joint_pos[:7])
        left_arm_velocities = np.array(joint_vel[:7])
        act_qpos[0:7] = left_arm_joints[0:7]
        act_qvel[0:7] = left_arm_velocities[0:7]
        
        # Left gripper (Monte01 range: 0-0.074m)
        left_gripper_normalized = np.clip(gripper_pos, 0.0, 0.074)
        act_qpos[7] = left_gripper_normalized
        act_qvel[7] = 0.0
        
        # Right arm mapping (second 8 DOF)
        right_arm_joints = np.array(right_joint_pos[:7])
        right_arm_velocities = np.array(right_joint_vel[:7])
        act_qpos[8:15] = right_arm_joints[0:7]
        act_qvel[8:15] = right_arm_velocities[0:7]
        
        # Right gripper (Monte01 range: 0-0.074m)
        right_gripper_normalized = np.clip(right_gripper_pos, 0.0, 0.074)
        act_qpos[15] = right_gripper_normalized
        act_qvel[15] = 0.0
        
    else:
        # Single-arm FR3: 8 DOF total (7 arm + 1 gripper)
        arm_joints = np.array(joint_pos[:7])
        arm_velocities = np.array(joint_vel[:7])
        
        act_qpos = np.zeros(8)
        act_qvel = np.zeros(8)
        
        # Direct mapping: 7 arm joints -> ACT 7 arm joints
        act_qpos[0:7] = arm_joints[0:7]
        act_qvel[0:7] = arm_velocities[0:7]
        
        # Gripper mapping (FR3 range: 0-0.08m)
        gripper_normalized = np.clip(gripper_pos, 0.0, 0.08)
        act_qpos[7] = gripper_normalized
        act_qvel[7] = 0.0
    
    return act_qpos, act_qvel

def convert_robot_episode_robust(episode_dir, output_dir, episode_idx):
    """Convert robot episode to ACT HDF5 format with robust error handling
    Automatically detects single-arm (FR3) vs dual-arm (Monte01) configuration
    """
    
    print(f"Converting episode {episode_idx:04d}...")
    
    # Check if episode data exists
    data_file = os.path.join(episode_dir, 'data.json')
    colors_dir = os.path.join(episode_dir, 'colors')
    
    if not os.path.exists(data_file):
        print(f"  ❌ Missing data.json for episode {episode_idx}")
        return None
    if not os.path.exists(colors_dir):
        print(f"  ❌ Missing colors directory for episode {episode_idx}")
        return None
    
    try:
        # Load FR3 data
        with open(data_file, 'r') as f:
            episode_data = json.load(f)
        
        data_points = episode_data['data']
        max_timesteps = len(data_points)
        
        # Detect if this is dual-arm (Monte01) or single-arm (FR3) data
        first_point = data_points[0] if data_points else {}
        has_right_arm = 'right' in first_point.get('joint_states', {})
        is_dual_arm = has_right_arm and 'left' in first_point.get('joint_states', {})
        
        if is_dual_arm:
            print(f"  Detected dual-arm Monte01 robot configuration")
            dof = 16  # 7+1 for each arm
            robot_type = 'monte01'
        else:
            print(f"  Detected single-arm FR3 robot configuration") 
            dof = 8  # 7+1 for single arm
            robot_type = 'fr3'
        
        # Pre-check for corrupt data points
        valid_indices = []
        corrupt_count = 0
        
        for i, data_point in enumerate(data_points):
            try:
                # Check if joint states exist - handle both single and left/right robots
                if 'single' in data_point.get('joint_states', {}):
                    joint_states = data_point['joint_states']['single']
                elif 'left' in data_point.get('joint_states', {}):
                    # Use left arm for single-arm ACT format
                    joint_states = data_point['joint_states']['left']
                else:
                    raise KeyError("No valid joint states found")
                    
                joint_pos = joint_states['position']
                joint_vel = joint_states['velocitie']  # Note: typo in original
                
                # Handle gripper position
                if 'tools' in data_point and 'single' in data_point['tools']:
                    gripper_pos = data_point['tools']['single']['position']
                elif 'tools' in data_point and 'left' in data_point['tools']:
                    gripper_pos = data_point['tools']['left']['position']
                else:
                    # Default gripper position if not found
                    gripper_pos = 0.04
                
                # Check if image paths exist - handle both left/right and single ee_cam
                colors = data_point.get('colors', {})
                
                # Try different camera naming conventions
                if 'left_ee_cam_color' in colors:
                    # Use left camera as primary ee_cam
                    ee_cam_path = os.path.join(episode_dir, colors['left_ee_cam_color'])
                elif 'ee_cam_color' in colors:
                    ee_cam_path = os.path.join(episode_dir, colors['ee_cam_color'])
                else:
                    ee_cam_path = None
                    
                if 'third_person_cam_color' in colors:
                    third_person_path = os.path.join(episode_dir, colors['third_person_cam_color'])
                else:
                    third_person_path = None
                
                if len(joint_pos) >= 7 and len(joint_vel) >= 7:
                    valid_indices.append(i)
                else:
                    corrupt_count += 1
                    
            except (KeyError, IndexError, TypeError) as e:
                corrupt_count += 1
                continue
        
        if len(valid_indices) < 10:  # Need at least 10 valid frames
            print(f"  ❌ Episode {episode_idx} has too few valid frames ({len(valid_indices)})")
            return None
        
        if corrupt_count > 0:
            print(f"  ⚠️  Episode {episode_idx}: {corrupt_count} corrupt frames, using {len(valid_indices)} valid frames")
        
        # Prepare data structures
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
            '/observations/images/ee_cam': [],
            '/observations/images/third_person_cam': []
        }
        
        # Add right camera for dual-arm
        if is_dual_arm:
            data_dict['/observations/images/right_ee_cam'] = []
        
        print(f"  Processing {len(valid_indices)} valid timesteps...")
        
        successful_frames = 0
        for idx_i, i in enumerate(valid_indices):
            if idx_i % 200 == 0:
                print(f"    Progress: {idx_i}/{len(valid_indices)} ({idx_i/len(valid_indices)*100:.1f}%)")
            
            try:
                data_point = data_points[i]
                
                # Extract joint states - handle both single and left/right robots
                if 'single' in data_point.get('joint_states', {}):
                    joint_states = data_point['joint_states']['single']
                elif 'left' in data_point.get('joint_states', {}):
                    # Use left arm for single-arm ACT format
                    joint_states = data_point['joint_states']['left']
                else:
                    raise KeyError("No valid joint states found")
                    
                joint_pos = joint_states['position']
                joint_vel = joint_states['velocitie']
                
                # Handle gripper position
                if 'tools' in data_point and 'single' in data_point['tools']:
                    gripper_pos = data_point['tools']['single']['position']
                elif 'tools' in data_point and 'left' in data_point['tools']:
                    gripper_pos = data_point['tools']['left']['position']
                else:
                    # Default gripper position if not found
                    gripper_pos = 0.04
                
                # Handle dual-arm data if present
                right_joint_pos = None
                right_joint_vel = None
                right_gripper_pos = None
                
                if is_dual_arm:
                    # Extract right arm data
                    if 'right' in data_point.get('joint_states', {}):
                        right_joint_states = data_point['joint_states']['right']
                        right_joint_pos = right_joint_states['position']
                        right_joint_vel = right_joint_states['velocitie']
                    
                    # Handle right gripper
                    if 'tools' in data_point and 'right' in data_point['tools']:
                        right_gripper_pos = data_point['tools']['right']['position']
                    else:
                        right_gripper_pos = 0.037  # Default to mid position for Monte01
                
                # Map to ACT format
                act_qpos, act_qvel = robot_to_act_joint_mapping(
                    joint_pos, joint_vel, gripper_pos,
                    is_dual_arm=is_dual_arm,
                    right_joint_pos=right_joint_pos,
                    right_joint_vel=right_joint_vel,
                    right_gripper_pos=right_gripper_pos
                )
                
                # For actions, use next valid frame if available
                if idx_i < len(valid_indices) - 1:
                    next_i = valid_indices[idx_i + 1]
                    next_data_point = data_points[next_i]
                    
                    # Handle joint states for next frame
                    if 'single' in next_data_point.get('joint_states', {}):
                        next_joint_states = next_data_point['joint_states']['single']
                    elif 'left' in next_data_point.get('joint_states', {}):
                        next_joint_states = next_data_point['joint_states']['left']
                    else:
                        next_joint_states = joint_states  # Fallback to current
                    
                    # Handle gripper for next frame
                    if 'tools' in next_data_point and 'single' in next_data_point['tools']:
                        next_gripper_pos = next_data_point['tools']['single']['position']
                    elif 'tools' in next_data_point and 'left' in next_data_point['tools']:
                        next_gripper_pos = next_data_point['tools']['left']['position']
                    else:
                        next_gripper_pos = gripper_pos  # Fallback to current
                        
                    # Handle dual-arm for next frame if needed
                    next_right_joint_pos = None
                    next_right_joint_vel = None
                    next_right_gripper_pos = None
                    
                    if is_dual_arm:
                        if 'right' in next_data_point.get('joint_states', {}):
                            next_right_joint_states = next_data_point['joint_states']['right']
                            next_right_joint_pos = next_right_joint_states['position']
                            next_right_joint_vel = next_right_joint_states['velocitie']
                        
                        if 'tools' in next_data_point and 'right' in next_data_point['tools']:
                            next_right_gripper_pos = next_data_point['tools']['right']['position']
                        else:
                            next_right_gripper_pos = right_gripper_pos if right_gripper_pos else 0.037
                    
                    action, _ = robot_to_act_joint_mapping(
                        next_joint_states['position'], 
                        next_joint_states['velocitie'], 
                        next_gripper_pos,
                        is_dual_arm=is_dual_arm,
                        right_joint_pos=next_right_joint_pos,
                        right_joint_vel=next_right_joint_vel,
                        right_gripper_pos=next_right_gripper_pos
                    )
                else:
                    action = act_qpos.copy()
                
                # Store joint data
                data_dict['/observations/qpos'].append(act_qpos)
                data_dict['/observations/qvel'].append(act_qvel)
                data_dict['/action'].append(action)
                
                # Load images with robust handling - support different camera naming
                colors = data_point.get('colors', {})
                
                # Handle different camera naming conventions
                if 'left_ee_cam_color' in colors:
                    ee_cam_path = os.path.join(episode_dir, colors['left_ee_cam_color'])
                elif 'ee_cam_color' in colors:
                    ee_cam_path = os.path.join(episode_dir, colors['ee_cam_color'])
                else:
                    # Use a gray fallback if no ee_cam found
                    ee_cam_path = None
                    
                if 'third_person_cam_color' in colors:
                    third_person_path = os.path.join(episode_dir, colors['third_person_cam_color'])
                else:
                    third_person_path = None
                
                ee_img = load_and_resize_image_robust(ee_cam_path)
                third_person_img = load_and_resize_image_robust(third_person_path)
                
                data_dict['/observations/images/ee_cam'].append(ee_img)
                data_dict['/observations/images/third_person_cam'].append(third_person_img)
                
                # Handle right camera for dual-arm
                if is_dual_arm:
                    if 'right_ee_cam_color' in colors:
                        right_ee_cam_path = os.path.join(episode_dir, colors['right_ee_cam_color'])
                    else:
                        right_ee_cam_path = None
                    right_ee_img = load_and_resize_image_robust(right_ee_cam_path)
                    data_dict['/observations/images/right_ee_cam'].append(right_ee_img)
                
                successful_frames += 1
                
            except Exception as e:
                print(f"    ⚠️  Error at frame {i}: {e}, skipping")
                continue
        
        if successful_frames < 10:
            print(f"  ❌ Episode {episode_idx}: Only {successful_frames} successful frames, skipping")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as HDF5
        output_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
        
        print(f"  Saving HDF5 file with {successful_frames} frames: {output_path}")
        t0 = time.time()
        
        try:
            with h5py.File(output_path, 'w', rdcc_nbytes=1024**2*2) as root:
                # Set attributes
                root.attrs['sim'] = False
                root.attrs['task'] = 'peg_in_hole'
                root.attrs['robot'] = robot_type
                root.attrs['dof'] = dof
                root.attrs['episode_idx'] = episode_idx
                root.attrs['timesteps'] = successful_frames
                root.attrs['original_timesteps'] = max_timesteps
                root.attrs['corrupt_frames'] = corrupt_count
                
                # Create groups
                obs = root.create_group('observations')
                image = obs.create_group('images')
                
                # Create image datasets
                ee_cam_data = image.create_dataset('ee_cam', (successful_frames, 480, 640, 3), 
                                                  dtype='uint8', chunks=(1, 480, 640, 3))
                third_person_data = image.create_dataset('third_person_cam', (successful_frames, 480, 640, 3),
                                                        dtype='uint8', chunks=(1, 480, 640, 3))
                
                # Add right camera dataset for dual-arm
                if is_dual_arm:
                    right_ee_cam_data = image.create_dataset('right_ee_cam', (successful_frames, 480, 640, 3),
                                                            dtype='uint8', chunks=(1, 480, 640, 3))
                
                # Create joint datasets with appropriate DOF
                qpos = obs.create_dataset('qpos', (successful_frames, dof))
                qvel = obs.create_dataset('qvel', (successful_frames, dof))
                action = root.create_dataset('action', (successful_frames, dof))
                
                # Fill datasets
                for name, array in data_dict.items():
                    root[name][...] = np.array(array)
            
            file_size = os.path.getsize(output_path) / (1024**2)  # MB
            print(f"  ✅ Episode {episode_idx:04d} saved in {time.time() - t0:.1f}s ({file_size:.1f}MB, {successful_frames} frames)")
            return output_path
            
        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)  # Clean up partial file
            raise
            
    except Exception as e:
        print(f"  ❌ Error converting episode {episode_idx:04d}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Robustly convert FR3/Monte01 robot data to ACT HDF5 format')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--episodes', type=str, default='all')
    parser.add_argument('--skip_existing', action='store_true')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Find all episode directories
    episode_dirs = sorted([d for d in input_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('episode_')])
    
    # Filter to complete episodes
    complete_episodes = []
    for episode_dir in episode_dirs:
        colors_dir = episode_dir / 'colors'
        data_json = episode_dir / 'data.json'
        if colors_dir.exists() and data_json.exists():
            complete_episodes.append(episode_dir)
    
    if args.episodes == 'all':
        episodes_to_convert = complete_episodes
    else:
        episode_nums = [int(x.strip()) for x in args.episodes.split(',')]
        episodes_to_convert = [d for d in complete_episodes 
                             if int(d.name.split('_')[1]) in episode_nums]
    
    # Skip existing files if requested
    if args.skip_existing:
        episodes_to_convert = [d for d in episodes_to_convert 
                             if not (output_dir / f'episode_{int(d.name.split("_")[1])}.hdf5').exists()]
    
    print(f"=== Robust Robot (FR3/Monte01) to ACT Conversion ===")
    print(f"Complete episodes: {len(complete_episodes)}")
    print(f"Episodes to convert: {len(episodes_to_convert)}")
    print(f"Features: Robust image loading, corrupt frame handling")
    print("-" * 50)
    
    if len(episodes_to_convert) == 0:
        print("No episodes to convert!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed_episodes = []
    total_timesteps = 0
    total_size = 0
    start_time = time.time()
    
    for i, episode_dir in enumerate(episodes_to_convert, 1):
        episode_num = int(episode_dir.name.split('_')[1])
        print(f"\n[{i}/{len(episodes_to_convert)}] Episode {episode_num:04d}:")
        
        try:
            output_path = convert_robot_episode_robust(episode_dir, output_dir, episode_num)
            if output_path:
                success_count += 1
                
                # Get file statistics
                file_size = os.path.getsize(output_path)
                total_size += file_size
                
                # Count timesteps
                with h5py.File(output_path, 'r') as f:
                    timesteps = f.attrs['timesteps']
                    total_timesteps += timesteps
            else:
                failed_episodes.append(episode_num)
                
        except Exception as e:
            print(f"  ❌ Error converting episode {episode_num:04d}: {e}")
            failed_episodes.append(episode_num)
            continue
    
    # Final summary
    elapsed_time = time.time() - start_time
    total_size_gb = total_size / (1024**3)
    
    print("\n" + "="*50)
    print("ROBUST CONVERSION SUMMARY")
    print("="*50)
    print(f"Total episodes processed:      {len(episodes_to_convert)}")
    print(f"Successfully converted:        {success_count}")
    print(f"Failed:                        {len(failed_episodes)}")
    print(f"Total timesteps:               {total_timesteps:,}")
    print(f"Total dataset size:            {total_size_gb:.2f} GB")
    print(f"Conversion time:               {elapsed_time/60:.1f} minutes")
    
    if failed_episodes:
        print(f"\nFailed episodes: {failed_episodes}")
    
    print(f"\nOutput directory: {output_dir}")
    print("Robust conversion completed!")

if __name__ == '__main__':
    main()
