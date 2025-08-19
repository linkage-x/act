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

def fr3_to_act_joint_mapping(fr3_joint_pos, fr3_joint_vel, fr3_gripper_pos):
    """Map FR3 8-DOF (7 arm + 1 gripper) directly to 8-DOF action space"""
    fr3_arm_joints = np.array(fr3_joint_pos[:7])  # 7 DOF arm
    fr3_arm_velocities = np.array(fr3_joint_vel[:7])
    
    # ACT format: 8 DOF total (matching FR3)
    act_qpos = np.zeros(8)
    act_qvel = np.zeros(8)
    
    # Direct mapping: FR3 7 arm joints -> ACT 7 arm joints
    act_qpos[0:7] = fr3_arm_joints[0:7]
    act_qvel[0:7] = fr3_arm_velocities[0:7]
    
    # Gripper mapping
    gripper_normalized = np.clip(fr3_gripper_pos, 0.0, 0.08)
    act_qpos[7] = gripper_normalized
    act_qvel[7] = 0.0
    
    return act_qpos, act_qvel

def convert_fr3_episode_robust(episode_dir, output_dir, episode_idx):
    """Convert single FR3 episode to ACT HDF5 format with robust error handling"""
    
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
        
        # Pre-check for corrupt data points
        valid_indices = []
        corrupt_count = 0
        
        for i, data_point in enumerate(data_points):
            try:
                # Check if joint states exist
                joint_states = data_point['joint_states']['single']
                joint_pos = joint_states['position']
                joint_vel = joint_states['velocitie']  # Note: typo in original
                gripper_pos = data_point['tools']['single']['position']
                
                # Check if image paths exist
                ee_cam_path = os.path.join(episode_dir, data_point['colors']['ee_cam_color'])
                third_person_path = os.path.join(episode_dir, data_point['colors']['third_person_cam_color'])
                
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
        
        print(f"  Processing {len(valid_indices)} valid timesteps...")
        
        successful_frames = 0
        for idx_i, i in enumerate(valid_indices):
            if idx_i % 200 == 0:
                print(f"    Progress: {idx_i}/{len(valid_indices)} ({idx_i/len(valid_indices)*100:.1f}%)")
            
            try:
                data_point = data_points[i]
                
                # Extract joint states
                joint_states = data_point['joint_states']['single']
                joint_pos = joint_states['position']
                joint_vel = joint_states['velocitie']
                gripper_pos = data_point['tools']['single']['position']
                
                # Map to ACT format
                act_qpos, act_qvel = fr3_to_act_joint_mapping(joint_pos, joint_vel, gripper_pos)
                
                # For actions, use next valid frame if available
                if idx_i < len(valid_indices) - 1:
                    next_i = valid_indices[idx_i + 1]
                    next_data_point = data_points[next_i]
                    next_joint_states = next_data_point['joint_states']['single']
                    next_gripper_pos = next_data_point['tools']['single']['position']
                    action, _ = fr3_to_act_joint_mapping(
                        next_joint_states['position'], 
                        next_joint_states['velocitie'], 
                        next_gripper_pos
                    )
                else:
                    action = act_qpos.copy()
                
                # Store joint data
                data_dict['/observations/qpos'].append(act_qpos)
                data_dict['/observations/qvel'].append(act_qvel)
                data_dict['/action'].append(action)
                
                # Load images with robust handling
                ee_cam_path = os.path.join(episode_dir, data_point['colors']['ee_cam_color'])
                third_person_path = os.path.join(episode_dir, data_point['colors']['third_person_cam_color'])
                
                ee_img = load_and_resize_image_robust(ee_cam_path)
                third_person_img = load_and_resize_image_robust(third_person_path)
                
                data_dict['/observations/images/ee_cam'].append(ee_img)
                data_dict['/observations/images/third_person_cam'].append(third_person_img)
                
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
                root.attrs['robot'] = 'fr3'
                root.attrs['dof'] = 8
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
                
                # Create joint datasets
                qpos = obs.create_dataset('qpos', (successful_frames, 8))
                qvel = obs.create_dataset('qvel', (successful_frames, 8))
                action = root.create_dataset('action', (successful_frames, 8))
                
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
    parser = argparse.ArgumentParser(description='Robustly convert FR3 data to ACT HDF5 format')
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
    
    print(f"=== Robust FR3 to ACT Conversion ===")
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
            output_path = convert_fr3_episode_robust(episode_dir, output_dir, episode_num)
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