#!/usr/bin/env python3

import os
import json
import numpy as np
import h5py
import cv2
import argparse
from pathlib import Path
import time

def load_fr3_episode_data(episode_dir):
    """Load FR3 episode data from directory"""
    data_file = os.path.join(episode_dir, 'data.json')
    
    with open(data_file, 'r') as f:
        episode_data = json.load(f)
    
    return episode_data

def fr3_to_act_joint_mapping(fr3_joint_pos, fr3_joint_vel, fr3_gripper_pos):
    """
    Map FR3 8-DOF (7 arm + 1 gripper) directly to 8-DOF action space
    FR3: [7 arm joints + 1 gripper] -> ACT: [7 arm joints + 1 gripper]
    
    This maintains the original FR3 structure without artificial padding
    """
    # FR3 has 7 arm joints + 1 gripper = 8 DOF total
    fr3_arm_joints = np.array(fr3_joint_pos[:7])  # 7 DOF arm
    fr3_arm_velocities = np.array(fr3_joint_vel[:7])
    
    # ACT format: 8 DOF total (matching FR3)
    act_qpos = np.zeros(8)
    act_qvel = np.zeros(8)
    
    # Direct mapping: FR3 7 arm joints -> ACT 7 arm joints
    act_qpos[0:7] = fr3_arm_joints[0:7]
    act_qvel[0:7] = fr3_arm_velocities[0:7]
    
    # Gripper mapping: FR3 gripper -> ACT gripper (index 7)
    # FR3 gripper position (0.0 = closed, ~0.08 = open)
    # Normalize to a reasonable range for ACT
    gripper_normalized = np.clip(fr3_gripper_pos, 0.0, 0.08)
    act_qpos[7] = gripper_normalized
    
    # Gripper velocity (approximated as 0 if not available)
    act_qvel[7] = 0.0
    
    return act_qpos, act_qvel

def load_and_resize_image(image_path, target_size=(480, 640)):
    """Load image and resize to target dimensions"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if necessary
    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    return img

def convert_fr3_episode_to_hdf5(episode_dir, output_dir, episode_idx):
    """Convert single FR3 episode to ACT HDF5 format"""
    
    print(f"Converting episode {episode_idx:04d}...")
    
    # Check if episode data exists
    data_file = os.path.join(episode_dir, 'data.json')
    colors_dir = os.path.join(episode_dir, 'colors')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing data.json for episode {episode_idx}")
    if not os.path.exists(colors_dir):
        raise FileNotFoundError(f"Missing colors directory for episode {episode_idx}")
    
    # Load FR3 data
    episode_data = load_fr3_episode_data(episode_dir)
    
    # Extract data points
    data_points = episode_data['data']
    max_timesteps = len(data_points)
    
    # Prepare data structures for ACT format (8-DOF)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
        '/observations/images/ee_cam': [],
        '/observations/images/third_person_cam': []
    }
    
    print(f"  Processing {max_timesteps} timesteps...")
    
    for i, data_point in enumerate(data_points):
        if i % 200 == 0:
            print(f"    Progress: {i}/{max_timesteps} ({i/max_timesteps*100:.1f}%)")
            
        try:
            # Extract joint states (7 arm joints)
            joint_states = data_point['joint_states']['single']
            joint_pos = joint_states['position']  # 7 arm joints
            joint_vel = joint_states['velocitie']  # Note: typo in original data
            
            # Extract gripper state (1 DOF)
            gripper_pos = data_point['tools']['single']['position']
            
            # Map FR3 8-DOF to ACT 8-DOF format
            act_qpos, act_qvel = fr3_to_act_joint_mapping(joint_pos, joint_vel, gripper_pos)
            
            # For actions, we use the next timestep's joint positions
            # For the last timestep, repeat the current position
            if i < len(data_points) - 1:
                next_joint_states = data_points[i+1]['joint_states']['single']
                next_gripper_pos = data_points[i+1]['tools']['single']['position']
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
            
            # Load and process images
            ee_cam_path = os.path.join(episode_dir, data_point['colors']['ee_cam_color'])
            third_person_path = os.path.join(episode_dir, data_point['colors']['third_person_cam_color'])
            
            ee_img = load_and_resize_image(ee_cam_path)
            third_person_img = load_and_resize_image(third_person_path)
            
            data_dict['/observations/images/ee_cam'].append(ee_img)
            data_dict['/observations/images/third_person_cam'].append(third_person_img)
            
        except Exception as e:
            print(f"    Error at timestep {i}: {e}")
            raise
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HDF5
    output_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
    
    print(f"  Saving HDF5 file: {output_path}")
    t0 = time.time()
    
    try:
        with h5py.File(output_path, 'w', rdcc_nbytes=1024**2*2) as root:
            # Set attributes
            root.attrs['sim'] = False  # Real robot data
            root.attrs['task'] = 'peg_in_hole'
            root.attrs['robot'] = 'fr3'
            root.attrs['dof'] = 8  # 8-DOF system
            root.attrs['episode_idx'] = episode_idx
            root.attrs['timesteps'] = max_timesteps
            
            # Create groups
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            # Create image datasets
            ee_cam_data = image.create_dataset('ee_cam', (max_timesteps, 480, 640, 3), 
                                              dtype='uint8', chunks=(1, 480, 640, 3))
            third_person_data = image.create_dataset('third_person_cam', (max_timesteps, 480, 640, 3),
                                                    dtype='uint8', chunks=(1, 480, 640, 3))
            
            # Create joint datasets (8 DOF for FR3)
            qpos = obs.create_dataset('qpos', (max_timesteps, 8))
            qvel = obs.create_dataset('qvel', (max_timesteps, 8))
            action = root.create_dataset('action', (max_timesteps, 8))
            
            # Fill datasets
            for name, array in data_dict.items():
                root[name][...] = np.array(array)
        
        file_size = os.path.getsize(output_path) / (1024**2)  # MB
        print(f"  ✓ Episode {episode_idx:04d} saved in {time.time() - t0:.1f}s ({file_size:.1f}MB)")
        return output_path
        
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)  # Clean up partial file
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert FR3 data to ACT HDF5 format')
    parser.add_argument('--input_dir', type=str, required=True, 
                      help='Input directory containing FR3 episodes (e.g., data/peg_in_hole)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for HDF5 files')
    parser.add_argument('--episodes', type=str, default='all',
                      help='Episodes to convert (e.g., "1,2,3" or "all")')
    parser.add_argument('--skip_existing', action='store_true',
                      help='Skip episodes that already have HDF5 files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Find all episode directories
    episode_dirs = sorted([d for d in input_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('episode_')])
    
    # Filter to complete episodes only (have both colors and data.json)
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
    
    print(f"=== FR3 to ACT Conversion ===")
    print(f"Found {len(episode_dirs)} total episodes")
    print(f"Complete episodes (with colors + data.json): {len(complete_episodes)}")
    print(f"Episodes to convert: {len(episodes_to_convert)}")
    print(f"FR3 format: 7-DOF arm + 1-DOF gripper = 8-DOF total")
    print(f"Output directory: {output_dir}")
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
            output_path = convert_fr3_episode_to_hdf5(episode_dir, output_dir, episode_num)
            success_count += 1
            
            # Get file statistics
            file_size = os.path.getsize(output_path)
            total_size += file_size
            
            # Count timesteps from HDF5 file
            with h5py.File(output_path, 'r') as f:
                timesteps = f.attrs['timesteps']
                total_timesteps += timesteps
            
        except Exception as e:
            print(f"  ✗ Error converting episode {episode_num:04d}: {e}")
            failed_episodes.append(episode_num)
            continue
    
    # Final summary
    elapsed_time = time.time() - start_time
    total_size_gb = total_size / (1024**3)
    
    print("\n" + "="*50)
    print("CONVERSION SUMMARY")
    print("="*50)
    print(f"Total episodes processed:      {len(episodes_to_convert)}")
    print(f"Successfully converted:        {success_count}")
    print(f"Failed:                        {len(failed_episodes)}")
    print(f"Total timesteps:               {total_timesteps:,}")
    print(f"Total dataset size:            {total_size_gb:.2f} GB")
    print(f"Conversion time:               {elapsed_time/60:.1f} minutes")
    print(f"Average per episode:           {elapsed_time/len(episodes_to_convert):.1f} seconds")
    
    if failed_episodes:
        print(f"\nFailed episodes: {failed_episodes}")
    
    print(f"\nOutput directory: {output_dir}")
    print("Ready for ACT training!")

if __name__ == '__main__':
    main()