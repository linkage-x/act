#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def validate_hdf5_data(hdf5_dir):
    """Validate converted HDF5 data"""
    
    files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.hdf5')])
    
    print(f"=== FR3 to ACT Conversion Validation ===")
    print(f"Found {len(files)} HDF5 files")
    print()
    
    valid_episodes = []
    total_timesteps = 0
    
    for file in files:
        filepath = os.path.join(hdf5_dir, file)
        episode_num = file.replace('episode_', '').replace('.hdf5', '')
        
        try:
            with h5py.File(filepath, 'r') as f:
                # Basic info
                timesteps = f['action'].shape[0]
                dof = f['action'].shape[1]
                
                # Check attributes
                robot = f.attrs.get('robot', 'unknown')
                task = f.attrs.get('task', 'unknown')
                is_sim = f.attrs.get('sim', True)
                
                print(f"Episode {episode_num:>3}: {timesteps:>4} steps, {dof} DOF, {robot}, task={task}")
                
                valid_episodes.append(episode_num)
                total_timesteps += timesteps
                
        except Exception as e:
            print(f"Episode {episode_num:>3}: ❌ Error - {str(e)[:50]}...")
    
    print(f"\n✅ Valid episodes: {len(valid_episodes)}")
    print(f"📊 Total timesteps: {total_timesteps:,}")
    print(f"📁 Total size: {sum(os.path.getsize(os.path.join(hdf5_dir, f)) for f in files) / (1024**3):.1f} GB")
    
    return valid_episodes

def inspect_episode_data(hdf5_path):
    """Detailed inspection of a single episode"""
    
    print(f"\n=== Detailed Inspection: {os.path.basename(hdf5_path)} ===")
    
    with h5py.File(hdf5_path, 'r') as f:
        
        # Datasets structure
        print("Dataset structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: {obj.shape} {obj.dtype}")
        f.visititems(print_structure)
        
        # Sample data
        qpos = f['observations/qpos'][:]
        actions = f['action'][:]
        
        print(f"\nJoint ranges:")
        for i in range(qpos.shape[1]):
            joint_name = f"Joint {i}" if i < 7 else "Gripper"
            print(f"  {joint_name:>8}: [{qpos[:, i].min():7.3f}, {qpos[:, i].max():7.3f}]")
        
        # Image data
        if 'observations/images' in f:
            cameras = list(f['observations/images'].keys())
            print(f"\nCameras: {cameras}")
            
            for cam in cameras:
                img_data = f[f'observations/images/{cam}']
                print(f"  {cam}: {img_data.shape} {img_data.dtype}")
        
        # Task progression - check gripper movement
        gripper_pos = qpos[:, 7]  # Last joint is gripper
        gripper_range = gripper_pos.max() - gripper_pos.min()
        print(f"\nGripper movement range: {gripper_range:.3f} m")
        
        if gripper_range > 0.01:  # If gripper moved more than 1cm
            print("✅ Gripper shows significant movement (likely grasp/release)")
        else:
            print("⚠️  Limited gripper movement detected")

def plot_episode_trajectory(hdf5_path, output_dir=None):
    """Plot joint trajectories for visualization"""
    
    with h5py.File(hdf5_path, 'r') as f:
        qpos = f['observations/qpos'][:]
        
    episode_name = os.path.basename(hdf5_path).replace('.hdf5', '')
    
    # Create subplot for each joint
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Joint Trajectories - {episode_name}', fontsize=14)
    
    for i in range(8):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        ax.plot(qpos[:, i])
        joint_name = f"Joint {i+1}" if i < 7 else "Gripper"
        ax.set_title(joint_name)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{episode_name}_trajectories.png'), dpi=150)
        print(f"Saved trajectory plot: {output_dir}/{episode_name}_trajectories.png")
    else:
        plt.show()

def main():
    hdf5_dir = "data/peg_in_hole_hdf5"
    
    if not os.path.exists(hdf5_dir):
        print(f"Error: Directory {hdf5_dir} not found")
        return
    
    # Validate all files
    valid_episodes = validate_hdf5_data(hdf5_dir)
    
    if valid_episodes:
        # Inspect first valid episode in detail
        first_episode = f"episode_{valid_episodes[0]}.hdf5"
        inspect_episode_data(os.path.join(hdf5_dir, first_episode))
        
        # Generate trajectory plot
        print(f"\n=== Generating Trajectory Plot ===")
        plot_episode_trajectory(os.path.join(hdf5_dir, first_episode), "plots")
    
    print(f"\n🎉 Validation complete! Ready for ACT training.")

if __name__ == '__main__':
    main()