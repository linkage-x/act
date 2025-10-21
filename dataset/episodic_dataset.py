#!/usr/bin/env python3
"""
Episodic Dataset for ACT
Handles loading individual episodes from HDF5 files
"""

import os
import h5py
import numpy as np
import torch
from typing import Dict, List, Tuple
import glog as log

# Import data augmentation
try:
    from data_augmentation import create_training_augmentation
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_augmentation import create_training_augmentation


class EpisodicDataset(torch.utils.data.Dataset):
    """
    Dataset for loading episodes from HDF5 files
    Supports both joint position control and EE pose control
    """

    def __init__(self, episode_ids: List[int], episode_id_to_dir: Dict[int, Tuple[str, int]],
                 camera_names: List[str], norm_stats: Dict, episode_len: int,
                 augmentation_config=None, control_mode='joint'):
        """
        Initialize Episodic Dataset

        Args:
            episode_ids: List of episode IDs to load
            episode_id_to_dir: Mapping from episode ID to (directory, local_episode_id)
            camera_names: List of camera names to load
            norm_stats: Normalization statistics
            episode_len: Maximum episode length (for padding)
            augmentation_config: Data augmentation configuration (training only)
            control_mode: 'joint' for joint position control, 'ee_pose' for EE pose control
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.episode_id_to_dir = episode_id_to_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.is_sim = None
        self.control_mode = control_mode

        # Initialize data augmentation
        if augmentation_config is not None:
            self.augmentation_fn = create_training_augmentation(augmentation_config, camera_names)
            log.info("✅ Training data augmentation enabled")
        else:
            self.augmentation_fn = None

        # Validate episodes and filter out corrupted ones
        self.valid_episode_ids = self._validate_episodes()
        if len(self.valid_episode_ids) > 0:
            first_valid_idx = self.episode_ids.index(self.valid_episode_ids[0])
            self.__getitem_safe__(first_valid_idx)

    def _validate_episodes(self) -> List[int]:
        """Validate episodes and return list of valid episode IDs"""
        valid_ids = []
        for episode_id in self.episode_ids:
            dir_path, local_episode_id = self.episode_id_to_dir[episode_id]
            dataset_path = os.path.join(dir_path, f'episode_{local_episode_id}.hdf5')
            try:
                with h5py.File(dataset_path, 'r') as root:
                    _ = root.attrs['sim']
                    action_shape = root['/action'].shape
                    episode_length = action_shape[0]

                    # Test multiple positions to ensure file integrity
                    test_indices = [0, episode_length // 2, episode_length - 1]
                    for idx in test_indices:
                        if idx < episode_length:
                            _ = root['/observations/qpos'][idx]
                            _ = root['/observations/qvel'][idx]
                            for cam_name in self.camera_names:
                                if f'/observations/images/{cam_name}' in root:
                                    _ = root[f'/observations/images/{cam_name}'][idx]
                            _ = root['/action'][idx]
                    valid_ids.append(episode_id)
            except Exception as e:
                log.error(f"Episode {episode_id} validation failed: {str(e)[:100]}")
        return valid_ids

    def __getitem_safe__(self, index):
        """Safe version of __getitem__ for initialization only"""
        try:
            return self.__getitem__(index)
        except Exception as e:
            log.warn(f"Could not initialize with episode at index {index}: {e}")
            return None

    def __len__(self):
        return len(self.valid_episode_ids)

    def __getitem__(self, index):
        """Load a single episode sample"""
        sample_full_episode = False

        episode_id = self.valid_episode_ids[index]
        dir_path, local_episode_id = self.episode_id_to_dir[episode_id]
        dataset_path = os.path.join(dir_path, f'episode_{local_episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]

            # Sample random start timestep
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            # Load observations at start_ts
            qpos = root['/observations/qpos'][start_ts]

            # Load EE pose if using EE pose control mode
            ee_pose = None
            if self.control_mode == 'ee_pose' and '/observations/ee_pose' in root:
                ee_pose = root['/observations/ee_pose'][start_ts]

            # Load images
            image_dict = dict()
            for cam_name in self.camera_names:
                if f'/observations/images/{cam_name}' in root:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            # Load actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts

                ee_action = None
                if self.control_mode == 'ee_pose' and '/ee_action' in root:
                    ee_action = root['/ee_action'][start_ts:]
            else:
                # Hack for real robot: start from one timestep earlier for alignment
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

                ee_action = None
                if self.control_mode == 'ee_pose' and '/ee_action' in root:
                    ee_action = root['/ee_action'][max(0, start_ts - 1):]

        self.is_sim = is_sim

        # Pad actions to target length
        target_len = self.episode_len

        # Choose action data based on control mode
        if self.control_mode == 'ee_pose' and ee_action is not None:
            action_to_use = ee_action
        else:
            action_to_use = action

        padded_action = np.zeros((target_len, action_to_use.shape[1]), dtype=np.float32)
        actual_action_len = min(action_len, target_len)

        if actual_action_len > 0:
            padded_action[:actual_action_len] = action_to_use[:actual_action_len]

            # Pad with last frame if insufficient data
            if action_len < target_len and action_len > 0:
                last_frame = action_to_use[-1]
                for i in range(action_len, target_len):
                    padded_action[i] = last_frame

        # Create padding mask
        is_pad = np.zeros(target_len)
        if action_len < target_len:
            is_pad[action_len:] = 1

        # Stack camera images
        all_cam_images = []
        for cam_name in self.camera_names:
            if cam_name in image_dict:
                all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # Convert to torch tensors
        image_data = torch.from_numpy(all_cam_images)

        # Use EE pose or joint positions based on control mode
        if self.control_mode == 'ee_pose' and ee_pose is not None:
            qpos_data = torch.from_numpy(ee_pose).float()
        else:
            qpos_data = torch.from_numpy(qpos).float()

        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Channel last -> channel first
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # Normalize images
        image_data = image_data / 255.0

        # Apply data augmentation (training only)
        if self.augmentation_fn is not None:
            image_data = self.augmentation_fn(image_data)

        # Normalize actions and qpos based on control mode
        if self.control_mode == 'ee_pose':
            action_data = (action_data - self.norm_stats["ee_action_mean"]) / self.norm_stats["ee_action_std"]
            qpos_data = (qpos_data - self.norm_stats["ee_pose_mean"]) / self.norm_stats["ee_pose_std"]
        else:
            action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad
