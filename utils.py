import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

# 导入数据增强模块
from data_augmentation import create_training_augmentation

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, episode_id_to_dir, camera_names, norm_stats, episode_len, augmentation_config=None, control_mode='joint'):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.episode_id_to_dir = episode_id_to_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.is_sim = None
        self.control_mode = control_mode  # 'joint' or 'ee_pose'

        # 初始化数据增强（传递camera_names以支持相机特定的增强）
        if augmentation_config is not None:
            self.augmentation_fn = create_training_augmentation(augmentation_config, camera_names)
            print("✅ 训练数据增强已启用")
        else:
            self.augmentation_fn = None
            print("ℹ️ 训练数据增强已禁用")
        # Validate episodes and filter out corrupted ones
        self.valid_episode_ids = self._validate_episodes()
        if len(self.valid_episode_ids) > 0:
            # Use first valid episode to initialize self.is_sim
            first_valid_idx = self.episode_ids.index(self.valid_episode_ids[0])
            self.__getitem_safe__(first_valid_idx)

    def _validate_episodes(self):
        """Validate episodes and return list of valid episode IDs"""
        valid_ids = []
        for episode_id in self.episode_ids:
            dir_path, local_episode_id = self.episode_id_to_dir[episode_id]
            dataset_path = os.path.join(dir_path, f'episode_{local_episode_id}.hdf5')
            try:
                with h5py.File(dataset_path, 'r') as root:
                    # Check if required datasets exist and are readable
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
                                _ = root[f'/observations/images/{cam_name}'][idx]
                            _ = root['/action'][idx]
                    valid_ids.append(episode_id)
            except Exception as e:
                print(f"Episode {episode_id} validation failed, excluding from dataset: {str(e)[:100]}")
        return valid_ids
    
    def __getitem_safe__(self, index):
        """Safe version of __getitem__ for initialization only"""
        try:
            return self.__getitem__(index)
        except Exception as e:
            print(f"Warning: Could not initialize with episode at index {index}: {e}")
            return None
    
    def __len__(self):
        return len(self.valid_episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.valid_episode_ids[index]
        dir_path, local_episode_id = self.episode_id_to_dir[episode_id]
        dataset_path = os.path.join(dir_path, f'episode_{local_episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]

            # Load EE pose if using EE pose control mode
            ee_pose = None
            if self.control_mode == 'ee_pose' and '/observations/ee_pose' in root:
                ee_pose = root['/observations/ee_pose'][start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts

                # Load EE pose actions if available
                ee_action = None
                if self.control_mode == 'ee_pose' and '/ee_action' in root:
                    ee_action = root['/ee_action'][start_ts:]
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

                # Load EE pose actions if available
                ee_action = None
                if self.control_mode == 'ee_pose' and '/ee_action' in root:
                    ee_action = root['/ee_action'][max(0, start_ts - 1):]

        self.is_sim = is_sim
        
        # Use episode_len for padding
        target_len = self.episode_len

        # Choose action data based on control mode
        if self.control_mode == 'ee_pose' and ee_action is not None:
            action_to_use = ee_action
        else:
            action_to_use = action

        padded_action = np.zeros((target_len, action_to_use.shape[1] if len(action_to_use.shape) > 1 else original_action_shape[1]), dtype=np.float32)
        actual_action_len = min(action_len, target_len)

        # Fill the padded_action array with available actions
        if actual_action_len > 0:
            padded_action[:actual_action_len] = action_to_use[:actual_action_len]

            # If input data is insufficient, pad with the last frame data
            if action_len < target_len and action_len > 0:
                last_frame = action_to_use[-1]  # Get the last frame
                # Repeat the last frame for the remaining timesteps
                for i in range(action_len, target_len):
                    padded_action[i] = last_frame

        # Set padding mask: 1 for padded timesteps (beyond original data), 0 for real actions
        is_pad = np.zeros(target_len)
        if action_len < target_len:
            is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)

        # Use EE pose or joint positions based on control mode
        if self.control_mode == 'ee_pose' and ee_pose is not None:
            qpos_data = torch.from_numpy(ee_pose).float()
        else:
            qpos_data = torch.from_numpy(qpos).float()

        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        # 应用数据增强（仅在训练时）
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


def get_norm_stats(dataset_dirs, episode_id_to_dir, episode_ids, control_mode='joint'):
    all_qpos_data = []
    all_action_data = []
    all_ee_pose_data = []
    all_ee_action_data = []

    has_ee_pose = False

    for episode_id in episode_ids:
        dir_path, local_episode_id = episode_id_to_dir[episode_id]
        dataset_path = os.path.join(dir_path, f'episode_{local_episode_id}.hdf5')
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                qvel = root['/observations/qvel'][()]
                action = root['/action'][()]

                # Check if EE pose data is available
                if '/observations/ee_pose' in root and '/ee_action' in root:
                    has_ee_pose = True
                    ee_pose = root['/observations/ee_pose'][()]
                    ee_action = root['/ee_action'][()]
                    all_ee_pose_data.append(torch.from_numpy(ee_pose))
                    all_ee_action_data.append(torch.from_numpy(ee_action))

            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
        except Exception as e:
            print(f"Skipping {dataset_path} due to error: {e}")
            continue

    # Concatenate all data instead of stacking (episodes have different lengths)
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    # Compute EE pose normalization stats if available
    if has_ee_pose and len(all_ee_pose_data) > 0:
        all_ee_pose_data = torch.cat(all_ee_pose_data, dim=0)
        all_ee_action_data = torch.cat(all_ee_action_data, dim=0)

        ee_pose_mean = all_ee_pose_data.mean(dim=0, keepdim=True)
        ee_pose_std = all_ee_pose_data.std(dim=0, keepdim=True)
        ee_pose_std = torch.clip(ee_pose_std, 1e-2, np.inf)

        ee_action_mean = all_ee_action_data.mean(dim=0, keepdim=True)
        ee_action_std = all_ee_action_data.std(dim=0, keepdim=True)
        ee_action_std = torch.clip(ee_action_std, 1e-2, np.inf)

        stats["ee_pose_mean"] = ee_pose_mean.numpy().squeeze()
        stats["ee_pose_std"] = ee_pose_std.numpy().squeeze()
        stats["ee_action_mean"] = ee_action_mean.numpy().squeeze()
        stats["ee_action_std"] = ee_action_std.numpy().squeeze()
        stats["has_ee_pose"] = True
    else:
        stats["has_ee_pose"] = False

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, episode_len=None, augmentation_config=None, control_mode='joint'):
    # Support both single directory (string) and multiple directories (list)
    if isinstance(dataset_dir, str):
        dataset_dirs = [dataset_dir]
    else:
        dataset_dirs = dataset_dir

    print(f'\nData from: {dataset_dirs}')
    print(f'Control mode: {control_mode}\n')

    # Get list of available episode files and extract episode IDs from all directories
    import glob
    episode_files = []
    episode_id_to_dir = {}  # Map episode_id to (directory, local_episode_id)

    for dir_path in dataset_dirs:
        dir_episode_files = sorted(glob.glob(os.path.join(dir_path, 'episode_*.hdf5')))
        for file_path in dir_episode_files:
            filename = os.path.basename(file_path)
            local_episode_id = int(filename.replace('episode_', '').replace('.hdf5', ''))

            # Create global unique episode ID by combining dir index and local ID
            global_episode_id = len(episode_files)
            episode_files.append(file_path)
            episode_id_to_dir[global_episode_id] = (dir_path, local_episode_id)

    available_episode_ids = []

    for episode_idx, file_path in enumerate(episode_files):
        try:
            # Test if file can be opened AND data can be read at multiple points
            with h5py.File(file_path, 'r') as root:
                # Verify essential data structures are accessible
                _ = root.attrs.get('sim')
                action_shape = root['/action'].shape
                episode_length = action_shape[0]

                # Test multiple random positions to ensure file integrity throughout
                test_indices = [0, episode_length // 2, episode_length - 1]
                if episode_length > 10:
                    # Add a few more random test points for longer episodes
                    test_indices.extend([episode_length // 4, 3 * episode_length // 4])

                for idx in test_indices:
                    if idx < episode_length:
                        _ = root['/observations/qpos'][idx]
                        _ = root['/observations/qvel'][idx]
                        # Check if camera data exists and is readable
                        if '/observations/images' in root:
                            cam_names = list(root['/observations/images'].keys())
                            for cam_name in cam_names:
                                _ = root[f'/observations/images/{cam_name}'][idx]
                        _ = root['/action'][idx]

            # Use global episode ID (index in episode_files list)
            available_episode_ids.append(episode_idx)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
    
    # Use all available episodes, ignore the num_episodes parameter for auto-calculation
    actual_num_episodes = len(available_episode_ids)
    print(f"Auto-detected {actual_num_episodes} available episodes from {len(dataset_dirs)} directories: {available_episode_ids}")

    # If num_episodes is specified and less than available episodes, use only that many
    if num_episodes is not None and num_episodes < actual_num_episodes:
        available_episode_ids = available_episode_ids[:num_episodes]
        actual_num_episodes = len(available_episode_ids)
        print(f"Limited to first {actual_num_episodes} episodes as requested")
    
    # Check if we have enough valid episodes
    if actual_num_episodes == 0:
        raise ValueError(f"No valid episodes found in {dataset_dirs}. All HDF5 files appear corrupted.")
    if actual_num_episodes < 2:
        raise ValueError(f"Need at least 2 valid episodes for train/val split, but only found {actual_num_episodes}")
    
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(actual_num_episodes)
    train_episode_ids = [available_episode_ids[i] for i in shuffled_indices[:int(train_ratio * actual_num_episodes)]]
    val_episode_ids = [available_episode_ids[i] for i in shuffled_indices[int(train_ratio * actual_num_episodes):]]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dirs, episode_id_to_dir, available_episode_ids, control_mode)

    # Validate control mode availability
    if control_mode == 'ee_pose' and not norm_stats.get('has_ee_pose', False):
        print("WARNING: EE pose control mode requested but no EE pose data found in dataset!")
        print("Falling back to joint control mode.")
        control_mode = 'joint'

    # If episode_len not provided, use the maximum episode length found
    if episode_len is None:
        episode_len = 5000  # Use the max from constants.py

    # construct dataset and dataloader
    # 注意：需要将camera_names传递给augmentation pipeline以支持相机特定的增强
    train_dataset = EpisodicDataset(train_episode_ids, episode_id_to_dir, camera_names, norm_stats, episode_len, augmentation_config, control_mode)
    val_dataset = EpisodicDataset(val_episode_ids, episode_id_to_dir, camera_names, norm_stats, episode_len, control_mode=control_mode)  # 验证集不使用增强
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    # Validation should be deterministic for stable metrics
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
