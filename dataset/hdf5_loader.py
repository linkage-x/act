#!/usr/bin/env python3
"""
HDF5 Data Loader for ACT
Inspired by lerobot_loader.py structure, handles HDF5 episode loading with EE pose support
"""

import os
import yaml
import h5py
import numpy as np
import torch
import glob
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
import glog as log

# Import base class
from dataset.data_loader_base import DataLoaderBase
from dataset.reader import ActionType, ObservationType, Action_Type_Mapping_Dict, Observation_Type_Mapping_Dict

# Import dataset
from dataset.episodic_dataset import EpisodicDataset

class HDF5Loader(DataLoaderBase):
    """
    HDF5 Data Loader for ACT
    Handles loading HDF5 episodes with support for joint positions and EE poses
    """

    def __init__(self, config: Dict[str, Any], dataset_dir: str,
                 action_type: ActionType = ActionType.JOINT_POSITION,
                 observation_type: ObservationType = ObservationType.JOINT_POSITION_ONLY,
                 skip_steps_nums: int = 2):
        """
        Initialize HDF5 Loader

        Args:
            config: Configuration dictionary
            dataset_dir: Directory containing HDF5 episode files
            action_type: Type of action (joint_position or end_effector_pose)
            observation_type: Type of observation
            skip_steps_nums: Number of steps to skip when loading (1 = no skip, 2 = use every 2nd frame, etc.)
        """
        # For compatibility with DataLoaderBase, use a dummy task_dir
        super().__init__(config, dataset_dir, "data.json", action_type, observation_type)

        self.dataset_dir = dataset_dir
        self.config = config
        self.skip_steps_nums = skip_steps_nums

        # Determine control mode from action_type and observation_type
        # Support ee_pose if either action or observation uses EE pose
        if (action_type == ActionType.END_EFFECTOR_POSE or
            observation_type == ObservationType.END_EFFECTOR_POSE or
            observation_type == ObservationType.JOINT_POSITION_END_EFFECTOR):
            self.control_mode = 'ee_pose'
        else:
            self.control_mode = 'joint'

        # Data loading parameters
        self.num_episodes = config.get('num_episodes', None)
        self.camera_names = config.get('camera_names', ['ee_cam', 'third_person_cam'])
        self.batch_size_train = config.get('batch_size_train', 32)
        self.batch_size_val = config.get('batch_size_val', 32)
        self.episode_len = config.get('episode_len', 5000)
        # DataLoader settings (configurable)
        self.num_workers_train = int(config.get('num_workers_train', 4))
        self.num_workers_val = int(config.get('num_workers_val', 4))
        self.prefetch_factor_train = int(config.get('prefetch_factor_train', 2))
        self.prefetch_factor_val = int(config.get('prefetch_factor_val', 2))
        self.persistent_workers = bool(config.get('persistent_workers', True))

        # Augmentation
        self.augmentation_config = config.get('augmentation_config', None)

        log.info(f"📂 HDF5Loader initialized:")
        log.info(f"   Dataset dir: {dataset_dir}")
        log.info(f"   Control mode: {self.control_mode}")
        log.info(f"   Skip steps: {self.skip_steps_nums}")
        log.info(f"   Camera names: {self.camera_names}")

    def load_episodes(self) -> Tuple[List[int], Dict[int, Tuple[str, int]]]:
        """
        Load episode files and create episode ID mapping

        Returns:
            available_episode_ids: List of valid episode IDs
            episode_id_to_dir: Mapping from episode ID to (dir_path, local_episode_id)
        """
        # Support both single directory (string) and multiple directories (list)
        if isinstance(self.dataset_dir, str):
            dataset_dirs = [self.dataset_dir]
        else:
            dataset_dirs = self.dataset_dir

        log.info(f'\n📁 Loading data from: {dataset_dirs}')
        log.info(f'🎯 Control mode: {self.control_mode}\n')

        # Get list of available episode files
        episode_files = []
        episode_id_to_dir = {}

        for dir_path in dataset_dirs:
            dir_episode_files = sorted(glob.glob(os.path.join(dir_path, 'episode_*.hdf5')))
            for file_path in dir_episode_files:
                filename = os.path.basename(file_path)
                local_episode_id = int(filename.replace('episode_', '').replace('.hdf5', ''))

                # Create global unique episode ID
                global_episode_id = len(episode_files)
                episode_files.append(file_path)
                episode_id_to_dir[global_episode_id] = (dir_path, local_episode_id)

        # Validate episodes
        available_episode_ids = []
        for episode_idx, file_path in enumerate(episode_files):
            try:
                with h5py.File(file_path, 'r') as root:
                    _ = root.attrs.get('sim')
                    action_shape = root['/action'].shape
                    episode_length = action_shape[0]

                    # Test multiple random positions
                    test_indices = [0, episode_length // 2, episode_length - 1]
                    if episode_length > 10:
                        test_indices.extend([episode_length // 4, 3 * episode_length // 4])

                    for idx in test_indices:
                        if idx < episode_length:
                            _ = root['/observations/qpos'][idx]
                            _ = root['/observations/qvel'][idx]
                            if '/observations/images' in root:
                                cam_names = list(root['/observations/images'].keys())
                                for cam_name in cam_names:
                                    _ = root[f'/observations/images/{cam_name}'][idx]
                            _ = root['/action'][idx]

                available_episode_ids.append(episode_idx)
            except Exception as e:
                log.error(f"Skipping {file_path} due to error: {e}")

        actual_num_episodes = len(available_episode_ids)
        log.info(f"📊 Auto-detected {actual_num_episodes} available episodes from {len(dataset_dirs)} directories")

        # Limit episodes if num_episodes is specified
        if self.num_episodes is not None and self.num_episodes < actual_num_episodes:
            available_episode_ids = available_episode_ids[:self.num_episodes]
            actual_num_episodes = len(available_episode_ids)
            log.info(f"📊 Limited to first {actual_num_episodes} episodes as requested")

        if actual_num_episodes == 0:
            raise ValueError(f"No valid episodes found in {dataset_dirs}")
        if actual_num_episodes < 2:
            raise ValueError(f"Need at least 2 valid episodes for train/val split, but only found {actual_num_episodes}")

        return available_episode_ids, episode_id_to_dir

    def compute_normalization_stats(self, episode_ids: List[int], episode_id_to_dir: Dict[int, Tuple[str, int]]) -> Dict[str, Any]:
        """
        Compute normalization statistics

        Args:
            episode_ids: List of episode IDs
            episode_id_to_dir: Episode ID to directory mapping

        Returns:
            Dictionary containing normalization statistics
        """
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
                log.error(f"Skipping {dataset_path} due to error: {e}")
                continue

        # Concatenate all data
        all_qpos_data = torch.cat(all_qpos_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)

        # Normalize joint action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf)

        # Normalize qpos data
        qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
        qpos_std = all_qpos_data.std(dim=0, keepdim=True)
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

        # Keep stats as torch tensors for compatibility with EpisodicDataset
        stats = {
            "action_mean": action_mean.squeeze(),
            "action_std": action_std.squeeze(),
            "qpos_mean": qpos_mean.squeeze(),
            "qpos_std": qpos_std.squeeze(),
            "example_qpos": qpos
        }

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

            # Keep as tensors, not numpy
            stats["ee_pose_mean"] = ee_pose_mean.squeeze()
            stats["ee_pose_std"] = ee_pose_std.squeeze()
            stats["ee_action_mean"] = ee_action_mean.squeeze()
            stats["ee_action_std"] = ee_action_std.squeeze()
            stats["has_ee_pose"] = True
        else:
            stats["has_ee_pose"] = False

        return stats

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any], bool]:
        """
        Create train and validation dataloaders

        Returns:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            norm_stats: Normalization statistics
            is_sim: Whether data is from simulation
        """
        # Load episodes
        available_episode_ids, episode_id_to_dir = self.load_episodes()

        # Split into train/val
        train_ratio = 0.8
        actual_num_episodes = len(available_episode_ids)
        shuffled_indices = np.random.permutation(actual_num_episodes)
        train_episode_ids = [available_episode_ids[i] for i in shuffled_indices[:int(train_ratio * actual_num_episodes)]]
        val_episode_ids = [available_episode_ids[i] for i in shuffled_indices[int(train_ratio * actual_num_episodes):]]

        # Compute normalization stats
        norm_stats = self.compute_normalization_stats(available_episode_ids, episode_id_to_dir)

        # Validate control mode availability
        if self.control_mode == 'ee_pose' and not norm_stats.get('has_ee_pose', False):
            raise Exception("⚠️ EE pose control mode requested but no EE pose data found in dataset!")

        # Create datasets
        train_dataset = EpisodicDataset(
            train_episode_ids, episode_id_to_dir, self.camera_names,
            norm_stats, self.episode_len, self.augmentation_config, self.control_mode
        )
        val_dataset = EpisodicDataset(
            val_episode_ids, episode_id_to_dir, self.camera_names,
            norm_stats, self.episode_len, None, self.control_mode  # No augmentation for validation
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers_train,
            prefetch_factor=self.prefetch_factor_train if self.num_workers_train > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers_train > 0 else False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers_val,
            prefetch_factor=self.prefetch_factor_val if self.num_workers_val > 0 else None,
            persistent_workers=self.persistent_workers if self.num_workers_val > 0 else False,
        )

        log.info(f"✅ Dataloaders created:")
        log.info(f"   Training episodes: {len(train_episode_ids)}")
        log.info(f"   Validation episodes: {len(val_episode_ids)}")

        return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

    def convert_dataset(self):
        """
        Convert raw episode data to HDF5 format with skip_steps_nums downsampling
        Similar to lerobot_loader.py but converts to HDF5 instead of LeRobotDataset
        """
        import json
        import cv2
        import time
        from tqdm import tqdm

        # Get source and output directories from config
        source_dir = self._config.get('task_dir')
        output_dir = self._config.get('output_dir', self.dataset_dir)
        image_size = tuple(self._config.get('image_size', [480, 640]))
        store_ee_pose = self._config.get('store_ee_pose', True)
        umi_mode = self._config.get('umi_mode', False)
        is_sim = self._config.get('is_sim', False)  # Default to False for real robot data
        min_episode_len = self._config.get('min_episode_len', None)  # None = no filter
        max_episode_len = self._config.get('max_episode_len', None)  # None = no filter

        log.info("🚀 Starting episode data conversion to HDF5...")
        log.info(f"   📁 Source: {source_dir}")
        log.info(f"   📁 Output: {output_dir}")
        log.info(f"   📉 Skip steps: {self.skip_steps_nums}")
        log.info(f"   🖼️  Image size: {image_size}")
        log.info(f"   🎯 Store EE pose: {store_ee_pose}")
        log.info(f"   🧭 UMI mode (EE relative-to-first): {umi_mode}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get episode directories
        episode_dirs = [d for d in os.listdir(source_dir)
                       if d.startswith('episode_') and
                       os.path.isdir(os.path.join(source_dir, d))]
        episode_dirs.sort()

        if not episode_dirs:
            log.error("❌ No episode directories found!")
            return 0

        log.info(f"📊 Found {len(episode_dirs)} episodes to process\n")

        total_converted = 0
        global_counter = [0]
        start_time = time.time()

        # Process each episode
        for episode_name in tqdm(episode_dirs, desc="Processing episodes", unit="episode"):
            episode_dir = os.path.join(source_dir, episode_name)

            log.info(f"🔄 Processing {episode_name}...")

            # Load episode data
            data_file = os.path.join(episode_dir, 'data.json')
            if not os.path.exists(data_file):
                log.error(f"   ❌ No data.json found in {episode_dir}")
                continue

            with open(data_file, 'r') as f:
                episode_data = json.load(f)

            data_points = episode_data.get('data', [])
            if not data_points:
                log.error(f"   ❌ No data points found in {episode_name}")
                continue

            original_len = len(data_points)
            log.info(f"   📊 Original length: {original_len} steps")

            # Apply episode length filtering if configured
            if min_episode_len is not None and original_len < min_episode_len:
                log.warn(f"   ⚠️  Episode too short ({original_len} < {min_episode_len} steps), PASS...")
                continue
            if max_episode_len is not None and original_len > max_episode_len:
                log.warn(f"   ⚠️  Episode too long ({original_len} > {max_episode_len} steps), PASS...")
                continue

            # Apply skip_steps_nums downsampling (like lerobot_loader)
            if self.skip_steps_nums > 1:
                data_points = data_points[::self.skip_steps_nums]
                log.info(f"   📉 After skip_steps_nums={self.skip_steps_nums}: {len(data_points)} steps")

            # Convert to HDF5
            output_name = f"episode_{global_counter[0]}.hdf5"
            global_counter[0] += 1
            output_path = os.path.join(output_dir, output_name)

            log.info(f"     🔄 Converting to HDF5 ({len(data_points)} steps)...")

            try:
                success = self._convert_episode_to_hdf5(
                    data_points, episode_dir, output_path,
                    image_size, store_ee_pose, is_sim, umi_mode
                )
                if success:
                    log.info(f"     ✅ Saved to {output_name}")
                    total_converted += 1
                else:
                    log.error(f"     ❌ Failed to convert episode")

            except Exception as e:
                log.error(f"     ❌ Error converting episode: {e}")
                import traceback
                traceback.print_exc()

        elapsed_time = time.time() - start_time
        log.info("✅ Conversion completed!")
        log.info(f"   📊 Original episodes: {len(episode_dirs)}")
        log.info(f"   📊 Generated HDF5 files: {total_converted}")
        log.info(f"   ⏱️  Total time: {elapsed_time:.1f}s")
        log.info(f"   📁 Output directory: {output_dir}")

        return total_converted

    def _convert_episode_to_hdf5(self, data_points, episode_dir, output_path,
                                 image_size=(480, 640), store_ee_pose=True, is_sim=False, umi_mode=False):
        """
        Convert episode data points to HDF5 format

        Args:
            data_points: List of data points to convert
            episode_dir: Episode directory path
            output_path: Output HDF5 file path
            image_size: Target image size (height, width)
            store_ee_pose: Whether to store end-effector poses
            is_sim: Whether data is from simulation (default: False for real robot)
            umi_mode: Whether to use UMI mode (EE pose relative to first frame)
        """
        import cv2

        episode_len = len(data_points)

        # ---- UMI helpers: pose math (quaternion [qx,qy,qz,qw]) ----
        def _quat_normalize(q):
            q = np.asarray(q, dtype=np.float32)
            n = np.linalg.norm(q)
            if n < 1e-8:
                return np.array([0, 0, 0, 1], dtype=np.float32)
            return q / n

        def _quat_mul(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            return _quat_normalize(np.array([x, y, z, w], dtype=np.float32))

        def _quat_inv(q):
            x, y, z, w = q
            return _quat_normalize(np.array([-x, -y, -z, w], dtype=np.float32))

        def _quat_apply(q, v):
            q_vec = q[:3]
            w = q[3]
            t = 2.0 * np.cross(q_vec, v)
            return v + w * t + np.cross(q_vec, t)

        def _pose_rel_to(pose_abs, base_abs):
            p = np.asarray(pose_abs[:3], dtype=np.float32)
            q = _quat_normalize(np.asarray(pose_abs[3:], dtype=np.float32))
            p0 = np.asarray(base_abs[:3], dtype=np.float32)
            q0 = _quat_normalize(np.asarray(base_abs[3:], dtype=np.float32))
            q0_inv = _quat_inv(q0)
            dp = _quat_apply(q0_inv, (p - p0))
            dq = _quat_mul(q0_inv, q)
            return np.concatenate([dp, dq], dtype=np.float32)

        # Find base EE pose for UMI (first valid)
        base_ee_pose_abs = None
        if umi_mode and store_ee_pose:
            for _pt in data_points:
                es = _pt.get('ee_states', {})
                if 'single' in es and 'pose' in es['single']:
                    base_ee_pose_abs = np.array(es['single']['pose'], dtype=np.float32)
                    break

        # Prepare arrays
        qpos_array = []
        qvel_array = []
        action_array = []
        ee_pose_array = []
        ee_action_array = []
        # Initialize image arrays based on configured camera names
        image_arrays = {cam_name: [] for cam_name in self.camera_names}

        # Process each data point
        for i, point in enumerate(data_points):
            try:
                # Extract joint states
                joint_states = point.get('joint_states') or {}
                positions = []
                velocities = []
                if isinstance(joint_states, dict) and 'single' in joint_states:
                    positions = joint_states['single'].get('position') or []
                    velocities = joint_states['single'].get('velocity') or []
                if (not positions or not velocities) and umi_mode:
                    # UMI数据可能无关节，使用零填充
                    positions = [0.0] * 7
                    velocities = [0.0] * 7
                if len(positions) < 7 or len(velocities) < 7:
                    log.warn(f"     ⚠️  Insufficient joint data at step {i}")
                    return False

                # Extract gripper position
                tools = point.get('tools') or {}
                gripper_pos = 0.04  # Default
                if isinstance(tools, dict) and 'single' in tools and isinstance(tools['single'], dict) and 'position' in tools['single']:
                    gripper_pos = tools['single']['position']

                # Convert to ACT format: 8 DOF (7 joints + 1 gripper)
                qpos = np.zeros(8, dtype=np.float32)
                qvel = np.zeros(8, dtype=np.float32)
                qpos[:7] = positions[:7]
                qvel[:7] = velocities[:7]
                qpos[7] = gripper_pos
                qvel[7] = 0.0

                qpos_array.append(qpos)
                qvel_array.append(qvel)

                # Extract EE pose if available
                if store_ee_pose:
                    ee_states = point.get('ee_states') or {}
                    if isinstance(ee_states, dict) and 'single' in ee_states and 'pose' in ee_states['single']:
                        # EE pose (absolute) → relative if UMI
                        ee_pose_abs = np.array(ee_states['single']['pose'], dtype=np.float32)
                        ee_pose_rel = _pose_rel_to(ee_pose_abs, base_ee_pose_abs) if (umi_mode and base_ee_pose_abs is not None) else ee_pose_abs
                        ee_pose_with_gripper = np.append(ee_pose_rel, gripper_pos)
                        ee_pose_array.append(ee_pose_with_gripper)
                    else:
                        ee_pose_array.append(np.zeros(8, dtype=np.float32))

                # Actions (use next state as action, or current for last step)
                if i < episode_len - 1:
                    next_point = data_points[i + 1]
                    next_joint_states = next_point.get('joint_states') or {}
                    if isinstance(next_joint_states, dict) and 'single' in next_joint_states:
                        next_positions = next_joint_states['single'].get('position', positions) or positions
                        next_tools = next_point.get('tools') or {}
                        next_gripper_pos = gripper_pos
                        if isinstance(next_tools, dict) and 'single' in next_tools and isinstance(next_tools['single'], dict) and 'position' in next_tools['single']:
                            next_gripper_pos = next_tools['single']['position']

                        action_qpos = np.zeros(8, dtype=np.float32)
                        action_qpos[:7] = next_positions[:7]
                        action_qpos[7] = next_gripper_pos
                        action_array.append(action_qpos)

                        # EE pose action
                        if store_ee_pose:
                            next_ee_states = next_point.get('ee_states') or {}
                            if isinstance(next_ee_states, dict) and 'single' in next_ee_states and 'pose' in next_ee_states['single']:
                                next_ee_pose_abs = np.array(next_ee_states['single']['pose'], dtype=np.float32)
                                next_ee_pose_rel = _pose_rel_to(next_ee_pose_abs, base_ee_pose_abs) if (umi_mode and base_ee_pose_abs is not None) else next_ee_pose_abs
                                next_ee_pose_with_gripper = np.append(next_ee_pose_rel, next_gripper_pos)
                                ee_action_array.append(next_ee_pose_with_gripper)
                            else:
                                ee_action_array.append(np.zeros(8, dtype=np.float32))
                    else:
                        action_array.append(qpos)
                        if store_ee_pose:
                            ee_action_array.append(ee_pose_array[-1] if ee_pose_array else np.zeros(8, dtype=np.float32))
                else:
                    action_array.append(qpos)
                    if store_ee_pose:
                        ee_action_array.append(ee_pose_array[-1] if ee_pose_array else np.zeros(8, dtype=np.float32))

                # Process images - match configured camera_names with *_color keys (strict by default)
                colors = point.get('colors', {}) or {}
                if isinstance(colors, dict):
                    selected_images = {}
                    missing_cams = []
                    for cam_name in self.camera_names:
                        key_exact = f"{cam_name}_color"
                        selected_key = None
                        if key_exact in colors:
                            selected_key = key_exact
                        else:
                            for k in colors.keys():
                                if k and k.endswith('_color') and cam_name in k:
                                    selected_key = k
                                    break
                        if selected_key is None or not (colors[selected_key] and 'path' in colors[selected_key]):
                            missing_cams.append(cam_name)
                            continue
                        selected_images[cam_name] = os.path.join(episode_dir, colors[selected_key]['path'])

                    if missing_cams:
                        log.warn(f"     ⚠️  Missing cameras at step {i}: {missing_cams}")
                        if bool(self._config.get('strict_camera', True)):
                            return False

                    for cam_name, img_path in selected_images.items():
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                if img.shape[:2] != image_size:
                                    img = cv2.resize(img, (image_size[1], image_size[0]))
                                image_arrays[cam_name].append(img)
                        except Exception as e:
                            log.warn(f"     ⚠️  Error loading {img_path}: {e}")

            except Exception as e:
                log.warn(f"     ⚠️  Error processing step {i}: {e}")
                import traceback
                traceback.print_exc()
                return False

        # Convert to numpy arrays
        qpos_array = np.array(qpos_array, dtype=np.float32)
        qvel_array = np.array(qvel_array, dtype=np.float32)
        action_array = np.array(action_array, dtype=np.float32)

        if store_ee_pose and ee_pose_array:
            ee_pose_array = np.array(ee_pose_array, dtype=np.float32)
            ee_action_array = np.array(ee_action_array, dtype=np.float32)

        for cam_name in image_arrays:
            if image_arrays[cam_name]:
                image_arrays[cam_name] = np.array(image_arrays[cam_name])

        log.info(f"     📊 Arrays: qpos{qpos_array.shape}, actions{action_array.shape}")
        if store_ee_pose and len(ee_pose_array) > 0:
            log.info(f"     📊 EE pose: {ee_pose_array.shape}, ee_actions{ee_action_array.shape}")

        # Save to HDF5
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('/observations/qpos', data=qpos_array)
                f.create_dataset('/observations/qvel', data=qvel_array)
                f.create_dataset('/action', data=action_array)

                # Save EE pose data if available
                if store_ee_pose and len(ee_pose_array) > 0:
                    f.create_dataset('/observations/ee_pose', data=ee_pose_array)
                    f.create_dataset('/ee_action', data=ee_action_array)

                for cam_name, images in image_arrays.items():
                    if len(images) > 0:
                        f.create_dataset(f'/observations/images/{cam_name}',
                                       data=images,
                                       compression=None)

                # Metadata
                f.attrs['sim'] = is_sim
                f.attrs['episode_length'] = episode_len
                f.attrs['has_ee_pose'] = store_ee_pose and len(ee_pose_array) > 0

            return True

        except Exception as e:
            log.error(f"     ❌ HDF5 save error: {e}")
            return False

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HDF5 Data Converter for ACT')
    parser.add_argument('--config', type=str, default='hdf5_loader_config.yaml',
                       help='Path to configuration YAML file')

    args = parser.parse_args()

    # Load configuration - handle both absolute and relative paths
    if os.path.isabs(args.config):
        cfg_file = args.config
    else:
        # If relative path, check current directory first, then module directory
        if os.path.exists(args.config):
            cfg_file = args.config
        else:
            cur_path = os.path.dirname(os.path.abspath(__file__))
            cfg_file = os.path.join(cur_path, args.config)

    if not os.path.exists(cfg_file):
        print(f"❌ Error: Config file not found: {cfg_file}")
        print(f"   Please create a config file or specify correct path with --config")
        exit(1)

    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)

    print(f"\n{'='*60}")
    print(f"HDF5Loader - CONVERT Mode")
    print(f"{'='*60}\n")
    print(f"📋 Configuration loaded from: {cfg_file}")

    # Parse configuration
    task_dir = config.get("task_dir")
    output_dir = config.get("output_dir")
    action_type_str = config.get("action_type", "joint_position")
    obs_type_str = config.get("obs_type", "joint_position_only")
    skip_steps_nums = config.get("skip_steps_nums", 1)
    image_size_list = config.get("image_size", [480, 640])
    image_size = tuple(image_size_list)
    store_ee_pose = config.get("store_ee_pose", True)

    # Convert action type and observation type strings to enums
    action_type = Action_Type_Mapping_Dict.get(action_type_str, ActionType.JOINT_POSITION)
    obs_type = Observation_Type_Mapping_Dict.get(obs_type_str, ObservationType.JOINT_POSITION_ONLY)

    print(f"\n📋 Configuration:")
    print(f"   Task dir (source): {task_dir}")
    print(f"   Output dir: {output_dir}")
    print(f"   Action type: {action_type_str}")
    print(f"   Obs type: {obs_type_str}")
    print(f"   Skip steps: {skip_steps_nums}")
    print(f"   Image size: {image_size}")
    print(f"   Store EE pose: {store_ee_pose}\n")

    if not task_dir or not output_dir:
        print("❌ Error: task_dir and output_dir are required in config")
        exit(1)

    # Create loader
    loader = HDF5Loader(
        config=config,
        dataset_dir=output_dir,
        action_type=action_type,
        observation_type=obs_type,
        skip_steps_nums=skip_steps_nums
    )

    # Perform conversion with skip_steps_nums downsampling
    total_converted = loader.convert_dataset()

    print(f"\n✅ Conversion completed: {total_converted} HDF5 files created")
    print(f"\n{'='*60}\n")
