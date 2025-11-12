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
from dataset.reader import ActionType, ObservationType

# Import dataset
from dataset.episodic_dataset import EpisodicDataset

class HDF5Loader(DataLoaderBase):
    """
    HDF5 Data Loader for ACT
    Handles loading HDF5 episodes with support for joint positions and EE poses
    """

    def __init__(self, config: Dict[str, Any], dataset_dir: str,
                 action_type: ActionType = ActionType.JointPosition,
                 observation_type: ObservationType = ObservationType.JointPosition,
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

        # Data loading parameters
        # self.num_episodes = config.get('num_episodes', None)
        self.camera_names = config.get('camera_names', ['ee_cam', 'third_person_cam'])
        self.batch_size_train = config.get('batch_size_train', 32)
        self.batch_size_val = config.get('batch_size_val', 32)
        self.episode_len = config.get('episode_len', 800)
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
        log.info(f"   Skip steps: {self.skip_steps_nums}")
        log.info(f"   Camera names: {self.camera_names}")

    def load_episodes_from_hdf5(self) -> Tuple[List[int], Dict[int, Tuple[str, int]]]:
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
        # available_episode_ids = []
        # for episode_idx, file_path in enumerate(episode_files):
        #     try:
                # with h5py.File(file_path, 'r') as root:
                    # _ = root.attrs.get('sim')
                    # action_shape = root['/action'].shape
                    # episode_length = action_shape[0]

                    # # Test multiple random positions
                    # test_indices = [0, episode_length // 2, episode_length - 1]
                    # if episode_length > 10:
                    #     test_indices.extend([episode_length // 4, 3 * episode_length // 4])

                    # for idx in test_indices:
                    #     if idx < episode_length:
                    #         _ = root['/observations/state'][idx]
                    #         if '/observations/images' in root:
                    #             cam_names = list(root['/observations/images'].keys())
                    #             for cam_name in cam_names:
                    #                 _ = root[f'/observations/images/{cam_name}'][idx]
                    #         _ = root['/action'][idx]

        #         available_episode_ids.append(episode_idx)
        #     except Exception as e:
        #         log.error(f"Skipping {file_path} due to error: {e}")

        # actual_num_episodes = len(available_episode_ids)
        # log.info(f"📊 Auto-detected {actual_num_episodes} available episodes from {len(dataset_dirs)} directories")

        # # Limit episodes if num_episodes is specified
        # if self.num_episodes is not None and self.num_episodes < actual_num_episodes:
        #     available_episode_ids = available_episode_ids[:self.num_episodes]
        #     actual_num_episodes = len(available_episode_ids)
        #     log.info(f"📊 Limited to first {actual_num_episodes} episodes as requested")

        # if actual_num_episodes == 0:
        #     raise ValueError(f"No valid episodes found in {dataset_dirs}")
        # if actual_num_episodes < 2:
        #     raise ValueError(f"Need at least 2 valid episodes for train/val split, but only found {actual_num_episodes}")

        # Build available episode ids from discovered files
        available_episode_ids = list(range(len(episode_files)))
        # Limit if num_episodes specified
        # if self.num_episodes is not None:
        #     try:
        #         n = int(self.num_episodes)
        #         available_episode_ids = available_episode_ids[:max(0, n)]
        #     except Exception:
        #         pass
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
        all_state_data = []
        all_action_data = []

        for episode_id in episode_ids:
            dir_path, local_episode_id = episode_id_to_dir[episode_id]
            dataset_path = os.path.join(dir_path, f'episode_{local_episode_id}.hdf5')
            try:
                with h5py.File(dataset_path, 'r') as root:
                    # Stats are computed on the canonical keys written at conversion time
                    state = root['/observations/state'][()]
                    action = root['/action'][()]

                all_state_data.append(torch.from_numpy(state))
                all_action_data.append(torch.from_numpy(action))
            except Exception as e:
                log.error(f"Skipping {dataset_path} due to error: {e}")
                continue

        # Concatenate all data
        all_state_data = torch.cat(all_state_data, dim=0)
        all_action_data = torch.cat(all_action_data, dim=0)

        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf)

        # Normalize state data
        state_mean = all_state_data.mean(dim=0, keepdim=True)
        state_std = all_state_data.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf)

        # Keep stats as torch tensors for compatibility with EpisodicDataset
        stats = {
            "action_mean": action_mean.squeeze(),
            "action_std": action_std.squeeze(),
            "state_mean": state_mean.squeeze(),
            "state_std": state_std.squeeze(),
            "example_state": state
        }

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
        available_episode_ids, episode_id_to_dir = self.load_episodes_from_hdf5()

        # Split into train/val
        train_ratio = 0.8
        actual_num_episodes = len(available_episode_ids)
        shuffled_indices = np.random.permutation(actual_num_episodes)
        train_episode_ids = [available_episode_ids[i] for i in shuffled_indices[:int(train_ratio * actual_num_episodes)]]
        val_episode_ids = [available_episode_ids[i] for i in shuffled_indices[int(train_ratio * actual_num_episodes):]]

        # Compute normalization stats
        norm_stats = self.compute_normalization_stats(available_episode_ids, episode_id_to_dir)

        # Create datasets (no transformation here; datasets consume HDF5 as-is)
        train_dataset = EpisodicDataset(
            train_episode_ids, episode_id_to_dir, self.camera_names,
            norm_stats, self.episode_len, self.augmentation_config
        )
        val_dataset = EpisodicDataset(
            val_episode_ids, episode_id_to_dir, self.camera_names,
            norm_stats, self.episode_len, None  # No augmentation for validation
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
        import time
        from tqdm import tqdm

        # Get source and output directories from config
        source_dir = self._config.get('task_dir')
        output_dir = self._config.get('output_dir', self.dataset_dir)
        image_size = tuple(self._config.get('image_size', [480, 640]))
        is_sim = self._config.get('is_sim', False)  # Default to False for real robot data
        min_episode_len = self._config.get('min_episode_len', None)  # None = no filter
        max_episode_len = self._config.get('max_episode_len', None)  # None = no filter

        log.info("🚀 Starting episode data conversion to HDF5...")
        log.info(f"   📁 Source: {source_dir}")
        log.info(f"   📁 Output: {output_dir}")
        log.info(f"   📉 Skip steps: {self.skip_steps_nums}")
        log.info(f"   🖼️  Image size: {image_size}")

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
            log.info(f"🔄 Processing {episode_name}...")
            
            # Load episode data via DataLoaderBase helper
            episode_data, _ = self.load_episode(source_dir, episode_name, self.skip_steps_nums)
            data_points = episode_data or []
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

            # Convert to HDF5
            output_name = f"episode_{global_counter[0]}.hdf5"
            global_counter[0] += 1
            output_path = os.path.join(output_dir, output_name)

            log.info(f"     🔄 Converting to HDF5 ({len(data_points)} steps)...")

            try:
                success = self._convert_episode_to_hdf5(
                    data_points,
                    output_path=output_path,
                    image_size=image_size,
                    is_sim=is_sim,
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

    def _convert_episode_to_hdf5(self, data_points, output_path,
                                 image_size=(480, 640), is_sim=False):
        """
        Convert episode data points (already processed by DataLoaderBase) to HDF5.

        Args:
            data_points: List of dicts containing colors/actions/observations.
            output_path: Output HDF5 file path.
            image_size: Target image size (height, width).
            is_sim: Whether the source episodes are from simulation.
        """
        import cv2

        if not data_points:
            log.error("     ❌ Empty data_points passed to conversion")
            return False

        strict_camera = bool(self._config.get('strict_camera', True))

        # Read types to decide whether to append gripper state at the end
        obs_type_str = str(self._config.get('obs_type', 'joint_position_only')).lower()
        act_type_str = str(self._config.get('action_type', 'joint_position')).lower()

        state_array = []
        action_array = []
        image_arrays = {cam_name: [] for cam_name in self.camera_names}

        for i, point in enumerate(data_points):
            try:
                obs_dict = point.get('observations') or {}
                act_dict = point.get('actions') or {}
                color_dict = point.get('colors') or {}

                if not obs_dict or not act_dict:
                    log.warn(f"     ⚠️  Missing observations/actions at step {i}")
                    return False

                # Flatten observation/actions by concatenating vectors from all keys in sorted order
                def _flatten_vecs(dct):
                    keys = sorted(dct.keys())
                    parts = []
                    for k in keys:
                        v = dct.get(k)
                        if v is None:
                            continue
                        v = np.asarray(v, dtype=np.float32).reshape(-1)
                        parts.append(v)
                    if not parts:
                        return None
                    return np.concatenate(parts, axis=0)

                obs_vec = _flatten_vecs(obs_dict)
                act_vec = _flatten_vecs(act_dict)

                if obs_vec is None or act_vec is None:
                    log.warn(f"     ⚠️  Empty obs/action vector at step {i}")
                    return False

                # Extract gripper/tool state if available; prefer key containing 'gripper'
                # tools_dict = point.get('tools') or {}
                # gripper_val = None
                # if isinstance(tools_dict, dict) and len(tools_dict) > 0:
                #     tkeys = sorted(tools_dict.keys())
                #     # prefer keys that mention gripper
                #     prefer = [k for k in tkeys if 'gripper' in k.lower()]
                #     sel_key = prefer[0] if prefer else tkeys[0]
                #     tpos = tools_dict.get(sel_key, {}).get('position', None)
                #     if tpos is not None:
                #         gv = np.asarray(tpos, dtype=np.float32).reshape(-1)
                #         # Use the first element if it is a vector; most grippers are scalar
                #         gripper_val = gv if gv.size == 1 else np.array([gv[0]], dtype=np.float32)

                # For EE obs/action modes, ensure EE part first and append gripper state last (absolute)
                # def _maybe_append_gripper(vec, mode: str):
                #     # mode: 'obs' or 'act'
                #     if gripper_val is None:
                #         return vec
                #     # expected base length for EE pose (position+quat): 7
                #     ee_mode = (
                #         (mode == 'obs' and obs_type_str in ('end_effector_pose', 'delta_ee_pose')) or
                #         (mode == 'act' and act_type_str in ('end_effector_pose', 'end_effector_pose_delta'))
                #     )
                #     if not ee_mode:
                #         return vec
                #     # Append only if it looks like gripper is not already included
                #     # if vec.shape[0] == 7:
                #     #     return np.concatenate([vec, gripper_val], axis=0)
                #     return vec

                # obs_vec = _maybe_append_gripper(obs_vec, 'obs')
                # act_vec = _maybe_append_gripper(act_vec, 'act')

                state_array.append(obs_vec)
                action_array.append(act_vec)

                # Process images
                missing_cams = []
                for cam_name in self.camera_names:
                    img = None
                    if cam_name in color_dict:
                        img = color_dict[cam_name]
                    else:
                        key_exact = f"{cam_name}_color"
                        if key_exact in color_dict:
                            img = color_dict[key_exact]
                        else:
                            for key in color_dict.keys():
                                if cam_name in key:
                                    img = color_dict[key]
                                    break
                    if img is None:
                        missing_cams.append(cam_name)
                        continue

                    img = np.asarray(img)
                    if img.ndim != 3:
                        log.warn(f"     ⚠️  Invalid image shape at step {i} for camera {cam_name}: {img.shape}")
                        return False

                    if img.shape[:2] != image_size:
                        img = cv2.resize(img, (image_size[1], image_size[0]))
                    image_arrays[cam_name].append(img)

                if missing_cams and strict_camera:
                    log.warn(f"     ⚠️  Missing cameras at step {i}: {missing_cams}")
                    return False

            except Exception as e:
                log.warn(f"     ⚠️  Error processing step {i}: {e}")
                import traceback
                traceback.print_exc()
                return False

        state_array = np.asarray(state_array, dtype=np.float32)
        action_array = np.asarray(action_array, dtype=np.float32)

        min_len = min(len(state_array), len(action_array))
        if min_len == 0:
            log.error("     ❌ No valid state/action pairs after processing")
            return False
        if len(state_array) != len(action_array):
            log.warn(f"     ⚠️  State/action length mismatch ({len(state_array)} vs {len(action_array)}); truncating")
            state_array = state_array[:min_len]
            action_array = action_array[:min_len]

        for cam_name, images in image_arrays.items():
            if images:
                image_arrays[cam_name] = np.stack(images, axis=0)
            else:
                image_arrays[cam_name] = np.empty((0, image_size[0], image_size[1], 3), dtype=np.uint8)

        # Dimension validation: For EE modes, actions must include gripper state (8 dims total)
        ee_obs_mode = obs_type_str in ("end_effector_pose", "delta_ee_pose")
        ee_act_mode = act_type_str in ("end_effector_pose", "end_effector_pose_delta")
        if ee_obs_mode or ee_act_mode:
            if action_array.shape[1] != 8:
                raise ValueError(
                    f"EE mode requires 8-dim actions (7D EE + 1D gripper), got actions{action_array.shape}. "
                    f"Ensure tool/gripper state is present and appended during conversion."
                )

        log.info(f"     📊 Arrays: state{state_array.shape}, actions{action_array.shape}")

        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('/observations/state', data=state_array)
                f.create_dataset('/action', data=action_array)

                for cam_name, images in image_arrays.items():
                    if images.size > 0:
                        f.create_dataset(
                            f'/observations/images/{cam_name}',
                            data=images,
                            compression=None
                        )

                f.attrs['sim'] = is_sim
                f.attrs['episode_length'] = len(state_array)

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

    # Infer io-mode (obs->act) from output_dir suffix like *_<obs>2<act>
    # Supported tokens: q, dq, ee, dee
    def parse_io_mode_from_output_dir(path: str):
        import re
        m = re.search(r"(dee|ee|dq|q)2(dee|ee|dq|q)(?![A-Za-z0-9])", path)
        if not m:
            return None, None
        return m.group(1), m.group(2)

    obs_tok, act_tok = parse_io_mode_from_output_dir(output_dir or "")

    # Map token to ObservationType / ActionType
    def map_obs(tok: str) -> ObservationType:
        return {
            'q': ObservationType.JointPosition,
            'dq': ObservationType.DeltaJointPosition,
            'ee': ObservationType.EEPose,
            'dee': ObservationType.DeltaEEPose,
        }.get(tok, ObservationType.JointPosition)

    def map_act(tok: str) -> ActionType:
        return {
            'q': ActionType.JointPosition,
            'dq': ActionType.DeltaJointPosition,
            'ee': ActionType.EEPose,
            'dee': ActionType.DeltaEEPose,
        }.get(tok, ActionType.JointPosition)

    if obs_tok and act_tok:
        obs_type = map_obs(obs_tok)
        action_type = map_act(act_tok)
        print(f"   🔎 Inferred io-mode from output_dir: {obs_tok}2{act_tok} -> obs={obs_type}, act={action_type}")

    print(f"\n📋 Configuration:")
    print(f"   Task dir (source): {task_dir}")
    print(f"   Output dir: {output_dir}")
    print(f"   Action type: {action_type}")
    print(f"   Obs type: {obs_type}")
    print(f"   Skip steps: {skip_steps_nums}")
    print(f"   Image size: {image_size}\n")

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
