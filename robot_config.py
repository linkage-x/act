#!/usr/bin/env python3
"""
Robot configuration module for ACT
Defines robot-specific parameters and configurations
"""

import numpy as np
from typing import Dict, Tuple, List


class RobotConfig:
    """Base robot configuration class"""
    
    def __init__(self):
        self.name = "base_robot"
        self.dof = 0
        self.gripper_range = (0.0, 1.0)
        self.camera_names = []
        
    def get_gripper_indices(self) -> List[int]:
        """Get indices of gripper joints in state vector"""
        raise NotImplementedError
        
    def normalize_gripper(self, pos: float) -> float:
        """Normalize gripper position to [0, 1]"""
        min_val, max_val = self.gripper_range
        return np.clip((pos - min_val) / (max_val - min_val), 0, 1)
        
    def unnormalize_gripper(self, normalized: float) -> float:
        """Unnormalize gripper position from [0, 1]"""
        min_val, max_val = self.gripper_range
        return normalized * (max_val - min_val) + min_val


class FR3Config(RobotConfig):
    """Configuration for FR3 single-arm robot"""
    
    def __init__(self):
        super().__init__()
        self.name = "fr3"
        self.dof = 8  # 7 arm joints + 1 gripper
        self.gripper_range = (0.0, 0.08)  # meters
        self.camera_names = ['ee_cam', 'third_person_cam']
        
        # Joint indices
        self.arm_indices = list(range(7))
        self.gripper_index = 7
        
    def get_gripper_indices(self) -> List[int]:
        return [self.gripper_index]
        
        
class Monte01Config(RobotConfig):
    """Configuration for Monte01 dual-arm robot"""
    
    def __init__(self):
        super().__init__()
        self.name = "monte01"
        self.dof = 16  # 2 arms: (7+1) * 2
        self.gripper_range = (0.0, 0.074)  # meters
        self.camera_names = ['ee_cam', 'right_ee_cam', 'third_person_cam']
        
        # Joint indices for dual-arm
        self.left_arm_indices = list(range(7))
        self.left_gripper_index = 7
        self.right_arm_indices = list(range(8, 15))
        self.right_gripper_index = 15
        
    def get_gripper_indices(self) -> List[int]:
        return [self.left_gripper_index, self.right_gripper_index]
        
    def split_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split combined state into left and right components"""
        left_state = state[..., :8]
        right_state = state[..., 8:16]
        return left_state, right_state
        
    def combine_state(self, left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
        """Combine left and right states"""
        return np.concatenate([left_state, right_state], axis=-1)


def get_robot_config(robot_type: str) -> RobotConfig:
    """
    Factory function to get robot configuration
    
    Args:
        robot_type: Type of robot ('fr3' or 'monte01')
        
    Returns:
        RobotConfig instance
    """
    configs = {
        'fr3': FR3Config,
        'monte01': Monte01Config,
    }
    
    if robot_type not in configs:
        raise ValueError(f"Unknown robot type: {robot_type}. Available: {list(configs.keys())}")
        
    return configs[robot_type]()


def detect_robot_from_task(task_name: str) -> str:
    """
    Auto-detect robot type from task name
    
    Args:
        task_name: Name of the task
        
    Returns:
        Robot type string
    """
    if task_name.startswith('fr3_'):
        return 'fr3'
    elif task_name.startswith('monte01_'):
        return 'monte01'
    elif task_name.startswith('sim_'):
        # Simulation tasks default to single-arm
        return 'fr3'
    else:
        # Try to detect from other patterns
        if 'bimanual' in task_name.lower() or 'dual' in task_name.lower():
            return 'monte01'
        else:
            return 'fr3'  # Default to FR3


def detect_robot_from_checkpoint(ckpt_dir: str) -> str:
    """
    Auto-detect robot type from checkpoint directory
    
    Args:
        ckpt_dir: Checkpoint directory path
        
    Returns:
        Robot type string
    """
    import os
    import pickle
    
    # Try to load dataset stats to infer robot type
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    if not os.path.exists(stats_path):
        stats_path = os.path.join(ckpt_dir, 'dataset_stats_bac.pkl')
        
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            
        # Check action dimension
        if 'action_mean' in stats:
            action_dim = len(stats['action_mean'])
            if action_dim == 16:
                return 'monte01'
            elif action_dim == 8:
                return 'fr3'
                
    # Check directory name patterns
    if 'monte01' in ckpt_dir.lower():
        return 'monte01'
    elif 'fr3' in ckpt_dir.lower():
        return 'fr3'
    elif 'bimanual' in ckpt_dir.lower() or 'dual' in ckpt_dir.lower():
        return 'monte01'
        
    # Default to FR3
    return 'fr3'