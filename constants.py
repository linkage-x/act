import pathlib
import numpy as np

### Task parameters
DATA_DIR = '<put your data dir here>'

SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },

    'fr3_peg_in_hole': {
        'dataset_dir': '/media/hanyu/ubuntu/act_project/8_25/data_0823_converted',
        'num_episodes': 39,
        'episode_len': 1200,  # Max timesteps from our data
        'camera_names': ['ee_cam', 'third_person_cam'],
        'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
    },

    'fr3_pickup_kiwi': {
        'dataset_dir': '/media/hanyu/winxdows11/Data/pick_up_kiwi_hdf5',
        'num_episodes': 48,  # Final: 98 successfully converted episodes
        'episode_len': 2000,  # Extended for larger episodes  
        'camera_names': ['ee_cam', 'third_person_cam'],
        'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
    },
    'monte01_peg_in_hole': {
        'dataset_dir': '/boot/common_data/peg_in_hole_hdf5',
        'num_episodes': 46,  # Updated based on converted data
        'episode_len': 2000,  # Extended for larger episodes
        'camera_names': ['ee_cam', 'right_ee_cam', 'third_person_cam'],  # Mapped from left->ee_cam, right->right_ee_cam
        'state_dim': 16  # Monte01: Dual-arm 7+1 DOF each = 16 total
    },
    'fr3_peg_in_hole_0914_50ep': {
        'dataset_dir': '/boot/common_data/fr3_peg_in_hole_0914_50ep_hdf5',
        'num_episodes': 50,  # Updated based on converted data
        'episode_len': 2000,  # Extended for larger episodes
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],  # Mapped from left->ee_cam, right->right_ee_cam
        'state_dim': 8
    },
    'fr3_block_stacking_0915_55ep': {
        'dataset_dir': '/boot/common_data/fr3_block_stacking_0915_55ep_hdf5',
        'num_episodes': 53,  # Updated based on converted data
        'episode_len': 2000,  # Extended for larger episodes
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],  # Mapped from left->ee_cam, right->right_ee_cam
        'state_dim': 8
    },
    'fr3_blockstacking_0916_50eps_fixloc': {
        'dataset_dir': '/boot/common_data/fr3_blockstacking_0916_50eps_fixloc_hdf5',
        'num_episodes': 50,  # Updated based on converted data
        'episode_len': 2000,  # Extended for larger episodes
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],  # Mapped from left->ee_cam, right->right_ee_cam
        'state_dim': 8
    },

    # Example of multi-directory mixed training configuration
    'fr3_mixed_training_bs': {
        'dataset_dir': [
            '/boot/common_data/fr3_blockstacking_0916_50eps_fixloc_hdf5',
            '/boot/common_data/fr3_block_stacking_0920_50ep_hdf5',
            '/boot/common_data/fr3_pih_0923_25ep_fixloc_hdf5',
        ],
        # 'num_episodes': Auto-detected from all .hdf5 files in dataset_dir directories
        'episode_len': 2000,  # Max episode length across all datasets
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],  # Common cameras across datasets
        'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
    },

    # Preprocessed block stacking dataset (optimized episode lengths)
    'fr3_bs_mix_ds_0924': {
        'dataset_dir': ['/boot/common_data/fr3_bs_0916_50ep_ds_hdf5',
            '/boot/common_data/fr3_bs_0920_50ep_ds_hdf5',],
        'episode_len': 550,  # Optimized episode length after preprocessing
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],
        'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
    },
    'fr3_bs_0916_50ep_ds': {
                    'dataset_dir': ['/boot/common_data/fr3_bs_0916_50ep_ds_hdf5'],
                            'episode_len': 550,  # Optimized episode length after preprocessing
                                    'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],
                                            'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
                                                },
    'fr3_mixed_training_pih': {
        'dataset_dir': [
            '/boot/common_data/fr3_peg_in_hole_0914_50ep_hdf5',
            '/boot/common_data/fr3_peginhole_0918_48ep_hdf5',
            '/boot/common_data/fr3_peg_in_hole_0920_50ep_hdf5'
        ],
        # 'num_episodes': Auto-detected from all .hdf5 files in dataset_dir directories
        'episode_len': 2000,  # Max episode length across all datasets
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],  # Common cameras across datasets
        'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
    },
    'fr3_liquidtransfer_0920': {
        'dataset_dir': [
            '/boot/common_data/fr3_liquid_transfer_0920_50ep_hdf5',
                                ],
        'episode_len': 2000,  # Max episode length across all datasets
        'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],  # Common cameras across datasets
        'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
    },
    'fr3_pih_0918_48ep_ds': {
                    'dataset_dir': ['/boot/common_data/fr3_pih_0918_48ep_ds_hdf5',
                                    ],
                            'episode_len': 600,  # Optimized episode length after preprocessing
                                    'camera_names': ['ee_cam', 'third_person_cam', 'side_cam'],
                                            'state_dim': 8  # FR3: 7 DOF arm + 1 DOF gripper
                                                },

}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2

### Monte01 robot gripper constants
MONTE01_GRIPPER_OPEN = 0.074  # meters
MONTE01_GRIPPER_CLOSE = 0.0   # meters
MONTE01_GRIPPER_NORMALIZE_FN = lambda x: np.clip(x / MONTE01_GRIPPER_OPEN, 0, 1)
MONTE01_GRIPPER_UNNORMALIZE_FN = lambda x: np.clip(x * MONTE01_GRIPPER_OPEN, 0, MONTE01_GRIPPER_OPEN)
