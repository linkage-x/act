import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import yaml

from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy

# Import new HDF5Loader
from dataset.hdf5_loader import HDF5Loader
from dataset.reader import ActionType, ObservationType

import re

def main(args):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 打印关键参数确认
    print("🚀 开始训练进程...")
    print(f"📋 任务: {args['task_name']}")
    print(f"⚙️  模式: {'评估' if args['eval'] else '训练'}")
    print(f"🎯 策略: {args['policy_class']}")
    print(f"📏 批大小: {args['batch_size']}")
    print(f"🔢 轮数: {args['num_epochs']}")
    print(f"📈 学习率: {args['lr']:.6f}")
    print()

    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters from YAML only (no SIM_TASK_CONFIGS fallback)
    is_sim = task_name[:4] == 'sim_' or task_name.startswith('fr3_')
    task_config = args.get('_task_config')
    if task_config is None:
        raise KeyError(
            f"Task config not provided for task '{task_name}'. "
            f"Ensure you load YAML via task_config_manager and pass it to main (args['_task_config'])."
        )
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config.get('num_episodes', None)  # Optional, auto-detect if not specified
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    state_dim = task_config.get('state_dim', 8)

    # Derive obs->action pair from ckpt_dir suffix, e.g., *_ee2ee, *_dee2dee, *_q2q
    # Supported tokens:
    #   - ee  : end-effector absolute pose
    #   - dee : delta end-effector pose
    #   - q   : joint position
    pair_match = re.search(r'(dee|ee|q)2(dee|ee|q)(?![a-zA-Z0-9])', ckpt_dir)
    if pair_match:
        obs_key, act_key = pair_match.group(1), pair_match.group(2)
    else:
        # Default to joint->joint if no suffix specified
        obs_key, act_key = 'q', 'q'

    # Map to enums
    if obs_key == 'dee':
        obs_type_from_suffix = ObservationType.DELTA_END_EFFECTOR_POSE
    elif obs_key == 'ee':
        obs_type_from_suffix = ObservationType.END_EFFECTOR_POSE
    else:
        obs_type_from_suffix = ObservationType.JOINT_POSITION_ONLY

    if act_key == 'dee':
        action_type_from_suffix = ActionType.END_EFFECTOR_POSE_DELTA
    elif act_key == 'ee':
        action_type_from_suffix = ActionType.END_EFFECTOR_POSE
    else:
        action_type_from_suffix = ActionType.JOINT_POSITION

    # Control mode controls how dataset constructs (obs, action)
    if (obs_key, act_key) in (('ee','ee'), ('dee','dee')):
        # EE pose (absolute or delta): [x, y, z, qx, qy, qz, qw, gripper] = 8 dimensions
        state_dim = 8
        print(f"Using {'dee2dee' if obs_key=='dee' else 'ee2ee'} mode inferred from ckpt_dir, state_dim set to {state_dim}")
    elif obs_key == 'q' and act_key == 'q':
        # Joint position mode: use state_dim from task config (default: 8 for FR3)
        print(f"Using q2q (joint position) mode inferred from ckpt_dir, state_dim = {state_dim}")
    else:
        # Mixed modes (ee2q or q2ee) require dataset/model changes; guard for now
        raise NotImplementedError(
            f"Requested mixed mode '{obs_key}2{act_key}' inferred from ckpt_dir is not supported yet."
        )

    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'dropout': args.get('dropout', 0.1),
                         'weight_decay': args.get('weight_decay', 1e-4),
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'device': device,
        'stats': None  # Will be filled after data loading
    }

    # 从 task_config 获取数据增强配置
    augmentation_config = task_config.get('augmentation_config')
    if augmentation_config:
        print(f"✅ 已从任务配置加载数据增强配置")
        # 显示启用的增强类型
        if augmentation_config.get('lighting_augmentation', {}).get('enabled'):
            print(f"   - 光照增强: 已启用")
        if augmentation_config.get('rotation_augmentation', {}).get('enabled'):
            print(f"   - 旋转增强: 已启用")
    else:
        print(f"⚠️  未配置数据增强")

    # Create HDF5 data loader with control mode support
    # Determine action type and observation type from control mode
    action_type = action_type_from_suffix
    observation_type = obs_type_from_suffix

    hdf5_loader_config = {
        'num_episodes': num_episodes,
        'camera_names': camera_names,
        'batch_size_train': batch_size_train,
        'batch_size_val': batch_size_val,
        'episode_len': episode_len,
        'augmentation_config': augmentation_config,
    }

    hdf5_loader = HDF5Loader(
        config=hdf5_loader_config,
        dataset_dir=dataset_dir,
        action_type=action_type,
        observation_type=observation_type
    )

    train_dataloader, val_dataloader, stats, _ = hdf5_loader.create_dataloaders()

    # Convert stats tensors to numpy for consistent serialization
    def _stats_to_numpy(d):
        out = {}
        for k, v in d.items():
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    out[k] = v.detach().cpu().numpy()
                else:
                    out[k] = v
            except Exception:
                out[k] = v
        return out
    stats_np = _stats_to_numpy(stats)
    # Add stats to config for eval
    config['stats'] = stats_np

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats_np, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, device):
    """
    Get images from observation and return as dict for policy inference

    Returns:
        dict: {cam_name: torch.Tensor (C, H, W)} on specified device
    """
    image_dict = {}
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        # Normalize to [0, 1] and convert to torch tensor
        curr_image = torch.from_numpy(curr_image / 255.0).float().to(device)
        image_dict[cam_name] = curr_image
    return image_dict

def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    device = config['device']

    set_seed(seed)

    print("🏗️  初始化模型和优化器...")
    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    # 打印模型和优化器信息
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"🧠 模型参数统计:")
    print(f"  总参数数量:   {total_params/1e6:.2f}M")
    print(f"  可训练参数:   {trainable_params/1e6:.2f}M")
    print(f"  参数利用率:   {trainable_params/total_params*100:.1f}%")

    print(f"⚡ 优化器配置:")
    print(f"  优化器类型:   {type(optimizer).__name__}")
    for group in optimizer.param_groups:
        print(f"  学习率:       {group['lr']:.8f}")
        if 'weight_decay' in group:
            print(f"  权重衰减:     {group['weight_decay']}")
        break  # 只打印第一个参数组
    print()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    print("=" * 60)
    print("🎓 开始训练循环")
    print("=" * 60)

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, device)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        train_epoch_dicts = []  # Per-epoch accumulator
        for _, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            train_epoch_dicts.append(detach_dict(forward_dict))
        # Compute epoch summary from this epoch's batches
        epoch_summary = compute_dict_mean(train_epoch_dicts)
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()  # 释放内存，防止内存泄漏
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ACT训练脚本 - 使用配置文件管理参数')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (例如: configs/tasks/fr3_bs_0916_50ep_ds.yaml)')

    args = parser.parse_args()

    # 加载配置
    try:
        from task_config_manager import load_task_config
        config_data = load_task_config(args.config, eval_mode=False)

        print(f"✓ 配置文件加载成功: {args.config}")
        print(f"  任务名: {config_data['args']['task_name']}")
        print(f"  机器人: {config_data['config']['robot']['name']}")
        print(f"  数据集: {config_data['task_config']['dataset_dir']}")
        print(f"  模式: 训练")
        print()

        # 详细打印训练超参数
        print("=" * 60)
        print("🔧 训练超参数配置")
        print("=" * 60)

        args_data = config_data['args']
        task_data = config_data['task_config']

        print("📊 基础训练参数:")
        print(f"  策略类型:     {args_data['policy_class']}")
        print(f"  批大小:       {args_data['batch_size']}")
        print(f"  训练轮数:     {args_data['num_epochs']}")
        print(f"  学习率:       {args_data['lr']:.6f}")
        print(f"  随机种子:     {args_data['seed']}")
        print()

        print("🤖 ACT模型参数:")
        print(f"  KL权重:       {args_data['kl_weight']}")
        print(f"  动作块大小:   {args_data['chunk_size']}")
        print(f"  隐藏层维度:   {args_data['hidden_dim']}")
        print(f"  前馈网络维度: {args_data['dim_feedforward']}")
        print(f"  Dropout:      {args_data.get('dropout', 0.1)}")
        print(f"  权重衰减:     {args_data.get('weight_decay', 1e-4)}")
        print()

        print("📁 数据配置:")
        print(f"  状态维度:     {task_data['state_dim']}")
        print(f"  Episode长度:  {task_data['episode_len']}")
        print(f"  Episode数量:  {task_data.get('num_episodes', 'Auto-detect')}")
        print(f"  相机:         {task_data['camera_names']}")
        print()

        print("💾 存储配置:")
        print(f"  检查点目录:   {args_data['ckpt_dir']}")
        print(f"  时序聚合:     {args_data['temporal_agg']}")
        print(f"  屏幕渲染:     {args_data['onscreen_render']}")
        print()

        print("=" * 60)
        print()

        # 运行主函数（统一从 YAML 注入任务配置）
        args_data = config_data['args']
        args_data['_task_config'] = config_data['task_config']
        main(args_data)

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        print("请检查配置文件路径和格式是否正确")
        sys.exit(1)
