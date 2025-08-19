#!/usr/bin/env python3

import os
import re
import time
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import subprocess

def parse_training_log(log_file, writer, start_epoch=0):
    """Parse training log and write to TensorBoard"""
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return start_epoch
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Regex patterns to extract metrics
    epoch_pattern = r'Epoch (\d+)'
    val_pattern = r'Val loss:\s+([\d.]+)\s+l1:\s+([\d.]+)\s+kl:\s+([\d.]+)\s+loss:\s+([\d.]+)'
    train_pattern = r'Train loss:\s+([\d.]+)\s+l1:\s+([\d.]+)\s+kl:\s+([\d.]+)\s+loss:\s+([\d.]+)'
    
    epochs = re.findall(epoch_pattern, content)
    val_matches = re.findall(val_pattern, content)
    train_matches = re.findall(train_pattern, content)
    
    last_epoch = start_epoch
    
    for i, epoch_str in enumerate(epochs):
        epoch = int(epoch_str)
        if epoch <= start_epoch:
            continue
            
        last_epoch = epoch
        
        # Write validation metrics
        if i < len(val_matches):
            val_loss, val_l1, val_kl, val_total = val_matches[i]
            writer.add_scalar('Loss/Validation_Total', float(val_total), epoch)
            writer.add_scalar('Loss/Validation_L1', float(val_l1), epoch)
            writer.add_scalar('Loss/Validation_KL', float(val_kl), epoch)
        
        # Write training metrics
        if i < len(train_matches):
            train_loss, train_l1, train_kl, train_total = train_matches[i]
            writer.add_scalar('Loss/Training_Total', float(train_total), epoch)
            writer.add_scalar('Loss/Training_L1', float(train_l1), epoch)
            writer.add_scalar('Loss/Training_KL', float(train_kl), epoch)
            
            # Log combined metrics
            writer.add_scalars('Loss/Total_Loss', {
                'Train': float(train_total),
                'Validation': float(val_total) if i < len(val_matches) else 0
            }, epoch)
            
            writer.add_scalars('Loss/L1_Loss', {
                'Train': float(train_l1),
                'Validation': float(val_l1) if i < len(val_matches) else 0
            }, epoch)
            
            writer.add_scalars('Loss/KL_Loss', {
                'Train': float(train_kl),
                'Validation': float(val_kl) if i < len(val_matches) else 0
            }, epoch)
    
    writer.flush()
    return last_epoch

def monitor_training(ckpt_dir, refresh_interval=30):
    """Monitor training progress and update TensorBoard logs"""
    
    # Create TensorBoard log directory
    tensorboard_dir = os.path.join(ckpt_dir, 'tensorboard_logs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    writer = SummaryWriter(tensorboard_dir)
    
    print(f"TensorBoard monitoring started")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Run: tensorboard --logdir {tensorboard_dir} --port 6006")
    print(f"Then open: http://localhost:6006")
    print("-" * 60)
    
    last_epoch = 0
    
    try:
        while True:
            # Monitor training output (we'll capture from the running process)
            # For now, let's create a simple log parser that reads from stdout
            
            # Since we can't directly access the running process stdout,
            # we'll create a simple metrics tracker based on saved plots
            print(f"Checking for new training data... (last epoch: {last_epoch})")
            
            # Look for any training log files or use a different approach
            # In this case, we'll parse the checkpoint directory for progress
            
            # List all checkpoint files to estimate progress
            ckpt_files = list(Path(ckpt_dir).glob('policy_epoch_*.ckpt'))
            if ckpt_files:
                # Extract epoch numbers from checkpoint files
                epoch_nums = []
                for ckpt_file in ckpt_files:
                    match = re.search(r'policy_epoch_(\d+)_seed_0\.ckpt', ckpt_file.name)
                    if match:
                        epoch_nums.append(int(match.group(1)))
                
                if epoch_nums:
                    current_max_epoch = max(epoch_nums)
                    if current_max_epoch > last_epoch:
                        print(f"Training progress: Epoch {current_max_epoch}")
                        
                        # Add a simple progress metric
                        writer.add_scalar('Training/Progress_Epochs', current_max_epoch, current_max_epoch)
                        writer.add_scalar('Training/Checkpoints_Saved', len(epoch_nums), current_max_epoch)
                        
                        last_epoch = current_max_epoch
            
            writer.flush()
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    finally:
        writer.close()

def start_tensorboard_server(tensorboard_dir, port=6006):
    """Start TensorBoard server"""
    try:
        cmd = f"tensorboard --logdir {tensorboard_dir} --port {port} --host 0.0.0.0"
        print(f"Starting TensorBoard server: {cmd}")
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"Failed to start TensorBoard: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Monitor ACT training with TensorBoard')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts/fr3_peg_in_hole_act',
                        help='Checkpoint directory to monitor')
    parser.add_argument('--refresh_interval', type=int, default=30,
                        help='Refresh interval in seconds')
    parser.add_argument('--port', type=int, default=6006,
                        help='TensorBoard port')
    parser.add_argument('--start_tensorboard', action='store_true',
                        help='Automatically start TensorBoard server')
    
    args = parser.parse_args()
    
    tensorboard_dir = os.path.join(args.ckpt_dir, 'tensorboard_logs')
    
    # Start TensorBoard server if requested
    tensorboard_process = None
    if args.start_tensorboard:
        tensorboard_process = start_tensorboard_server(tensorboard_dir, args.port)
        if tensorboard_process:
            print(f"TensorBoard started on http://localhost:{args.port}")
        time.sleep(2)  # Give TensorBoard time to start
    
    try:
        # Start monitoring
        monitor_training(args.ckpt_dir, args.refresh_interval)
    except KeyboardInterrupt:
        pass
    finally:
        if tensorboard_process:
            print("Stopping TensorBoard server...")
            tensorboard_process.terminate()

if __name__ == '__main__':
    main()