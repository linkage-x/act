#!/usr/bin/env python3

import re
import sys
import time
import os
from torch.utils.tensorboard import SummaryWriter

def parse_and_log_training():
    """Parse training output in real-time and log to TensorBoard"""
    
    # Create TensorBoard writer
    tensorboard_dir = 'ckpts/fr3_peg_in_hole_act/tensorboard_logs'
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    print(f"TensorBoard logging started. Run:")
    print(f"tensorboard --logdir {tensorboard_dir} --port 6006")
    print(f"Then open: http://localhost:6006")
    print("-" * 60)
    
    # Patterns to match training output
    epoch_pattern = r'Epoch (\d+)'
    val_pattern = r'Val loss:\s+([\d.]+)\s+l1:\s+([\d.]+)\s+kl:\s+([\d.]+)\s+loss:\s+([\d.]+)'
    train_pattern = r'Train loss:\s+([\d.]+)\s+l1:\s+([\d.]+)\s+kl:\s+([\d.]+)\s+loss:\s+([\d.]+)'
    
    current_epoch = None
    
    try:
        for line in sys.stdin:
            line = line.strip()
            
            # Parse epoch
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                writer.add_scalar('Training/Current_Epoch', current_epoch, current_epoch)
                print(f"Epoch {current_epoch}")
            
            # Parse validation metrics
            val_match = re.search(val_pattern, line)
            if val_match and current_epoch is not None:
                val_loss, val_l1, val_kl, val_total = map(float, val_match.groups())
                
                writer.add_scalar('Loss/Validation_Total', val_total, current_epoch)
                writer.add_scalar('Loss/Validation_L1', val_l1, current_epoch)
                writer.add_scalar('Loss/Validation_KL', val_kl, current_epoch)
                
                print(f"  Val - Total: {val_total:.3f}, L1: {val_l1:.3f}, KL: {val_kl:.3f}")
            
            # Parse training metrics  
            train_match = re.search(train_pattern, line)
            if train_match and current_epoch is not None:
                train_loss, train_l1, train_kl, train_total = map(float, train_match.groups())
                
                writer.add_scalar('Loss/Training_Total', train_total, current_epoch)
                writer.add_scalar('Loss/Training_L1', train_l1, current_epoch)
                writer.add_scalar('Loss/Training_KL', train_kl, current_epoch)
                
                # Combined plots
                writer.add_scalars('Loss/Total_Loss', {
                    'Train': train_total,
                    'Validation': val_total if 'val_total' in locals() else 0
                }, current_epoch)
                
                writer.add_scalars('Loss/L1_Loss', {
                    'Train': train_l1,
                    'Validation': val_l1 if 'val_l1' in locals() else 0
                }, current_epoch)
                
                writer.add_scalars('Loss/KL_Loss', {
                    'Train': train_kl,
                    'Validation': val_kl if 'val_kl' in locals() else 0
                }, current_epoch)
                
                print(f"  Train - Total: {train_total:.3f}, L1: {train_l1:.3f}, KL: {train_kl:.3f}")
                
                writer.flush()
    
    except KeyboardInterrupt:
        print("\nTensorBoard logging stopped")
    finally:
        writer.close()

if __name__ == '__main__':
    parse_and_log_training()