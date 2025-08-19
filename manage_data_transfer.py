#!/usr/bin/env python3

import os
import time
import subprocess
import shutil

def check_transfer_complete():
    """检查rsync是否完成"""
    try:
        result = subprocess.run(['pgrep', '-f', 'rsync.*peg_in_hole_hdf5_extended'], 
                               capture_output=True, text=True)
        return len(result.stdout.strip()) == 0
    except:
        return True

def create_symlinks():
    """创建软链接指向外部存储"""
    
    # 等待传输完成
    print("等待数据传输完成...")
    while not check_transfer_complete():
        time.sleep(30)
        print("传输仍在进行中...")
    
    print("传输完成！开始创建软链接...")
    
    # 删除本地目录
    local_dir = "data/peg_in_hole_hdf5_extended"
    external_dir = "/media/hanyu/ubuntu/act_project/peg_in_hole_hdf5_extended"
    
    if os.path.exists(local_dir):
        print(f"删除本地目录: {local_dir}")
        shutil.rmtree(local_dir)
    
    # 创建软链接
    print(f"创建软链接: {local_dir} -> {external_dir}")
    os.symlink(external_dir, local_dir)
    
    # 验证软链接
    if os.path.islink(local_dir):
        print("✓ 软链接创建成功")
        print(f"链接目标: {os.readlink(local_dir)}")
        
        # 检查文件数量
        file_count = len([f for f in os.listdir(local_dir) if f.endswith('.hdf5')])
        print(f"可用HDF5文件数: {file_count}")
        
        # 检查磁盘空间
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        print("磁盘空间状态:")
        print(result.stdout.split('\n')[1])
        
    else:
        print("✗ 软链接创建失败")

if __name__ == "__main__":
    create_symlinks()