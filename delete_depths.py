#!/usr/bin/env python3

import os
import shutil
import glob
from pathlib import Path

def delete_depths_folders():
    """删除 /home/hanyu/Data/peg_in_hole/episode_*/depths 文件夹"""
    
    base_dir = "/home/hanyu/Data/peg_in_hole"
    
    if not os.path.exists(base_dir):
        print(f"错误：基础目录 {base_dir} 不存在")
        return
    
    # 查找所有 episode_* 目录
    episode_dirs = glob.glob(os.path.join(base_dir, "episode_*"))
    episode_dirs.sort()
    
    print(f"找到 {len(episode_dirs)} 个episode目录")
    
    deleted_count = 0
    total_size_freed = 0
    
    for episode_dir in episode_dirs:
        depths_dir = os.path.join(episode_dir, "depths")
        
        if os.path.exists(depths_dir) and os.path.isdir(depths_dir):
            # 计算文件夹大小
            folder_size = get_folder_size(depths_dir)
            
            print(f"删除: {depths_dir} (大小: {format_size(folder_size)})")
            
            try:
                shutil.rmtree(depths_dir)
                deleted_count += 1
                total_size_freed += folder_size
            except Exception as e:
                print(f"删除失败 {depths_dir}: {e}")
        else:
            print(f"跳过: {depths_dir} (不存在或非目录)")
    
    print(f"\n删除总结:")
    print(f"- 成功删除 {deleted_count} 个depths文件夹")
    print(f"- 释放空间: {format_size(total_size_freed)}")

def get_folder_size(folder_path):
    """计算文件夹大小（字节）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception as e:
        print(f"计算大小时出错 {folder_path}: {e}")
    return total_size

def format_size(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_bytes = float(size_bytes)
    i = 0
    
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def main():
    """主函数"""
    print("FR3数据深度文件夹删除脚本")
    print("=" * 50)
    
    # 确认操作
    response = input("确认要删除所有 /home/hanyu/Data/peg_in_hole/episode_*/depths 文件夹？(y/N): ")
    
    if response.lower() in ['y', 'yes']:
        delete_depths_folders()
    else:
        print("操作已取消")

if __name__ == "__main__":
    main()