#!/usr/bin/env python3

import os
from pathlib import Path

def check_episode_completeness(base_dir):
    """检查episode数据完整性"""
    base_path = Path(base_dir)
    episode_dirs = sorted([d for d in base_path.iterdir() 
                          if d.is_dir() and d.name.startswith('episode_')])
    
    complete_episodes = []
    incomplete_episodes = []
    
    for episode_dir in episode_dirs:
        episode_num = int(episode_dir.name.split('_')[1])
        colors_dir = episode_dir / 'colors'
        data_json = episode_dir / 'data.json'
        
        if colors_dir.exists() and data_json.exists():
            complete_episodes.append(episode_num)
            print(f"✓ Episode {episode_num:04d} - COMPLETE")
        else:
            incomplete_episodes.append(episode_num)
            missing = []
            if not colors_dir.exists():
                missing.append("colors")
            if not data_json.exists():
                missing.append("data.json")
            print(f"✗ Episode {episode_num:04d} - MISSING: {', '.join(missing)}")
    
    print(f"\n=== 总结 ===")
    print(f"总episode数: {len(episode_dirs)}")
    print(f"完整episode数: {len(complete_episodes)}")
    print(f"不完整episode数: {len(incomplete_episodes)}")
    
    if incomplete_episodes:
        print(f"不完整episode编号: {incomplete_episodes}")
    
    return complete_episodes, incomplete_episodes

if __name__ == "__main__":
    base_dir = "/home/hanyu/Data/peg_in_hole"
    complete, incomplete = check_episode_completeness(base_dir)