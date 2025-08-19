#!/usr/bin/env python3

import os
import time
import subprocess
from pathlib import Path

class TransferMonitor:
    def __init__(self):
        self.local_dir = Path("data/peg_in_hole_hdf5_extended")
        self.external_dir = Path("/media/hanyu/ubuntu/act_project/peg_in_hole_hdf5_extended")
        
    def get_file_count_and_size(self, directory):
        """获取目录中HDF5文件数量和总大小"""
        if not directory.exists():
            return 0, 0
        
        hdf5_files = list(directory.glob("*.hdf5"))
        file_count = len(hdf5_files)
        
        total_size = 0
        for file in hdf5_files:
            try:
                total_size += file.stat().st_size
            except:
                pass
                
        return file_count, total_size
    
    def format_size(self, size_bytes):
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
    
    def check_rsync_status(self):
        """检查rsync是否还在运行"""
        try:
            result = subprocess.run(['pgrep', '-f', 'rsync.*peg_in_hole_hdf5_extended'], 
                                   capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except:
            return False
    
    def get_disk_usage(self):
        """获取磁盘使用情况"""
        try:
            result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                return parts[3], parts[4]  # 可用空间, 使用百分比
        except:
            pass
        return "Unknown", "Unknown"
    
    def monitor(self, interval=30):
        """监控传输进度"""
        print("🚀 FR3数据传输监控器启动")
        print("=" * 60)
        
        start_time = time.time()
        
        while True:
            # 清屏（可选）
            os.system('clear')
            
            print("🚀 FR3数据传输监控器")
            print("=" * 60)
            print(f"监控时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 检查rsync状态
            rsync_running = self.check_rsync_status()
            status_icon = "🔄" if rsync_running else "✅"
            status_text = "传输中" if rsync_running else "已完成"
            print(f"传输状态: {status_icon} {status_text}")
            
            # 本地文件统计
            local_count, local_size = self.get_file_count_and_size(self.local_dir)
            print(f"本地文件: {local_count} 个HDF5文件 ({self.format_size(local_size)})")
            
            # 外部存储文件统计
            external_count, external_size = self.get_file_count_and_size(self.external_dir)
            print(f"已传输: {external_count} 个HDF5文件 ({self.format_size(external_size)})")
            
            # 进度计算
            if local_count > 0:
                progress = (external_count / (local_count + external_count)) * 100
                print(f"传输进度: {progress:.1f}% ({external_count}/{local_count + external_count})")
            
            # 磁盘空间
            available, used_pct = self.get_disk_usage()
            print(f"本地磁盘: {available} 可用 (已用 {used_pct})")
            
            # 传输速度估算（简单版本）
            elapsed = time.time() - start_time
            if elapsed > 0 and external_size > 0:
                speed = external_size / elapsed  # bytes/second
                print(f"平均速度: {self.format_size(speed)}/s")
            
            print("-" * 60)
            
            # 显示一些具体文件（最新传输的）
            if external_count > 0:
                try:
                    hdf5_files = sorted(self.external_dir.glob("*.hdf5"), 
                                       key=lambda x: x.stat().st_mtime, reverse=True)[:5]
                    print("最近传输的文件:")
                    for i, file in enumerate(hdf5_files[:5], 1):
                        size = self.format_size(file.stat().st_size)
                        mtime = time.strftime('%H:%M:%S', time.localtime(file.stat().st_mtime))
                        print(f"  {i}. {file.name} ({size}) - {mtime}")
                except:
                    pass
            
            print("=" * 60)
            
            # 如果传输完成，退出监控
            if not rsync_running and external_count > 0:
                print("🎉 传输完成！")
                self.show_final_summary()
                break
                
            # 等待下一次检查
            if rsync_running:
                print(f"⏳ {interval}秒后刷新... (Ctrl+C 退出)")
                try:
                    time.sleep(interval)
                except KeyboardInterrupt:
                    print("\n\n👋 监控已停止")
                    break
            else:
                break
    
    def show_final_summary(self):
        """显示最终统计"""
        external_count, external_size = self.get_file_count_and_size(self.external_dir)
        local_count, local_size = self.get_file_count_and_size(self.local_dir)
        
        print("\n📊 传输完成统计:")
        print(f"✅ 已传输: {external_count} 个episode ({self.format_size(external_size)})")
        print(f"📁 本地剩余: {local_count} 个episode ({self.format_size(local_size)})")
        print(f"🎯 数据扩展: 从13个episode扩展到{external_count + local_count}个episode")
        print(f"💾 存储位置: {self.external_dir}")
        
        if local_count == 0:
            print("\n🔗 准备创建软链接...")
            self.create_symlink()
    
    def create_symlink(self):
        """创建软链接"""
        try:
            # 删除本地目录（如果存在）
            if self.local_dir.exists():
                import shutil
                shutil.rmtree(self.local_dir)
                print(f"✅ 已删除本地目录: {self.local_dir}")
            
            # 创建软链接
            self.local_dir.symlink_to(self.external_dir)
            print(f"✅ 软链接创建成功: {self.local_dir} -> {self.external_dir}")
            
            # 验证
            if self.local_dir.is_symlink():
                count, size = self.get_file_count_and_size(self.local_dir)
                print(f"✅ 验证成功: {count} 个文件可访问 ({self.format_size(size)})")
                return True
            else:
                print("❌ 软链接验证失败")
                return False
                
        except Exception as e:
            print(f"❌ 创建软链接失败: {e}")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='监控FR3数据传输')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='刷新间隔（秒），默认30秒')
    parser.add_argument('--once', '-o', action='store_true',
                       help='只检查一次，不持续监控')
    
    args = parser.parse_args()
    
    monitor = TransferMonitor()
    
    if args.once:
        # 单次检查
        monitor.monitor_once()
    else:
        # 持续监控
        monitor.monitor(args.interval)