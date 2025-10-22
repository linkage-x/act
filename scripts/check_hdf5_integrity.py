#!/usr/bin/env python3
"""
HDF5文件完整性检查脚本
用于检查ACT训练数据中的HDF5文件是否存在数据长度不一致问题
"""

import h5py
import sys
import os
import glob
import argparse

def check_hdf5_file(file_path):
    """检查单个HDF5文件的数据完整性"""
    try:
        with h5py.File(file_path, 'r') as root:
            print(f'📁 检查文件: {os.path.basename(file_path)}')

            # 检查action数据
            action_shape = root['/action'].shape
            episode_length = action_shape[0]
            print(f'   Action shape: {action_shape}, Episode length: {episode_length}')

            # 检查observations数据
            qpos_shape = root['/observations/qpos'].shape
            qvel_shape = root['/observations/qvel'].shape
            print(f'   qpos shape: {qpos_shape}')
            print(f'   qvel shape: {qvel_shape}')

            # 检查图像数据
            camera_shapes = {}
            if '/observations/images' in root:
                images_group = root['/observations/images']
                for cam_name in images_group.keys():
                    img_shape = images_group[cam_name].shape
                    camera_shapes[cam_name] = img_shape
                    print(f'   {cam_name} images shape: {img_shape}')

            # 检查各个数据集的长度是否一致
            lengths = {
                'action': action_shape[0],
                'qpos': qpos_shape[0],
                'qvel': qvel_shape[0]
            }

            if '/observations/images' in root:
                images_group = root['/observations/images']
                for cam_name in images_group.keys():
                    lengths[f'{cam_name}_images'] = images_group[cam_name].shape[0]

            print(f'   数据长度对比: {lengths}')

            # 检查是否有长度不一致的情况
            unique_lengths = set(lengths.values())
            if len(unique_lengths) > 1:
                print(f'   ⚠️ 警告: 发现长度不一致! {lengths}')

                # 找出最短的长度作为参考
                min_length = min(lengths.values())
                max_length = max(lengths.values())
                print(f'   📊 长度范围: {min_length} - {max_length}')

                # 详细分析哪些数据集有问题
                problematic_datasets = []
                for dataset, length in lengths.items():
                    if length != min_length:
                        problematic_datasets.append(f"{dataset}({length})")

                print(f'   🔍 问题数据集: {", ".join(problematic_datasets)}')
                return False, min_length, max_length
            else:
                print(f'   ✅ 所有数据长度一致: {episode_length}')
                return True, episode_length, episode_length

    except Exception as e:
        print(f'   ❌ 文件读取错误: {e}')
        return False, 0, 0

def check_directory(data_dir):
    """检查整个目录下的所有HDF5文件"""
    print(f'🔍 检查目录: {data_dir}')
    print('=' * 80)

    # 查找所有HDF5文件
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, 'episode_*.hdf5')))

    if not hdf5_files:
        print('❌ 未找到任何episode_*.hdf5文件')
        return

    print(f'📂 发现 {len(hdf5_files)} 个HDF5文件')
    print()

    # 统计结果
    valid_files = []
    invalid_files = []

    for file_path in hdf5_files:
        is_valid, min_len, max_len = check_hdf5_file(file_path)

        if is_valid:
            valid_files.append((file_path, min_len))
        else:
            invalid_files.append((file_path, min_len, max_len))

        print()

    # 输出总结
    print('=' * 80)
    print('📊 检查总结:')
    print(f'   ✅ 有效文件: {len(valid_files)}')
    print(f'   ❌ 无效文件: {len(invalid_files)}')

    if invalid_files:
        print('\n🚨 有问题的文件列表:')
        for file_path, min_len, max_len in invalid_files:
            filename = os.path.basename(file_path)
            print(f'   - {filename}: 长度范围 {min_len}-{max_len}')

        print('\n💡 建议:')
        print('   1. 检查数据转换脚本是否正确处理了数据对齐')
        print('   2. 重新运行预处理脚本修复这些文件')
        print('   3. 或者在训练时跳过这些损坏的文件')

def check_specific_files(file_list):
    """检查指定的文件列表"""
    print('🔍 检查指定的HDF5文件:')
    print('=' * 80)

    for file_path in file_list:
        if os.path.exists(file_path):
            check_hdf5_file(file_path)
            print()
        else:
            print(f'❌ 文件不存在: {file_path}')
            print()

def main():
    parser = argparse.ArgumentParser(description='检查HDF5文件的数据完整性')
    parser.add_argument('--dir', type=str, help='要检查的数据目录路径')
    parser.add_argument('--files', nargs='+', help='要检查的具体文件路径列表')
    parser.add_argument('--problem-files', action='store_true',
                       help='检查已知有问题的文件')

    args = parser.parse_args()

    if args.problem_files:
        # 检查已知有问题的文件
        problem_files = [
            '/boot/common_data/2025/fr3/1021_insert_tube_fr3_pika_150eps_hdf5/episode_42.hdf5',
            '/boot/common_data/2025/fr3/1021_insert_tube_fr3_pika_150eps_hdf5/episode_54.hdf5',
            '/boot/common_data/2025/fr3/1021_insert_tube_fr3_pika_150eps_hdf5/episode_6.hdf5'
        ]
        check_specific_files(problem_files)
    elif args.files:
        check_specific_files(args.files)
    elif args.dir:
        check_directory(args.dir)
    else:
        print('❌ 请指定要检查的目录 (--dir) 或文件 (--files) 或使用 --problem-files')
        parser.print_help()

if __name__ == '__main__':
    main()