#!/bin/bash

set -e  # 遇到错误立即退出

# 默认配置
DEFAULT_BASE_DIR='/boot/common_data/2025/fr3'
SCRIPT_NAME='scripts/convert_fr3_preprocess.py'

# 参数检查
if [ $# -lt 1 ]; then
    echo "❌ 错误: 缺少必要参数"
    echo "用法: $0 <dataset_name> [base_data_dir]"
    echo "示例: $0 fr3_pp_1013 /boot/common_data/2025/fr3"
    exit 1
fi

# 获取参数
DATASET_NAME="$1"
BASE_DIR="${2:-$DEFAULT_BASE_DIR}"
INPUT_DIR="$BASE_DIR/$DATASET_NAME"
OUTPUT_DIR="$BASE_DIR/${DATASET_NAME}_hdf5"

echo "🔧 开始数据预处理..."
echo "  数据集: $DATASET_NAME"
echo "  输入目录: $INPUT_DIR"
echo "  输出目录: $OUTPUT_DIR"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ 错误: 找不到预处理脚本: $SCRIPT_NAME"
    echo "请确保在正确的目录下运行此脚本"
    exit 1
fi

# 创建输出目录
sudo mkdir -p "$OUTPUT_DIR"
sudo chown -R $user:$user "$OUTPUT_DIR"

# 运行预处理
echo "🚀 执行数据下采样..."
python "$SCRIPT_NAME" --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 数据预处理完成"
    echo "  处理后数据保存在: $OUTPUT_DIR"
else
    echo "❌ 数据预处理失败"
    exit 1
fi