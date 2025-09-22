#!/bin/bash

# FR3 数据处理自动化脚本
# 用法: ./process_fr3_data.sh <zip文件路径>
# 示例: ./process_fr3_data.sh /boot/common_data/fr3_pih_0923_25ep_fixloc.zip

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -eq 0 ]; then
    print_error "请提供 zip 文件路径"
    echo "用法: $0 <zip文件路径>"
    echo "示例: $0 /boot/common_data/fr3_pih_0923_25ep_fixloc.zip"
    exit 1
fi

ZIP_FILE="$1"
COMMON_DATA_DIR="/boot/common_data"
ACT_CODE_DIR="$HOME/code/act"
CURRENT_USER=$(whoami)

# 检查 zip 文件是否存在
if [ ! -f "$ZIP_FILE" ]; then
    print_error "文件不存在: $ZIP_FILE"
    exit 1
fi

# 获取文件名（不含扩展名）
BASENAME=$(basename "$ZIP_FILE" .zip)
EXTRACT_DIR="$COMMON_DATA_DIR/$BASENAME"
OUTPUT_DIR="${EXTRACT_DIR}_hdf5"

print_info "开始处理数据: $BASENAME"
print_info "解压目录: $EXTRACT_DIR"
print_info "输出目录: $OUTPUT_DIR"

# 步骤 1: 切换到 common_data 目录
print_info "切换到 $COMMON_DATA_DIR 目录"
cd "$COMMON_DATA_DIR"

# 步骤 2: 解压文件
print_info "解压 zip 文件..."
if [ -d "$EXTRACT_DIR" ]; then
    print_warning "目录已存在: $EXTRACT_DIR"
    read -p "是否删除并重新解压? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo rm -rf "$EXTRACT_DIR"
        print_info "已删除旧目录"
    else
        print_info "跳过解压步骤"
    fi
fi

if [ ! -d "$EXTRACT_DIR" ]; then
    sudo unzip "$ZIP_FILE" -d "$BASENAME"
    print_success "解压完成"
else
    print_info "使用现有解压目录"
fi

# 步骤 3: 创建输出目录并设置权限
print_info "准备输出目录..."
if [ -d "$OUTPUT_DIR" ]; then
    print_warning "输出目录已存在: $OUTPUT_DIR"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo rm -rf "$OUTPUT_DIR"
        print_info "已删除旧输出目录"
    fi
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    sudo mkdir -p "$OUTPUT_DIR"
    sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$OUTPUT_DIR"
    print_success "创建并设置输出目录权限"
else
    # 确保权限正确
    sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$OUTPUT_DIR"
    print_success "输出目录权限已更新"
fi

# 步骤 4: 切换到 ACT 代码目录
print_info "切换到 ACT 代码目录: $ACT_CODE_DIR"
cd "$ACT_CODE_DIR"

# 检查转换脚本是否存在
if [ ! -f "convert_fr3_robust.py" ]; then
    print_error "转换脚本不存在: $ACT_CODE_DIR/convert_fr3_robust.py"
    exit 1
fi

# 步骤 5: 激活环境并运行转换
print_info "开始数据转换..."
echo "==================== 数据转换日志 ===================="

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "未检测到 conda 环境，尝试激活 aloha 环境"
    # 尝试激活环境
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        conda activate aloha
    else
        print_error "未找到 conda，请手动激活 aloha 环境"
        exit 1
    fi
fi

# 运行转换脚本
python convert_fr3_robust.py --input_dir "$EXTRACT_DIR" --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    print_success "数据转换完成！"
    print_info "输入目录: $EXTRACT_DIR"
    print_info "输出目录: $OUTPUT_DIR"
    
    # 显示输出目录信息
    if [ -d "$OUTPUT_DIR" ]; then
        print_info "输出目录内容:"
        ls -la "$OUTPUT_DIR" | head -10
        if [ $(ls -1 "$OUTPUT_DIR" | wc -l) -gt 10 ]; then
            echo "... (还有更多文件)"
        fi
    fi
else
    print_error "数据转换失败！"
    exit 1
fi

print_success "所有步骤完成！"