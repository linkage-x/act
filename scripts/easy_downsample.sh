et -e  # 遇到错误立即退出

# 检查参数
if [ $# -eq 0 ]; then
	    echo "错误: 请提供数据集名称"
	        echo "用法: $0 <数据集名称>"
		    echo "例如: $0 fr3_liquid_transfer_0920_50ep"
		        exit 1
fi

# 设置变量
DATASET_NAME=$1
INPUT_DIR="/boot/common_data/${DATASET_NAME}"
OUTPUT_DIR="/boot/common_data/${DATASET_NAME}_ds_hdf5"
SCREEN_NAME="${DATASET_NAME}_ds"
USERNAME=$(whoami)

echo "=========================================="
echo "FR3 数据转换脚本"
echo "=========================================="
echo "数据集名称: $DATASET_NAME"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Screen会话: $SCREEN_NAME"
echo "当前用户: $USERNAME"
echo "=========================================="

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
	    echo "错误: 输入目录不存在: $INPUT_DIR"
	        exit 1
fi

# 检查输出目录是否已存在
if [ -d "$OUTPUT_DIR" ]; then
	    echo "警告: 输出目录已存在: $OUTPUT_DIR"
	        read -p "是否要删除现有目录并重新创建? (y/N): " -n 1 -r
		    echo
		        if [[ $REPLY =~ ^[Yy]$ ]]; then
				        echo "删除现有输出目录..."
					        sudo rm -rf "$OUTPUT_DIR"
						    else
							            echo "取消操作"
								            exit 1
									        fi
fi

# 步骤1: 创建输出目录
echo "步骤1: 创建输出目录..."
sudo mkdir -p "$OUTPUT_DIR"

# 步骤2: 修改目录权限
echo "步骤2: 修改目录权限..."
sudo chown -R ${USERNAME}:${USERNAME} "$OUTPUT_DIR"

# 步骤3: 检查conda环境
echo "步骤3: 检查conda环境..."
if ! command -v conda &> /dev/null; then
	    echo "错误: conda未找到，请确保conda已正确安装"
	        exit 1
fi

# 检查aloha环境是否存在
if ! conda info --envs | grep -q "aloha"; then
	    echo "警告: aloha conda环境不存在"
	        read -p "是否要继续使用当前环境? (y/N): " -n 1 -r
		    echo
		        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
				        echo "请先创建aloha环境后再运行此脚本"
					        exit 1
						    fi
						        CONDA_ENV=""
						else
							    CONDA_ENV="conda activate aloha && "
fi

# 检查代码目录
CODE_DIR="$HOME/code/act"
if [ ! -d "$CODE_DIR" ]; then
	    echo "错误: ACT代码目录不存在: $CODE_DIR"
	        exit 1
fi

if [ ! -f "$CODE_DIR/scripts/convert_fr3_preprocess.py" ]; then
	    echo "错误: 转换脚本不存在: $CODE_DIR/scripts/convert_fr3_preprocess.py"
	        exit 1
fi

# 步骤4: 检查是否有同名screen会话在运行
echo "步骤4: 检查screen会话..."
if screen -list | grep -q "$SCREEN_NAME"; then
	    echo "警告: screen会话 '$SCREEN_NAME' 已存在"
	        read -p "是否要终止现有会话并创建新的? (y/N): " -n 1 -r
		    echo
		        if [[ $REPLY =~ ^[Yy]$ ]]; then
				        screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
					        sleep 2
						    else
							            echo "取消操作"
								            exit 1
									        fi
fi

# 步骤5: 创建并运行screen会话
echo "步骤5: 启动数据转换..."
echo "创建screen会话: $SCREEN_NAME"

# 创建临时脚本文件
TEMP_SCRIPT="/tmp/convert_${DATASET_NAME}_$(date +%s).sh"
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
set -e

echo "=========================================="
echo "开始数据转换..."
echo "时间: \$(date)"
echo "=========================================="

# 激活conda环境并切换到代码目录
${CONDA_ENV}cd $CODE_DIR

# 执行转换
echo "执行转换命令..."
python scripts/convert_fr3_preprocess.py \\
    --input_dir "$INPUT_DIR" \\
    --output_dir "$OUTPUT_DIR"

echo "=========================================="
echo "数据转换完成!"
echo "时间: \$(date)"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 检查输出文件
echo "输出文件列表:"
ls -la "$OUTPUT_DIR"

echo ""
echo "按任意键退出screen会话..."
read -n 1
EOF

chmod +x "$TEMP_SCRIPT"

# 启动screen会话并执行脚本
screen -dmS "$SCREEN_NAME" bash -c "$TEMP_SCRIPT; exec bash"

echo ""
echo "=========================================="
echo "脚本启动完成!"
echo "=========================================="
echo "Screen会话已创建: $SCREEN_NAME"
echo "查看进度: screen -r $SCREEN_NAME"
echo "分离会话: Ctrl+A, D"
echo "终止会话: screen -S $SCREEN_NAME -X quit"
echo ""
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

# 显示screen会话列表
echo "当前screen会话:"
screen -list

# 清理临时脚本
sleep 1
rm -f "$TEMP_SCRIPT"

echo ""
echo "提示: 使用 'screen -r $SCREEN_NAME' 查看转换进度"
