#!/bin/bash

# 定义 Conda 环境名称
conda_env="probsky"

# 检查 Conda 是否已安装
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# 检查指定的 Conda 环境是否存在
if ! conda info --envs | grep -q "$conda_env"; then
    echo "Conda environment '$conda_env' does not exist. Please create it first."
    exit 1
fi

# 激活 Conda 环境
echo "Activating Conda environment: $conda_env"
conda activate "$conda_env"

# 执行其他操作，例如运行 Python 脚本或其他命令
# Your commands here

python GPU_GRID3_gridcmp2.py

