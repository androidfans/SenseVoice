#!/bin/bash
# SenseVoice API 启动脚本
# 使用方法: ./start_api.sh

export SENSEVOICE_DEVICE=cpu

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 启动API服务
exec uvicorn api:app --host 0.0.0.0 --port 50000 --timeout-keep-alive 300
