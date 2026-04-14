#!/bin/bash
# SenseVoice API 启动脚本

export SENSEVOICE_DEVICE=cpu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 如果子项目 venv 不存在，自动创建并安装依赖
if [ ! -d ".venv" ]; then
    echo "创建 SenseVoice venv..."
    uv venv -p python3.11 .venv
    uv pip install -r requirements.txt -p .venv
fi

exec .venv/bin/uvicorn api:app --host 0.0.0.0 --port 50000 --timeout-keep-alive 300
