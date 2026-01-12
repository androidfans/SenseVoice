#!/bin/bash
# SenseVoice API 启动脚本

export SENSEVOICE_DEVICE=cpu
cd /Users/rejectliu/Projects/VideoFlow/third_party/sensevoice/SenseVoice
exec /Users/rejectliu/Projects/VideoFlow/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 50000 --timeout-keep-alive 300
