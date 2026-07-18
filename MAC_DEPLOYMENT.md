# SenseVoice Mac 本地部署指南

## 踩坑记录

### 1. Docker 性能问题

**现象**: Docker容器内运行SenseVoice,5分钟音频识别需要13-16秒,而本机直接运行只需3-4秒。

**原因**:
- Mac本机PyTorch使用Apple的**Accelerate框架**进行CPU加速(BLAS_INFO=accelerate)
- Docker容器运行的是Linux系统,只能使用OpenBLAS,无法调用Accelerate
- 这是macOS虚拟化的固有限制,与Docker/OrbStack配置无关

**验证方法**:
```bash
# 本机PyTorch配置
python -c "import torch; print(torch.__config__.show())" | grep BLAS
# 输出: BLAS_INFO=accelerate ✅

# 容器内PyTorch配置
docker exec <container> python3 -c "import torch; print(torch.__config__.show())" | grep BLAS
# 输出: BLAS_INFO=open ❌
```

**矩阵运算性能对比**:
| 环境 | 2000x2000矩阵乘法 |
|-----|------------------|
| 本机 | 0.009秒 |
| 容器 | 0.050秒 |

**结论**: Docker 无法使用 Apple Accelerate 或 MPS。Apple Silicon Mac 建议本机直接运行，并优先使用 MPS。

### 2. CUDA报错

**现象**: 启动时报错 `AssertionError: Torch not compiled with CUDA enabled`

**原因**: `api.py` 的兜底 device 是 `cuda:0`，但 Mac 没有 CUDA。

**解决**: 使用 `start_api.sh` 启动时默认使用 MPS，无需额外设置。需要回退到 Accelerate CPU 时设置环境变量：

```bash
export SENSEVOICE_DEVICE=cpu
```

也可以在单次启动时指定：

```bash
SENSEVOICE_DEVICE=cpu ./start_api.sh
```

### 3. API clean_text字段只返回最后一段

**现象**: `clean_text`只有几十个字符,丢失了大部分内容

**原因**: 正则表达式使用贪婪匹配
```python
# 错误 - 贪婪匹配,会删除第一个<|到最后一个|>之间的所有内容
regex = r"<\|.*\|>"

# 正确 - 非贪婪匹配
regex = r"<\|.*?\|>"
```

### 4. 原版API不支持长音频

**现象**: 处理超过1分钟的音频时,容器崩溃或内存溢出

**原因**: 原版api.py使用`SenseVoiceSmall.from_pretrained`,不带VAD分段

**解决**: 改用`AutoModel`并启用VAD模型,参考webui.py的实现:
```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
    trust_remote_code=True,
)

# 推理时使用generate而不是inference
res = model.generate(
    input=input_wav,
    cache={},
    language=lang,
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
)
```

---

## PM2 部署 (Mac)

### 前置依赖

- [uv](https://docs.astral.sh/uv/) — 用于创建 venv 和安装依赖
- Python 3.11
- [Node.js](https://nodejs.org/) 与 PM2 (`npm install --global pm2`)
- [ffmpeg](https://ffmpeg.org/) — 用于解码 M4A 和视频容器

### 独立 venv

SenseVoice 使用**独立 venv**（`third_party/sensevoice/SenseVoice/.venv/`），与主项目 venv 隔离，避免 torch/funasr 等重型 ML 依赖污染主项目。

`start_api.sh` 会在首次启动时自动创建 venv 并安装 `requirements.txt` 中的依赖，无需手动操作。如需重建：

```bash
cd third_party/sensevoice/SenseVoice
rm -rf .venv
# 下次 pm2 restart 会自动重建
```

### 启动服务

```bash
cd /path/to/SenseVoice
pm2 start ./start_api.sh --name sensevoice-api
pm2 save
```

`start_api.sh` 默认设置 `SENSEVOICE_DEVICE=mps`，服务监听 `http://localhost:50000`。首次启动会自动创建 `.venv`、安装依赖并下载模型，因此耗时会明显长于后续启动。

不经过 PM2、直接在前台运行：

```bash
cd /path/to/SenseVoice
./start_api.sh
```

需要临时切换到 Accelerate CPU：

```bash
SENSEVOICE_DEVICE=cpu pm2 restart sensevoice-api --update-env
```

恢复 MPS：

```bash
SENSEVOICE_DEVICE=mps pm2 restart sensevoice-api --update-env
```

### 设置开机自启

```bash
# 保存当前进程列表
pm2 save

# 生成启动脚本(需要sudo)
pm2 startup
# 按提示执行输出的sudo命令
```

开机恢复链路为 `launchd -> pm2 resurrect -> start_api.sh -> uvicorn`。如果 PM2 守护进程已退出，也可以手动恢复：

```bash
pm2 resurrect
```

### 常用命令

```bash
pm2 list                    # 查看所有服务
pm2 monit                   # 监控面板
pm2 logs                    # 查看日志
pm2 logs sensevoice-api     # 查看指定服务日志
pm2 restart sensevoice-api  # 重启服务
pm2 stop sensevoice-api     # 停止服务
pm2 delete sensevoice-api   # 删除服务
pm2 save                    # 保存进程列表(增删服务后执行)
```

### PM2配置说明

- **进程列表保存位置**: `~/.pm2/dump.pm2`
- **日志位置**: `~/.pm2/logs/`
- **开机启动原理**:
  1. `pm2 startup` 创建launchd配置
  2. 开机时launchd启动pm2
  3. pm2执行`resurrect`从dump.pm2恢复服务

---

## API 使用

服务启动后可访问 `http://localhost:50000/docs` 查看交互式接口文档。

### 普通转写

```text
POST http://localhost:50000/api/v1/asr
```

请求参数：

- `files`: 一个或多个音频/视频文件。常用格式包括 WAV、MP3、M4A、MP4、MOV、MKV 和 WebM
- `keys`: 可选，多个文件对应的名称，以逗号分隔
- `lang`: `auto`、`zh`、`en`、`ja`、`ko`、`yue` 或 `nospeech`

```bash
curl "http://localhost:50000/api/v1/asr" \
  -F "files=@your_audio.m4a" \
  -F "lang=auto"
```

`result` 数组中的主要字段：

- `raw_text`: 原始识别文本，包含模型标签
- `clean_text`: 移除标签后的纯文本
- `text`: 经过后处理的文本，包含情绪、事件等 emoji 标注

### 带逐句时间戳的转写

```text
POST http://localhost:50000/api/v1/asr-with-timestamps
```

JSON 格式：

```bash
curl "http://localhost:50000/api/v1/asr-with-timestamps" \
  -F "files=@your_audio.m4a" \
  -F "lang=auto" \
  -F "response_format=json"
```

除普通转写字段外，每个结果还包含：

- `segments`: 逐句结果，每项包含 `index`、`start`、`end` 和 `text`
- `srt`: 同一结果生成的完整 SRT 文本

直接返回 SRT：

```bash
curl "http://localhost:50000/api/v1/asr-with-timestamps" \
  -F "files=@your_audio.m4a" \
  -F "lang=auto" \
  -F "response_format=srt" \
  --output subtitles.srt
```

`response_format=srt` 一次只支持一个文件；JSON 格式支持多个文件。

---

## 性能参考

不同音频内容、分段数量和首次编译开销都会影响结果，以下数据用于判断部署方式，不代表固定基准。

普通转写的迁移实测：

| 音频时长 | Accelerate CPU | MPS 首轮 | MPS 热态 |
|---|---:|---:|---:|
| 63 分 45 秒 | 约 59 秒 | 约 17.2 秒 | 约 8.85 秒 |

逐句时间戳接口实测：

| 音频时长 | Accelerate CPU 热态 | MPS 热态 |
|---|---:|---:|
| 10 秒 | 约 0.22 秒 | 约 0.18 秒 |
| 60 秒 | 约 1.18 秒 | 约 0.91 秒 |

MPS 在服务重启后的第一次推理会进行初始化和编译，短音频的首次请求可能比 CPU 慢；长音频或连续请求时，MPS 优势更明显。

历史 Docker 对比中，5 分钟音频在 macOS 原生 Accelerate CPU 上约需 3–4 秒，在 Docker/OpenBLAS 中约需 13–16 秒。由于 Docker 无法使用 Apple Accelerate 和 MPS，Mac 上应优先采用原生部署。
