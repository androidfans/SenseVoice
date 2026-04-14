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

**结论**: Mac上CPU密集型任务不适合用Docker,建议本机直接运行。

### 2. CUDA报错

**现象**: 启动时报错 `AssertionError: Torch not compiled with CUDA enabled`

**原因**: 默认device是`cuda:0`,但Mac没有CUDA

**解决**: 设置环境变量
```bash
export SENSEVOICE_DEVICE=cpu
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
- PM2 (`npm install -g pm2`)

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
```

### 设置开机自启
```bash
# 保存当前进程列表
pm2 save

# 生成启动脚本(需要sudo)
pm2 startup
# 按提示执行输出的sudo命令
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

## API使用

### 接口地址
```
POST http://localhost:50000/api/v1/asr
```

### 请求参数
- `files`: 音频文件 (支持mp3, wav等)
- `lang`: 语言 (`auto`/`zh`/`en`/`ja`/`ko`/`yue`)

### curl示例
```bash
curl -X POST "http://localhost:50000/api/v1/asr" \
  -F "files=@your_audio.mp3" \
  -F "lang=auto"
```

### 返回字段
- `raw_text`: 原始识别文本(含标签)
- `clean_text`: 清理后的纯文本
- `text`: 富文本(含emoji标注)

---

## 性能参考

| 测试环境 | 5分钟音频识别耗时 |
|---------|-----------------|
| Mac本机 (Apple Silicon) | **3-4秒** |
| Docker容器 | 13-16秒 |

本机运行比Docker快**3-4倍**,强烈建议Mac上直接本机部署。
