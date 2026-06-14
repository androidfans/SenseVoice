# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re, time, threading, subprocess
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
import torch
import numpy as np
import gradio as gr
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.vad_utils import merge_vad
from io import BytesIO

TARGET_FS = 16000
TIMESTAMP_MERGE_LENGTH_S = 15
SUBTITLE_MAX_DURATION_S = 8.0
SUBTITLE_MAX_CHARS = 42
SUBTITLE_MIN_DURATION_S = 1.0
SUBTITLE_GAP_S = 0.8
FFMPEG_CHUNK_SIZE = 65536

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts", ".m4v"}

# 空闲超时设置 (秒)
IDLE_TIMEOUT = int(os.getenv("SENSEVOICE_IDLE_TIMEOUT", 900))  # 默认15分钟
last_request_time = time.time()


def idle_checker():
    """后台线程: 检测空闲超时,自动退出释放内存"""
    while True:
        time.sleep(60)  # 每分钟检查一次
        idle_time = time.time() - last_request_time
        if idle_time > IDLE_TIMEOUT:
            print(f"空闲超时 ({IDLE_TIMEOUT}秒), 自动退出释放内存...")
            os._exit(0)


# 启动空闲检测线程
threading.Thread(target=idle_checker, daemon=True).start()


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


class TimestampResponseFormat(str, Enum):
    json = "json"
    srt = "srt"


model_dir = "iic/SenseVoiceSmall"
# 使用AutoModel并启用VAD模型,与webui保持一致
model = AutoModel(
    model=model_dir,
    remote_code="./model.py",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
    trust_remote_code=True,
)

regex = r"<\|.*?\|>"

app = FastAPI()


def normalize_language(lang):
    if isinstance(lang, Language):
        lang = lang.value
    return lang or "auto"


def extract_audio_from_path(filepath: str):
    """用 ffmpeg 从文件提取音频，返回 16kHz mono float32 numpy 数组。"""
    proc = subprocess.Popen(
        ["ffmpeg", "-i", filepath, "-vn",
         "-ar", str(TARGET_FS), "-ac", "1", "-f", "s16le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    pcm_bytes, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr.decode(errors='replace')}")
    if not pcm_bytes:
        raise RuntimeError("ffmpeg produced no audio output")
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


async def extract_audio_from_upload(file: UploadFile):
    """将 UploadFile 写入临时文件后用 ffmpeg 提取音频（MP4 moov atom 需要 seekable 输入）。"""
    import asyncio, tempfile
    spooled = file.file

    def _run():
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or "")[1]) as tmp:
            while True:
                chunk = spooled.read(FFMPEG_CHUNK_SIZE)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.flush()
            return extract_audio_from_path(tmp.name)

    return await asyncio.get_event_loop().run_in_executor(None, _run)


def _is_video_file(filename: str) -> bool:
    if not filename:
        return False
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTS


async def load_upload_audio(file: UploadFile):
    if _is_video_file(file.filename):
        return await extract_audio_from_upload(file)

    file_io = BytesIO(await file.read())
    data_or_path_or_list, audio_fs = torchaudio.load(file_io)

    if audio_fs != TARGET_FS:
        resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
        data_or_path_or_list = resampler(data_or_path_or_list)

    if len(data_or_path_or_list.shape) > 1:
        data_or_path_or_list = data_or_path_or_list.mean(0)

    return data_or_path_or_list.numpy().astype(np.float32)


def strip_rich_tags(text: str):
    return re.sub(regex, "", text or "", 0, re.MULTILINE)


def token_to_text(token):
    token = str(token)
    token = token.replace("▁", " ")
    return strip_rich_tags(token)


def normalize_segment_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?，。！？；：、])", r"\1", text)
    text = re.sub(r"([（(【\[])\s+", r"\1", text)
    text = re.sub(r"\s+([）)】\]])", r"\1", text)
    return text


def is_sentence_boundary(text):
    return bool(re.search(r"[。！？!?；;]$", text))


def parse_token_timestamps(token_timestamps, offset_seconds=0.0):
    parsed = []
    for item in token_timestamps or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue

        token, start, end = item[0], item[1], item[2]
        try:
            start = float(start) + offset_seconds
            end = float(end) + offset_seconds
        except (TypeError, ValueError):
            continue

        text = token_to_text(token)
        if not text or end < start:
            continue

        parsed.append({"text": text, "start": start, "end": end})

    return parsed


def format_srt_time(seconds):
    seconds = max(float(seconds), 0.0)
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_subtitle_segments(tokens, fallback_text=""):
    segments = []
    current = []

    def flush():
        if not current:
            return
        text = normalize_segment_text("".join(item["text"] for item in current))
        if not text:
            current.clear()
            return
        segments.append(
            {
                "index": len(segments) + 1,
                "start": round(current[0]["start"], 3),
                "end": round(max(current[-1]["end"], current[0]["start"] + 0.2), 3),
                "text": text,
            }
        )
        current.clear()

    for token in tokens:
        if current and token["start"] - current[-1]["end"] >= SUBTITLE_GAP_S:
            flush()

        current.append(token)
        text = normalize_segment_text("".join(item["text"] for item in current))
        duration = current[-1]["end"] - current[0]["start"]
        enough_content = len(text) >= 8 or duration >= SUBTITLE_MIN_DURATION_S

        if (
            (is_sentence_boundary(text) and enough_content)
            or duration >= SUBTITLE_MAX_DURATION_S
            or len(text) >= SUBTITLE_MAX_CHARS
        ):
            flush()

    flush()

    if not segments and fallback_text:
        text = rich_transcription_postprocess(strip_rich_tags(fallback_text))
        if text:
            segments.append({"index": 1, "start": 0.0, "end": 0.2, "text": text})

    return segments


def build_srt(segments):
    blocks = []
    for segment in segments:
        blocks.append(
            "\n".join(
                [
                    str(segment["index"]),
                    f"{format_srt_time(segment['start'])} --> {format_srt_time(segment['end'])}",
                    segment["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


def get_vad_segments(input_wav):
    if model.vad_model is None:
        duration_ms = int(len(input_wav) / TARGET_FS * 1000)
        return [[0, duration_ms]]

    model._reset_runtime_configs()
    vad_res = model.inference(
        input_wav,
        model=model.vad_model,
        kwargs=model.vad_kwargs,
        batch_size=1,
    )
    if not vad_res:
        return []

    segments = vad_res[0].get("value", [])
    return merge_vad(segments, TIMESTAMP_MERGE_LENGTH_S * 1000)


def transcribe_segment_with_timestamps(input_wav, language, key):
    model._reset_runtime_configs()
    res = model.inference(
        input_wav,
        model=model.model,
        kwargs=model.kwargs,
        key=key,
        language=language,
        use_itn=True,
        batch_size=1,
        output_timestamp=True,
    )
    return res[0] if res else {"text": "", "timestamp": []}


def transcribe_with_subtitles(input_wav, language, key):
    vad_segments = get_vad_segments(input_wav)
    token_timestamps = []
    raw_texts = []

    for idx, (start_ms, end_ms) in enumerate(vad_segments):
        start_sample = max(int(start_ms * TARGET_FS / 1000), 0)
        end_sample = min(int(end_ms * TARGET_FS / 1000), len(input_wav))
        if end_sample <= start_sample:
            continue

        segment_audio = input_wav[start_sample:end_sample]
        segment_res = transcribe_segment_with_timestamps(
            segment_audio,
            language=language,
            key=f"{key}_seg_{idx}",
        )
        raw_text = segment_res.get("text", "")
        if raw_text:
            raw_texts.append(raw_text)
        token_timestamps.extend(
            parse_token_timestamps(
                segment_res.get("timestamp", []),
                offset_seconds=start_ms / 1000.0,
            )
        )

    model._reset_runtime_configs()
    raw_text = " ".join(raw_texts).strip()
    text = rich_transcription_postprocess(raw_text)
    clean_text = strip_rich_tags(raw_text)
    segments = build_subtitle_segments(token_timestamps, fallback_text=raw_text)

    return {
        "key": key,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "text": text,
        "segments": segments,
        "srt": build_srt(segments),
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    global last_request_time
    last_request_time = time.time()  # 更新最后请求时间

    results = []

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    for idx, file in enumerate(files):
        input_wav = await load_upload_audio(file)
        language = normalize_language(lang)

        # 使用AutoModel的generate方法,启用VAD和batch处理
        res = model.generate(
            input=input_wav,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  # 关键:启用VAD分段合并
        )

        if len(res) > 0:
            text = res[0]["text"]
            result_item = {
                "key": key[idx] if idx < len(key) else file.filename,
                "raw_text": text,
                "clean_text": strip_rich_tags(text),
                "text": rich_transcription_postprocess(text)
            }
            results.append(result_item)

    return {"result": results}


@app.post("/api/v1/asr-with-timestamps")
async def turn_audio_to_text_with_timestamps(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
    response_format: Annotated[
        TimestampResponseFormat,
        Form(description="response format: json or srt"),
    ] = TimestampResponseFormat.json,
):
    global last_request_time
    last_request_time = time.time()

    if response_format == TimestampResponseFormat.srt and len(files) != 1:
        raise HTTPException(
            status_code=400,
            detail="response_format=srt only supports one audio file per request",
        )

    results = []
    language = normalize_language(lang)

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    for idx, file in enumerate(files):
        input_wav = await load_upload_audio(file)
        item_key = key[idx] if idx < len(key) else file.filename
        results.append(transcribe_with_subtitles(input_wav, language, item_key))

    if response_format == TimestampResponseFormat.srt:
        return PlainTextResponse(results[0]["srt"], media_type="application/x-subrip")

    return {"result": results}


def _run_asr(input_wav, language):
    """共用的 ASR 推理逻辑。"""
    language = "auto" if not language else language
    res = model.generate(
        input=input_wav,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
    )
    if not res:
        return ""
    return rich_transcription_postprocess(strip_rich_tags(res[0]["text"]))


def webui_inference(input_wav, language):
    global last_request_time
    last_request_time = time.time()

    if isinstance(input_wav, tuple):
        fs, audio_data = input_wav
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(-1)
        if fs != TARGET_FS:
            resampler = torchaudio.transforms.Resample(fs, TARGET_FS)
            audio_data = resampler(torch.from_numpy(audio_data).unsqueeze(0))[0].numpy()
        input_wav = audio_data

    return _run_asr(input_wav, language)


def webui_file_inference(filepath, language):
    global last_request_time
    last_request_time = time.time()

    if not filepath:
        return "请上传文件"
    input_wav = extract_audio_from_path(filepath)
    return _run_asr(input_wav, language)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h2>SenseVoice ASR</h2>"
        "<p>上传音频或使用麦克风录音，选择语言后点击开始识别。</p>"
    )
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="上传音频或使用麦克风")
            language_input = gr.Dropdown(
                choices=["auto", "zh", "en", "yue", "ja", "ko"],
                value="auto",
                label="语言",
            )
            run_button = gr.Button("开始识别", variant="primary")
            text_output = gr.Textbox(label="识别结果", lines=6)
        with gr.Column():
            file_input = gr.File(label="上传视频/大文件", file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".flac", ".m4a"])
            file_language_input = gr.Dropdown(
                choices=["auto", "zh", "en", "yue", "ja", "ko"],
                value="auto",
                label="语言",
            )
            file_run_button = gr.Button("开始识别", variant="primary")
            file_text_output = gr.Textbox(label="识别结果", lines=6)

    run_button.click(webui_inference, inputs=[audio_input, language_input], outputs=text_output)
    file_run_button.click(webui_file_inference, inputs=[file_input, file_language_input], outputs=file_text_output)

app = gr.mount_gradio_app(app, demo, path="/ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50000)
