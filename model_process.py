import multiprocessing
import traceback


class ModelProcessClient:
    """Run the heavy ASR model in a disposable child process."""

    def __init__(self, model_config):
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._process = context.Process(
            target=_model_worker,
            args=(child_connection, model_config),
            name="sensevoice-model",
            daemon=True,
        )
        self._process.start()
        child_connection.close()

    def _request(self, operation, payload):
        if not self._process.is_alive():
            raise RuntimeError("SenseVoice model process is not running")
        self._connection.send((operation, payload))
        succeeded, result = self._connection.recv()
        if not succeeded:
            raise RuntimeError(f"SenseVoice model process failed:\n{result}")
        return result

    def generate(self, **kwargs):
        return self._request("generate", kwargs)

    def transcribe_segments(self, input_wav, language, key, target_fs, merge_length_s):
        return self._request(
            "transcribe_segments",
            {
                "input_wav": input_wav,
                "language": language,
                "key": key,
                "target_fs": target_fs,
                "merge_length_s": merge_length_s,
            },
        )

    def close(self):
        if self._process.is_alive():
            try:
                self._connection.send(("shutdown", None))
                self._connection.recv()
            except (BrokenPipeError, EOFError):
                pass
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()


def _load_model(model_config):
    from funasr import AutoModel

    return AutoModel(**model_config)


def _transcribe_segments(model, payload):
    from funasr.utils.vad_utils import merge_vad

    input_wav = payload["input_wav"]
    language = payload["language"]
    key = payload["key"]
    target_fs = payload["target_fs"]

    if model.vad_model is None:
        vad_segments = [[0, int(len(input_wav) / target_fs * 1000)]]
    else:
        model._reset_runtime_configs()
        vad_result = model.inference(
            input_wav,
            model=model.vad_model,
            kwargs=model.vad_kwargs,
            batch_size=1,
        )
        segments = vad_result[0].get("value", []) if vad_result else []
        vad_segments = merge_vad(segments, payload["merge_length_s"] * 1000)

    results = []
    for index, (start_ms, end_ms) in enumerate(vad_segments):
        start_sample = max(int(start_ms * target_fs / 1000), 0)
        end_sample = min(int(end_ms * target_fs / 1000), len(input_wav))
        if end_sample <= start_sample:
            continue

        model._reset_runtime_configs()
        inference_result = model.inference(
            input_wav[start_sample:end_sample],
            model=model.model,
            kwargs=model.kwargs,
            key=f"{key}_seg_{index}",
            language=language,
            use_itn=True,
            batch_size=1,
            output_timestamp=True,
        )
        segment = inference_result[0] if inference_result else {}
        results.append(
            {
                "start_ms": start_ms,
                "text": segment.get("text", ""),
                "timestamp": segment.get("timestamp", []),
            }
        )

    model._reset_runtime_configs()
    return results


def _model_worker(connection, model_config):
    try:
        model = _load_model(model_config)
        while True:
            operation, payload = connection.recv()
            if operation == "shutdown":
                connection.send((True, None))
                return
            try:
                if operation == "generate":
                    result = model.generate(**payload)
                elif operation == "transcribe_segments":
                    result = _transcribe_segments(model, payload)
                else:
                    raise ValueError(f"unsupported model operation: {operation}")
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    finally:
        connection.close()
