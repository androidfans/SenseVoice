import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor


class InferenceQueueTimeout(TimeoutError):
    pass


class InferenceQueue:
    """Serialize access to a shared model while keeping async handlers responsive."""

    def __init__(self, wait_timeout_seconds=None):
        self.wait_timeout_seconds = float(
            wait_timeout_seconds
            if wait_timeout_seconds is not None
            else os.getenv("SENSEVOICE_QUEUE_TIMEOUT", "300")
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sensevoice-inference",
        )

    def _submit(self, function, on_started=None):
        started = threading.Event()
        admission_decided = threading.Event()
        rejected = threading.Event()

        def run():
            started.set()
            if on_started is not None:
                on_started()
            admission_decided.wait()
            if rejected.is_set():
                return None
            return function()

        future = self._executor.submit(run)
        return started, admission_decided, rejected, future

    def run_sync(self, function):
        started, admission_decided, rejected, future = self._submit(function)
        if not started.wait(self.wait_timeout_seconds):
            rejected.set()
            admission_decided.set()
            future.cancel()
            raise InferenceQueueTimeout("SenseVoice inference queue timed out")
        admission_decided.set()
        return future.result()

    async def run_async(self, function):
        loop = asyncio.get_running_loop()
        started = loop.create_future()

        def mark_started():
            if not started.done():
                started.set_result(None)

        _started_event, admission_decided, rejected, future = self._submit(
            function,
            lambda: loop.call_soon_threadsafe(mark_started),
        )
        try:
            await asyncio.wait_for(
                started,
                self.wait_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            rejected.set()
            admission_decided.set()
            future.cancel()
            raise InferenceQueueTimeout(
                "SenseVoice inference queue timed out"
            ) from error
        except asyncio.CancelledError:
            rejected.set()
            admission_decided.set()
            future.cancel()
            raise
        admission_decided.set()
        return await asyncio.wrap_future(future)
