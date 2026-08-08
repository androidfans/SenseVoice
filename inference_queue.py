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

    def _submit(self, function):
        started = threading.Event()

        def run():
            started.set()
            return function()

        return started, self._executor.submit(run)

    def run_sync(self, function):
        started, future = self._submit(function)
        if not started.wait(self.wait_timeout_seconds):
            future.cancel()
            raise InferenceQueueTimeout("SenseVoice inference queue timed out")
        return future.result()

    async def run_async(self, function):
        started, future = self._submit(function)
        try:
            did_start = await asyncio.to_thread(
                started.wait,
                self.wait_timeout_seconds,
            )
        except asyncio.CancelledError:
            future.cancel()
            raise
        if not did_start:
            future.cancel()
            raise InferenceQueueTimeout("SenseVoice inference queue timed out")
        return await asyncio.wrap_future(future)
