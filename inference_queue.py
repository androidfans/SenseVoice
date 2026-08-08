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
        loop = asyncio.get_running_loop()
        started = loop.create_future()

        def mark_started():
            if not started.done():
                started.set_result(None)

        def run():
            loop.call_soon_threadsafe(mark_started)
            return function()

        future = self._executor.submit(run)
        try:
            await asyncio.wait_for(
                started,
                self.wait_timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            future.cancel()
            raise InferenceQueueTimeout(
                "SenseVoice inference queue timed out"
            ) from error
        except asyncio.CancelledError:
            future.cancel()
            raise
        return await asyncio.wrap_future(future)
