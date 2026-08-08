import asyncio
import threading
import time
import unittest

from huoshan_compat import build_huoshan_result
from inference_queue import InferenceQueue, InferenceQueueTimeout


class HuoshanCompatibilityTest(unittest.TestCase):
    def test_build_huoshan_result_matches_expected_contract(self):
        result = build_huoshan_result(
            "全世界95%以上。",
            [
                [
                    {"text": "全", "start": 0.03, "end": 0.27},
                    {"text": "世界", "start": 0.27, "end": 0.55},
                    {"text": "95%", "start": 0.55, "end": 1.15},
                    {"text": "以上", "start": 1.15, "end": 1.47},
                    {"text": "。", "start": 1.47, "end": 1.55},
                ]
            ],
        )

        self.assertEqual(result["code"], 1000)
        self.assertEqual(result["message"], "Success")
        self.assertEqual(result["additions"], {})
        self.assertEqual(result["text"], "全世界95%以上。")
        self.assertEqual(
            result["utterances"],
            [
                {
                    "additions": {},
                    "start_time": 30,
                    "end_time": 1550,
                    "text": "全世界95%以上。",
                    "words": [
                        {"text": "全", "start_time": 30, "end_time": 270},
                        {"text": "世界", "start_time": 270, "end_time": 550},
                        {"text": "95%", "start_time": 550, "end_time": 1150},
                        {"text": "以上", "start_time": 1150, "end_time": 1550},
                    ],
                }
            ],
        )

    def test_build_huoshan_result_rejects_unaligned_text(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_huoshan_result(
                "不匹配",
                [[{"text": "别的文本", "start": 0, "end": 1}]],
            )

    def test_empty_audio_is_a_successful_empty_result(self):
        result = build_huoshan_result("", [])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["utterances"], [])

    def test_inference_queue_serializes_async_jobs(self):
        queue = InferenceQueue(wait_timeout_seconds=1)
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        def job():
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1

        async def run_jobs():
            await asyncio.gather(*(queue.run_async(job) for _ in range(3)))

        asyncio.run(run_jobs())
        self.assertEqual(maximum_active, 1)

    def test_inference_queue_times_out_before_a_queued_job_starts(self):
        queue = InferenceQueue(wait_timeout_seconds=0.02)
        release = threading.Event()
        first_started = threading.Event()
        second_ran = threading.Event()

        def first_job():
            first_started.set()
            release.wait(1)

        first_future = queue._executor.submit(first_job)
        self.assertTrue(first_started.wait(1))
        try:
            with self.assertRaises(InferenceQueueTimeout):
                queue.run_sync(second_ran.set)
        finally:
            release.set()
            first_future.result(timeout=1)

        time.sleep(0.03)
        self.assertFalse(second_ran.is_set())


if __name__ == "__main__":
    unittest.main()
