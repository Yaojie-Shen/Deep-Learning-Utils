# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : qps_control.py

import asyncio
import concurrent.futures
import time
from typing import Optional, Any, Callable


class QPSLimiter:
    """
    This is a QPS Limiter based on token bucket. It is suitable for tasks with QPS below 100.
    """

    def __init__(
            self,
            max_qps: int = 1,
            max_concurrent: Optional[int] = None,
            init_tokens: int = 0,
    ):
        assert max_qps > 0, "max_qps must be greater than 0"

        if max_concurrent is None:
            max_concurrent = max_qps * 100

        self.qps = max_qps

        self.tokens = init_tokens  # Token bucket

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.lock = asyncio.Lock()

        self.running = True
        self.refill_task = asyncio.create_task(self._refill_tokens())  # Refill token in background
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)

        self.start_time = time.time()
        self.end_time = None
        self.query_count = 0

    def real_qps(self) -> float:
        """Return the real QPS based on the query count and time."""
        if self.end_time is None:
            return self.query_count / (time.time() - self.start_time)
        return self.query_count / (self.end_time - self.start_time)

    async def _refill_tokens(self):
        """Refill token"""
        s_time = time.time()
        while self.running:
            # Sleep at the start to ensure wait for enough time when refilling the first token
            await asyncio.sleep(1 / self.qps - (time.time() - s_time))
            # Skip if already have enough token in bucket to reduce computation
            if self.tokens == self.qps:
                continue
            # Refill token
            async with self.lock:
                self.tokens = min(self.qps, self.tokens + 1)
            s_time = time.time()

    async def shutdown(self):
        """Shutdown token refill"""
        self.running = False
        self.refill_task.cancel()
        try:
            await self.refill_task
        except asyncio.CancelledError:
            pass
        self.end_time = time.time()

    async def run(self, func: Callable, *args) -> dict[str, Any]:
        if not self.running:
            raise RuntimeError("QPSLimiter is already shutdown")

        async with self.semaphore:  # Limit max concurrency
            while True:
                async with self.lock:
                    if self.tokens > 0:
                        self.tokens -= 1
                        self.query_count += 1
                        break
                await asyncio.sleep(0.0001)  # Wait for token

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self.executor, func, *args)
            return result


__all__ = ["QPSLimiter"]
