# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_qps_control.py
import asyncio
import time

from pytest import approx

from dl_utils import QPSLimiter


def test_qps_limiter():
    max_qps = 10
    max_iter = 10

    def fn():
        # print(f"Current: {time.time()}")
        return 123

    async def main():
        qps_limiter = QPSLimiter(max_qps=max_qps)

        s_time = time.time()
        for _ in range(max_iter):
            res = await qps_limiter.run(fn)
            assert res == 123
        duration = time.time() - s_time

        print(duration)
        assert duration == approx(max_iter / max_qps, 0.1)

    asyncio.run(main())


def test_qps_limiter_performance():
    max_qps = 1000
    sleep_time = 0.1
    test_query = max_qps * 5

    def fn():
        time.sleep(sleep_time)
        print(".", end="")
        return 123

    async def main():
        qps_limiter = QPSLimiter(max_qps=max_qps)

        async def worker():
            res = await qps_limiter.run(fn)
            assert res == 123

        # schedule 100 runs, and wait then to finish
        tasks = [asyncio.create_task(worker()) for _ in range(test_query)]
        await asyncio.gather(*tasks)

        real_qps = qps_limiter.real_qps()

        assert real_qps < max_qps
        assert real_qps == approx(max_qps, rel=0.2)
        assert qps_limiter.query_count == test_query

        print(f"real qps: {real_qps:.2f} qps, "
              f"expected_qps: {max_qps:.2f} qps")

    asyncio.run(main())


def test_qps_limiter_peak_performance():
    test_query = 1000
    sleep_time = 0.1

    def fn():
        time.sleep(sleep_time)
        return 123

    async def main():
        qps_limiter = QPSLimiter(max_qps=9999)

        # Schedule runs and wait for them to finish
        s_time = time.time()
        results = [qps_limiter.run(fn) for _ in range(test_query)]
        await asyncio.gather(*results)

        e_time = time.time()

        print(f"Took {e_time - s_time:.2f} seconds for {test_query} queries")
        print(f"Performance: {len(results) / (e_time - s_time) :.2f} qps")

    asyncio.run(main())


def test_qps_limiter_long_run_task_concurrency():
    max_qps = 10

    def fn():
        time.sleep(5)
        return 123

    async def main():
        qps_limiter = QPSLimiter(max_qps=max_qps)

        s_time = time.time()

        # Append `max_qps` tasks to the queue
        tasks = [qps_limiter.run(fn) for _ in range(max_qps)]

        # Wait for all the task to finish
        results = await asyncio.gather(*tasks)
        duration = time.time() - s_time

        for res in results:
            assert res == 123

        assert duration == approx(6, 0.1)

    asyncio.run(main())
