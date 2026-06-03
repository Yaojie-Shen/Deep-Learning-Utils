# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_qps_control.py
import asyncio
import time

from pytest import approx, mark, param

from dl_utils import QPSLimiter


def _fn(sleep_time, return_value):
    time.sleep(sleep_time)
    return return_value


@mark.parametrize(
    "max_qps, test_queries, sleep_time",
    [
        param(0.5, 1, 0.1),
        param(1, 1, 0.1),
        param(1000, 1000, 0.1),
        param(1000, 1000, 3),
    ],
)
def test_qps_limiter(max_qps, test_queries, sleep_time):
    return_value = 123456

    async def main():
        qps_limiter = QPSLimiter(max_qps=max_qps)

        s_time = time.time()
        results = [
            qps_limiter.run(_fn, sleep_time, return_value=return_value)
            for _ in range(test_queries)
        ]
        results = await asyncio.gather(*results)
        real_qps = qps_limiter.real_qps()
        duration = time.time() - s_time
        expected_duration = test_queries / max_qps + sleep_time
        print(
            f"Queries: {test_queries}, Sleep Time: {sleep_time}, "
            f"QPS Limit: {max_qps}, Real QPS: {real_qps}, "
            f"Took: {duration}s, Expected: {expected_duration}s"
        )

        assert all(x == return_value for x in results), (
            f"Return value is not correct: {results}"
        )
        assert duration == approx(expected_duration, 0.1), (
            "Task do not finish in expected duration"
        )

    asyncio.run(main())


def test_qps_limiter_peak_performance():
    test_queries = 1000

    async def main():
        qps_limiter = QPSLimiter(max_qps=9999)

        s_time = time.time()
        results = [
            qps_limiter.run(_fn, 0.1, return_value=123456) for _ in range(test_queries)
        ]
        await asyncio.gather(*results)
        e_time = time.time()

        print(
            f"Took {e_time - s_time:.2f} seconds for {test_queries} queries. "
            f"Performance: {len(results) / (e_time - s_time):.2f} qps"
        )

    asyncio.run(main())
