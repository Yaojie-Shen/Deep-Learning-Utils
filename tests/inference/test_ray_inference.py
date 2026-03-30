# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : test_ray_inference.py

import time
import pytest

ray = pytest.importorskip("ray")

from dl_utils.inference.ray_inference_utils import RayActorScheduler


@ray.remote
class Worker:
    def __init__(self, sleep_time: float):
        self.sleep_time = sleep_time

    def process(self, x):
        time.sleep(self.sleep_time)
        return x

    def ping(self):
        """Lightweight readiness check"""
        return True


def test_scheduler_balances_load():
    """Create 10 actors (9x 0.1s, 1x 1.0s), run 91 tasks and ensure elapsed time is ~1s.

    With queue_max_size=1 there are 10 concurrent slots. The 9 fast actors
    will each process ~10 tasks in ~1s (0.1s per task), and the slow actor will
    process 1 task in ~1s, so 91 tasks should finish close to 1 second.
    """
    # Start Ray
    ray.init(ignore_reinit_error=True)

    try:
        # Create 9 fast actors and 1 slow actor
        actors = [Worker.remote(0.1) for _ in range(9)] + [Worker.remote(1.0)]

        # Wait for all actors to be ready via a lightweight ping
        ray.get([a.ping.remote() for a in actors])

        # actor_fn: call the remote process method
        def actor_fn(actor, x):
            return actor.process.remote(x)

        scheduler = RayActorScheduler(actors, actor_fn, queue_max_size=1)

        start = time.time()
        refs = [scheduler.submit(i) for i in range(101)]
        # wait for all tasks to finish
        ray.get(refs)
        elapsed = time.time() - start

        # Shutdown scheduler
        scheduler.shutdown()

        assert 0.9 <= elapsed <= 2.1, f"Elapsed time {elapsed:.3f}s not within expected range"
    finally:
        ray.shutdown()
