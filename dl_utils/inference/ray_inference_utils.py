# -*- coding: utf-8 -*-
# @Time    : 7/18/25
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : ray_inference_utils.py

import logging
import threading
import time
from queue import Queue, Empty
from typing import List, Callable

import ray

logger = logging.getLogger(__name__)


class Scheduler:
    """
    A high-performance load balanced scheduler for Ray actors.
    Balance the load of multiple Ray actors using token bucket.
    """

    def __init__(
            self,
            actors: List[ray.actor.ActorHandle],
            actor_fn: Callable[[ray.actor.ActorHandle, ...], ray.ObjectRef],
            queue_max_size: int = 4
    ):
        self.actors = actors
        self.actor_fn = actor_fn
        self.queue_max_size = queue_max_size
        self.actor_queue = Queue()
        self.pending_ref_queue = Queue()  # Submit task to the monitor thread
        self.active_refs = {}  # Use to store the active tasks in monitor thread
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Initialize token bucket
        for actor in actors:
            for _ in range(queue_max_size):
                self.actor_queue.put(actor)

        # Monitor task status in background to add token back to the bucket
        self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self._monitor_thread.start()

    def submit(self, *args, **kwargs):
        s_time = time.time()
        actor = self.actor_queue.get()  # wait for available actor
        obj_ref = self.actor_fn(actor, *args, **kwargs)
        assert isinstance(obj_ref, ray.ObjectRef), \
            f"The actor_fn should return a ray.ObjectRef, but got {type(obj_ref)}"
        self.pending_ref_queue.put((obj_ref, actor))
        logger.debug(f"Schedule task took {time.time() - s_time:.6f}s")
        return obj_ref

    def _background_monitor(self):
        while not self._stop_event.is_set():
            # move all task from queue into active_refs
            try:
                while True:
                    obj_ref, actor = self.pending_ref_queue.get_nowait()
                    self.active_refs[obj_ref] = actor
            except Empty:
                pass

            if self.active_refs:
                ready_refs, _ = ray.wait(
                    list(self.active_refs.keys()), num_returns=1, timeout=0.001
                )
                for ref in ready_refs:
                    actor = self.active_refs.pop(ref, None)
                    if actor:
                        self.actor_queue.put(actor)

    def shutdown(self):
        self._stop_event.set()
        self._monitor_thread.join()


__all__ = ["Scheduler"]
