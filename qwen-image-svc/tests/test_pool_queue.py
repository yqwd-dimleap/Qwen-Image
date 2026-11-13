"""WorkerPool 入队背压：GPU 忙时应排队，队列满才 429。"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from errors import QueueFull, ServiceError
from pool import DEFAULT_GPU_ASSIGNMENT, QUEUE_SLOTS_PER_OP, WorkerPool
from protocol import Task


def test_default_gpu_assignment_is_z_image_turbo_only():
    assert DEFAULT_GPU_ASSIGNMENT == {
        "t2i": [],
        "edit": [],
        "z_image": [],
        "z_image_turbo": [0, 1, 2, 3, 4, 5, 6, 7],
    }


def test_pool_drops_ops_with_no_gpus(monkeypatch):
    monkeypatch.setattr("pool.get_context", lambda *a, **k: _Ctx())
    pool = WorkerPool(
        {
            "t2i": [],
            "edit": [],
            "z_image": [],
            "z_image_turbo": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    try:
        assert list(pool.assignment) == ["z_image_turbo"]
        assert pool.assignment["z_image_turbo"] == list(range(8))
        with pytest.raises(ServiceError, match="no worker for op"):
            pool.submit(_task())
    finally:
        pool.close()


class _MemQueue:
    def __init__(self, maxsize: int = 0):
        self.maxsize = maxsize
        self._items: list = []
        self._lock = threading.Lock()

    def put(self, item, timeout=None):
        with self._lock:
            if self.maxsize and len(self._items) >= self.maxsize:
                raise TimeoutError("queue full")
            self._items.append(item)

    def get(self, timeout=None):
        # drain/supervise 轮询用；保持阻塞片刻再空读
        time.sleep(min(timeout or 0.05, 0.05))
        raise TimeoutError("empty")

    def qsize(self):
        with self._lock:
            return len(self._items)


class _Ctx:
    def Event(self):
        return threading.Event()

    def Queue(self, maxsize=0):
        return _MemQueue(maxsize=maxsize)

    def Process(self, **kwargs):
        proc = MagicMock()
        proc.is_alive.return_value = True
        return proc


def _task() -> Task:
    return Task(
        op="t2i",
        prompt="p",
        negative_prompt=" ",
        width=64,
        height=64,
        seed=1,
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        num_inference_steps=1,
        num_images_per_prompt=1,
    )


@pytest.fixture
def one_gpu_pool(monkeypatch):
    monkeypatch.setattr("pool.get_context", lambda *a, **k: _Ctx())
    pool = WorkerPool({"t2i": [0]})
    yield pool
    pool.close()


def test_second_submit_queues_instead_of_429(one_gpu_pool):
    """一张卡时，第二个请求应进入队列，而不是立刻 saturated。"""
    pool = one_gpu_pool
    assert QUEUE_SLOTS_PER_OP >= 2

    pool.submit(_task())
    pool.submit(_task())  # must queue, not raise


def test_submit_rejects_only_when_queue_depth_exceeded(one_gpu_pool, monkeypatch):
    monkeypatch.setattr("pool.SLOT_WAIT_SECONDS", 0.05)
    pool = one_gpu_pool
    for _ in range(QUEUE_SLOTS_PER_OP):
        pool.submit(_task())

    with pytest.raises(QueueFull):
        pool.submit(_task())


def test_full_queue_accepts_if_a_slot_frees_within_the_wait(one_gpu_pool, monkeypatch):
    monkeypatch.setattr("pool.SLOT_WAIT_SECONDS", 2.0)
    pool = one_gpu_pool
    for _ in range(QUEUE_SLOTS_PER_OP):
        pool.submit(_task())

    def _free_one():
        time.sleep(0.1)
        queue = pool._task_q["t2i"]
        with queue._lock:
            queue._items.pop()
        pool._slots["t2i"].release()

    threading.Thread(target=_free_one).start()
    pool.submit(_task())
