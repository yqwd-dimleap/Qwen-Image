"""CUDA 致命错误应退出 worker，让 supervisor 拉起干净进程。"""
from __future__ import annotations

import pytest

from worker import is_fatal_accelerator_error


@pytest.mark.parametrize(
    "exc",
    [
        Exception(
            "AcceleratorError: CUDA error: unspecified launch failure\n"
            "Search for `cudaErrorLaunchFailure'"
        ),
        type("AcceleratorError", (Exception,), {})(
            "CUDA error: unspecified launch failure"
        ),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        type("OutOfMemoryError", (Exception,), {})("CUDA out of memory"),
    ],
)
def test_cuda_like_errors_are_fatal(exc):
    assert is_fatal_accelerator_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("edit task requires at least one input image"),
        RuntimeError("prompt too long"),
        Exception("disk full"),
    ],
)
def test_ordinary_errors_are_not_fatal(exc):
    assert is_fatal_accelerator_error(exc) is False
