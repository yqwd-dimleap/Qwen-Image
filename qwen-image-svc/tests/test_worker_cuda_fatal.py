"""CUDA 致命错误应退出 worker，让 supervisor 拉起干净进程。"""
from __future__ import annotations

import pytest

from protocol import Result, Task
from worker import FatalAcceleratorError, ModelHost, is_fatal_accelerator_error


def make_task(op: str = "z_image_turbo") -> Task:
    return Task(
        op=op,
        prompt="家庭教师",
        negative_prompt=" ",
        width=1584,
        height=1056,
        seed=1,
        true_cfg_scale=0.0,
        guidance_scale=0.0,
        num_inference_steps=8,
        num_images_per_prompt=1,
    )


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


def test_run_raises_fatal_accelerator_error_for_cuda_failure(monkeypatch):
    monkeypatch.setenv("SVC_FAKE", "1")
    host = ModelHost("z_image_turbo", gpu_id=0)

    def boom(_task):
        raise RuntimeError("CUDA error: unspecified launch failure")

    monkeypatch.setattr(host, "_fake_generate", boom)

    with pytest.raises(FatalAcceleratorError) as caught:
        host.run(make_task())

    result = caught.value.result
    assert result.ok is False
    assert "CUDA" in (result.error or "")


def test_run_returns_result_for_ordinary_failure(monkeypatch):
    monkeypatch.setenv("SVC_FAKE", "1")
    host = ModelHost("z_image_turbo", gpu_id=0)

    def boom(_task):
        raise ValueError("bad prompt")

    monkeypatch.setattr(host, "_fake_generate", boom)

    result = host.run(make_task())
    assert isinstance(result, Result)
    assert result.ok is False
    assert "bad prompt" in (result.error or "")
