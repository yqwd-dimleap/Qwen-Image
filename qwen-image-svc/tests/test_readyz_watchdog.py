"""进程内 readyz 定时告警（飞书）。"""
from __future__ import annotations

import asyncio
import json
from urllib.error import URLError

import pytest

from readyz_watchdog import (
    DEFAULT_INTERVAL_SECONDS,
    build_alert_text,
    cancel_watchdog,
    check_ready,
    feishu_notify_enabled,
    get_feishu_webhook_url,
    notify_if_not_ready,
    resolve_interval_seconds,
    run_readyz_watchdog,
    send_feishu_text,
)


def test_default_interval_is_20_minutes():
    assert DEFAULT_INTERVAL_SECONDS == 1200


def test_webhook_comes_only_from_env(monkeypatch):
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    assert get_feishu_webhook_url() == ""

    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/qwen-own",
    )
    assert get_feishu_webhook_url().endswith("/hook/qwen-own")


def test_interval_env_override(monkeypatch, caplog):
    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "1")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/super-secret-token",
    )
    monkeypatch.setenv("READYZ_ALERT_INTERVAL_SECONDS", "30")
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        raise asyncio.CancelledError()

    monkeypatch.setattr("readyz_watchdog.asyncio.sleep", fake_sleep)

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await run_readyz_watchdog(lambda: {"z_image_turbo": "ready"})

    with caplog.at_level("INFO", logger="readyz_watchdog"):
        asyncio.run(scenario())
    assert slept == [30.0]
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "webhook_configured=true" in joined
    assert "super-secret-token" not in joined
    assert "/hook/" not in joined


def test_feishu_notify_enabled_requires_flag_and_webhook(monkeypatch):
    monkeypatch.delenv("FEISHU_NOTIFY_ENABLED", raising=False)
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    assert feishu_notify_enabled() is False

    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "1")
    assert feishu_notify_enabled() is False

    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    assert feishu_notify_enabled() is True

    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "0")
    assert feishu_notify_enabled() is False


def test_check_ready_true_when_all_ready():
    assert check_ready({"z_image_turbo": "ready"}) is True


def test_check_ready_false_when_loading_or_empty():
    assert check_ready({"z_image_turbo": "loading"}) is False
    assert check_ready({}) is False
    assert check_ready({"z_image_turbo": "ready", "t2i": "init_failed: boom"}) is False


def test_build_alert_text_says_ready_probe_failed():
    text = build_alert_text({"z_image_turbo": "loading"})
    assert "探测 ready 失败" in text
    assert "z_image_turbo=loading" in text
    assert "qwen-image-svc" in text


def test_resolve_interval_falls_back_on_invalid(monkeypatch, caplog):
    monkeypatch.delenv("READYZ_ALERT_INTERVAL_SECONDS", raising=False)
    assert resolve_interval_seconds() == float(DEFAULT_INTERVAL_SECONDS)

    with caplog.at_level("ERROR", logger="readyz_watchdog"):
        assert resolve_interval_seconds("abc") == float(DEFAULT_INTERVAL_SECONDS)
        assert resolve_interval_seconds("0") == float(DEFAULT_INTERVAL_SECONDS)
        assert resolve_interval_seconds("-5") == float(DEFAULT_INTERVAL_SECONDS)
    assert resolve_interval_seconds("45") == 45.0


def test_notify_if_not_ready_skips_when_ready(monkeypatch):
    calls = []
    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "1")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    monkeypatch.setattr(
        "readyz_watchdog.send_feishu_text",
        lambda text: calls.append(text),
    )
    assert asyncio.run(notify_if_not_ready({"z_image_turbo": "ready"})) is False
    assert calls == []


def test_notify_if_not_ready_sends_when_not_ready(monkeypatch):
    calls = []
    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "1")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )

    def fake_send(text: str) -> bool:
        calls.append(text)
        return True

    monkeypatch.setattr("readyz_watchdog.send_feishu_text", fake_send)
    assert asyncio.run(notify_if_not_ready({"z_image_turbo": "loading"})) is True
    assert len(calls) == 1
    assert "z_image_turbo=loading" in calls[0]


def test_notify_if_not_ready_respects_disabled_flag(monkeypatch):
    calls = []
    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "0")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    monkeypatch.setattr(
        "readyz_watchdog.send_feishu_text",
        lambda text: calls.append(text),
    )
    assert asyncio.run(notify_if_not_ready({"z_image_turbo": "loading"})) is False
    assert calls == []


def test_watchdog_continues_after_snapshot_error(monkeypatch):
    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "1")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    monkeypatch.setenv("READYZ_ALERT_INTERVAL_SECONDS", "1")
    ticks = {"n": 0}
    slept = []

    def get_snap():
        ticks["n"] += 1
        if ticks["n"] == 1:
            raise RuntimeError("boom")
        return {"z_image_turbo": "ready"}

    async def fake_sleep(seconds):
        slept.append(seconds)
        # 两次 tick 之后再取消：sleep1→tick boom，sleep2→tick ok，sleep3→cancel
        if len(slept) >= 3:
            raise asyncio.CancelledError()

    monkeypatch.setattr("readyz_watchdog.asyncio.sleep", fake_sleep)

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await run_readyz_watchdog(get_snap)

    asyncio.run(scenario())
    assert ticks["n"] >= 2
    assert len(slept) >= 3


def test_cancel_watchdog_awaits_task():
    async def scenario() -> None:
        started = asyncio.Event()

        async def forever():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(forever())
        await started.wait()
        await cancel_watchdog(task)
        assert task.done()

    asyncio.run(scenario())


def test_send_feishu_text_posts_text_payload(monkeypatch):
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"code": 0}).encode()

    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _Resp()

    monkeypatch.setattr("readyz_watchdog.urlopen", fake_urlopen)
    assert (
        send_feishu_text("【qwen-image-svc】探测 ready 失败\nz_image_turbo=loading")
        is True
    )
    assert captured["url"].endswith("/hook/test")
    assert captured["body"]["msg_type"] == "text"
    assert "探测 ready 失败" in captured["body"]["content"]["text"]


def test_send_feishu_text_swallows_network_errors(monkeypatch):
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    monkeypatch.setattr(
        "readyz_watchdog.urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(URLError("down")),
    )
    assert send_feishu_text("hello") is False


def test_send_feishu_text_noops_without_webhook(monkeypatch):
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    calls = []
    monkeypatch.setattr(
        "readyz_watchdog.urlopen",
        lambda *_a, **_k: calls.append(1),
    )
    assert send_feishu_text("hello") is False
    assert calls == []


def test_watchdog_logs_attempted_when_send_fails(monkeypatch, caplog):
    monkeypatch.setenv("FEISHU_NOTIFY_ENABLED", "1")
    monkeypatch.setenv(
        "FEISHU_WEBHOOK_URL",
        "https://open.feishu.cn/open-apis/bot/v2/hook/test",
    )
    monkeypatch.setenv("READYZ_ALERT_INTERVAL_SECONDS", "1")
    monkeypatch.setattr("readyz_watchdog.send_feishu_text", lambda _text: False)

    async def fake_sleep(seconds):
        raise asyncio.CancelledError()

    # 先睡完一个周期再探测：改成第一次 sleep 后继续、第二次取消太绕；
    # 直接让 sleep 返回一次后取消，并在取消前跑完 tick——用计数控制。
    slept = {"n": 0}

    async def sleep_once_then_cancel(seconds):
        slept["n"] += 1
        if slept["n"] >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr("readyz_watchdog.asyncio.sleep", sleep_once_then_cancel)

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await run_readyz_watchdog(lambda: {"z_image_turbo": "loading"})

    with caplog.at_level("WARNING", logger="readyz_watchdog"):
        asyncio.run(scenario())
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "attempted but not delivered" in joined
    assert "feishu alert sent" not in joined
