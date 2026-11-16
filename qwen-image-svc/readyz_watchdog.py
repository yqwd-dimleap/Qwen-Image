"""进程内定时探测 ready 状态，异常时发飞书。

挂在 FastAPI lifespan 里跑，不依赖 crontab。
告警内容只描述「探测 ready」结果。
Webhook 必须由本服务自己的 FEISHU_WEBHOOK_URL 配置，不内置默认地址。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import suppress
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 1200  # 20min；可用 READYZ_ALERT_INTERVAL_SECONDS 覆盖


def get_feishu_webhook_url() -> str:
    return (os.environ.get("FEISHU_WEBHOOK_URL") or "").strip()


def feishu_notify_enabled() -> bool:
    """默认关闭；需同时设置 FEISHU_NOTIFY_ENABLED=1 和 FEISHU_WEBHOOK_URL。"""
    if not get_feishu_webhook_url():
        return False
    flag = (os.environ.get("FEISHU_NOTIFY_ENABLED") or "0").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def resolve_interval_seconds(raw: str | None = None) -> float:
    """解析探测间隔；非法或 <=0 时回退默认并打 error 日志。"""
    if raw is None:
        raw = (os.environ.get("READYZ_ALERT_INTERVAL_SECONDS") or "").strip()
    else:
        raw = raw.strip()
    if not raw:
        return float(DEFAULT_INTERVAL_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        logger.error(
            "invalid READYZ_ALERT_INTERVAL_SECONDS=%r; using default %s",
            raw,
            DEFAULT_INTERVAL_SECONDS,
        )
        return float(DEFAULT_INTERVAL_SECONDS)
    if value <= 0:
        logger.error(
            "non-positive READYZ_ALERT_INTERVAL_SECONDS=%r; using default %s",
            raw,
            DEFAULT_INTERVAL_SECONDS,
        )
        return float(DEFAULT_INTERVAL_SECONDS)
    return value


def check_ready(snapshot: dict[str, str]) -> bool:
    if not snapshot:
        return False
    return all(state == "ready" for state in snapshot.values())


def build_alert_text(snapshot: dict[str, str]) -> str:
    lines = [f"{op}={state}" for op, state in snapshot.items()]
    return "【qwen-image-svc】探测 ready 失败\n" + "\n".join(lines)


def send_feishu_text(text: str, *, timeout_s: float = 10.0) -> bool:
    """发送飞书文本。成功返回 True；未配置或失败返回 False（不抛异常）。"""
    url = get_feishu_webhook_url()
    if not url:
        return False
    payload = {"msg_type": "text", "content": {"text": text}}
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        code = body.get("code", body.get("StatusCode"))
        if code not in (0, None):
            logger.warning("feishu readyz alert rejected: %s", body)
            return False
        return True
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        logger.warning("feishu readyz alert failed: %s", exc)
        return False


async def notify_if_not_ready(snapshot: dict[str, str]) -> bool:
    """非 ready 时尝试发飞书。返回是否真正发送成功。"""
    if not feishu_notify_enabled():
        return False
    if check_ready(snapshot):
        return False
    # 同步 HTTP 放到线程，避免卡住事件循环。
    return await asyncio.to_thread(send_feishu_text, build_alert_text(snapshot))


async def cancel_watchdog(task: asyncio.Task) -> None:
    """取消看门狗并等待其退出，避免与 pool.close() 竞态。"""
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


async def run_readyz_watchdog(
    get_snapshot: Callable[[], dict[str, str]],
    *,
    interval_seconds: float | None = None,
) -> None:
    """周期性探测 ready；非 ready 时发飞书。先睡一个周期，避开启动加载误报。"""
    if interval_seconds is None:
        interval_seconds = resolve_interval_seconds()

    if not feishu_notify_enabled():
        logger.info(
            "readyz watchdog disabled (need FEISHU_NOTIFY_ENABLED=1 and FEISHU_WEBHOOK_URL)"
        )
        return

    logger.info(
        "readyz watchdog started interval=%ss webhook_configured=true",
        interval_seconds,
    )
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            snap = get_snapshot()
            delivered = await notify_if_not_ready(snap)
            if delivered:
                logger.warning("readyz not ready, feishu alert sent: %s", snap)
            elif not check_ready(snap) and feishu_notify_enabled():
                logger.warning(
                    "readyz not ready, feishu alert attempted but not delivered: %s",
                    snap,
                )
        except Exception:  # noqa: BLE001 —— 告警失败不能拖垮 API 进程
            logger.exception("readyz watchdog tick failed")
