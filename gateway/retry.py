"""上游重试策略：只对 retryable 错误做有限次指数退避重试。

设计要点：
- 重试决策（should_retry）与退避计算（backoff_delay_s）是纯函数，
  与传输层（httpx）/适配器解耦，便于单测
- 上游 Retry-After 头优先于指数退避（429/503 场景遵循上游指示），
  解析失败（如 HTTP-date 格式）时回退指数退避
- 流式重试安全约束由 Gateway.stream 保证：仅当尚未向下游发出任何事件
  （首事件前）才允许重试；流中途失败无法回退已发送内容，直接透传 error 事件

环境变量（进程入口设置一次，from_env 读取）：
  RETRY_MAX_ATTEMPTS    总尝试次数上限（含首次），默认 3；设为 1 = 禁用重试
  RETRY_BACKOFF_BASE_S  指数退避基数（秒），默认 0.5，第 n 次重试延迟 = base * 2^n
  RETRY_BACKOFF_MAX_S   单次退避上限（秒），默认 8.0
"""

import os
import time
from dataclasses import dataclass

from gateway.errors import GatewayError


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """重试策略参数（不可变，进程内共享）。"""

    max_attempts: int = 3        # 总尝试次数上限（含首次）
    backoff_base_s: float = 0.5  # 指数退避基数
    backoff_max_s: float = 8.0   # 单次退避上限

    @staticmethod
    def from_env() -> "RetryPolicy":
        return RetryPolicy(
            max_attempts=max(1, int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))),
            backoff_base_s=max(0.0, float(os.getenv("RETRY_BACKOFF_BASE_S", "0.5"))),
            backoff_max_s=max(0.0, float(os.getenv("RETRY_BACKOFF_MAX_S", "8.0"))),
        )


def should_retry(err: GatewayError, attempt: int, policy: RetryPolicy) -> bool:
    """是否应该发起第 attempt+2 次尝试（attempt 从 0 计数，表示刚失败第 attempt+1 次）。

    条件：错误标记为可重试，且尚未达到总尝试次数上限。
    """
    return bool(err.retryable) and (attempt + 1) < policy.max_attempts


def _parse_retry_after(raw: str | None) -> float | None:
    """解析 Retry-After 头（秒数）；HTTP-date 等不可解析格式返回 None。"""
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def backoff_delay_s(err: GatewayError, attempt: int, policy: RetryPolicy) -> float:
    """计算第 attempt+1 次失败后的等待秒数。

    优先用上游 Retry-After（截断到 backoff_max_s），否则指数退避 base * 2^attempt。
    """
    retry_after = _parse_retry_after(err.retry_after)
    if retry_after is not None:
        return min(retry_after, policy.backoff_max_s)
    return min(policy.backoff_base_s * (2 ** attempt), policy.backoff_max_s)


def sleep_backoff(delay_s: float) -> None:
    """退避等待（独立函数便于测试时 monkeypatch）。"""
    if delay_s > 0:
        time.sleep(delay_s)
