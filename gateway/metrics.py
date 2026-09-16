"""进程内指标注册表：请求/错误/限流/重试计数 + 延迟样本，供 /v1/metrics 暴露。

设计要点：
- 零依赖（threading + deque），线程安全（uvicorn 线程池 + gateway 层都会写）
- 双层延迟口径：
    http_latency_ms  = 中间件统计的所有业务请求往返（流式请求为响应头就绪耗时）
    llm_latency_ms   = Gateway 层统计的上游 LLM 往返（流式含完整消费时长）
  另有 llm_ttft_ms 流式首 token 延迟样本
- 统计范围：/v1/* 业务请求（/v1/metrics 自身与 /health 探活不计入，避免自噪声）
- snapshot() 返回 JSON 可序列化 dict，p50/p95/p99 就地计算（样本封顶 10000）

对应 OTel/Prometheus 语义：
  requests_total / errors_total / rate_limited_total / retries_total
  llm_calls_total / llm_errors_total / http_latency_ms / llm_latency_ms / llm_ttft_ms
"""

import threading
import time
from collections import deque
from typing import Any

_LOCK = threading.Lock()
_STARTED_AT = time.time()
_MAX_SAMPLES = 10000

_counters: dict[str, int] = {
    "requests_total": 0,      # 业务请求总数（含被限流/失败的）
    "errors_total": 0,        # HTTP >= 4xx 的请求数
    "rate_limited_total": 0,  # 被限流中间件拒绝（429）的请求数
    "retries_total": 0,       # Gateway 对上游发起的重试次数
    "llm_calls_total": 0,     # 上游 LLM 调用成功次数
    "llm_errors_total": 0,    # 上游 LLM 调用失败次数（重试耗尽后）
}
_status_counts: dict[str, int] = {}          # "200" -> n
_http_latencies_ms: deque[float] = deque(maxlen=_MAX_SAMPLES)
_llm_latencies_ms: deque[float] = deque(maxlen=_MAX_SAMPLES)
_llm_ttft_ms: deque[float] = deque(maxlen=_MAX_SAMPLES)


def inc(name: str, n: int = 1) -> None:
    """计数器自增；未注册的计数器名静默忽略（防拼写错误扩散）。"""
    if name not in _counters:
        return
    with _LOCK:
        _counters[name] += n


def inc_status(status: int) -> None:
    """按 HTTP 状态码归类计数。"""
    with _LOCK:
        key = str(status)
        _status_counts[key] = _status_counts.get(key, 0) + 1


def observe_http_latency(ms: float) -> None:
    with _LOCK:
        _http_latencies_ms.append(float(ms))


def observe_llm_latency(ms: float) -> None:
    with _LOCK:
        _llm_latencies_ms.append(float(ms))


def observe_llm_ttft(ms: float) -> None:
    with _LOCK:
        _llm_ttft_ms.append(float(ms))


def _stats(samples: deque) -> dict[str, Any] | None:
    """样本统计摘要；空序列返回 None（JSON 里省略）。"""
    if not samples:
        return None
    ordered = sorted(samples)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return round(ordered[idx], 1)

    return {
        "count": n,
        "avg_ms": round(sum(ordered) / n, 1),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": round(ordered[-1], 1),
        "min_ms": round(ordered[0], 1),
    }


def snapshot() -> dict[str, Any]:
    """指标快照（JSON 可序列化），供 /v1/metrics 端点与监控脚本消费。"""
    with _LOCK:
        counters = dict(_counters)
        status_counts = dict(_status_counts)
        http_lat = list(_http_latencies_ms)
        llm_lat = list(_llm_latencies_ms)
        llm_ttft = list(_llm_ttft_ms)

    errors = counters["errors_total"]
    requests = counters["requests_total"]
    llm_total = counters["llm_calls_total"] + counters["llm_errors_total"]

    snap: dict[str, Any] = {
        "uptime_s": round(time.time() - _STARTED_AT, 1),
        "counters": counters,
        "status_counts": status_counts,
        "error_rate": round(errors / requests, 4) if requests else 0.0,
        "llm_success_rate": round(counters["llm_calls_total"] / llm_total, 4) if llm_total else None,
        "http_latency": _stats(http_lat),
        "llm_latency": _stats(llm_lat),
        "llm_ttft": _stats(llm_ttft),
    }
    return snap


def reset() -> None:
    """清零全部指标（仅测试用）。"""
    global _STARTED_AT
    with _LOCK:
        _STARTED_AT = time.time()
        for k in _counters:
            _counters[k] = 0
        _status_counts.clear()
        _http_latencies_ms.clear()
        _llm_latencies_ms.clear()
        _llm_ttft_ms.clear()
