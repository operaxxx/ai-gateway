"""限流器模块：滑动窗口 + 令牌桶双算法，统一 Limiter 协议。

多级限流体系：
- 第一级 IP 限流（SlidingWindowLimiter）：server 中间件层，per 客户端 IP
  滑动窗口，精确控制窗口边界，无边界突刺——适合防单客户端刷量
- 第二级模型限流（TokenBucketLimiter）：server /v1/chat 路由内，per model
  令牌桶，允许瞬时突发 + 平均速率受限——适合保护上游模型配额
  （LLM 调用天然突发：用户连续对话后停顿，令牌桶比滑动窗口更贴合）

设计要点（滑动窗口）：
- 滑动窗口（非固定窗口）：窗口随时间滑动，无边界突刺；恢复机制 = 旧时间戳
  滑出窗口后名额自动释放，无需人工干预
- 被拒绝的请求不写入窗口（不占名额），Retry-After = 最旧命中滑出窗口的剩余秒数
- 内存态 per-client deque，threading.Lock 保护（uvicorn 线程池并发写）
- 环境变量：
    RATE_LIMIT_RPM       每窗口允许的请求数（0 = 禁用限流，默认）
    RATE_LIMIT_WINDOW_S  窗口长度（秒，默认 60）
  名义上 RPM = requests per minute；配合自定义窗口可表达
  "N 次 / M 秒"（验证脚本用短窗口加速恢复验证）

设计要点（令牌桶）：
- 桶容量 burst（允许瞬时突发），每秒填充 rpm/60 个令牌，请求消耗 1 个
- 惰性填充：每次 check 时按时间差补充令牌，无后台线程
- 初始满桶：启动后即可突发 burst 个请求（冷启动友好）
- 环境变量：
    MODEL_RATE_LIMITS  JSON，如
      {"deepseek-v4-pro": {"rpm": 10, "burst": 20},
       "deepseek-v4-flash": {"rpm": 60}}
    未设置/为空 = 禁用模型限流；burst 省略时默认等于 rpm
"""

import json
import logging
import math
import os
import threading
import time
from collections import defaultdict, deque
from typing import Protocol


class Limiter(Protocol):
    """限流器统一协议：滑动窗口与令牌桶共同满足的鸭子类型接口。

    check() 语义：
    - 登记一次访问（放行时）并判定是否放行
    - allowed=True  放行（已登记）
    - allowed=False 拒绝（不占名额），retry_after_s = 建议等待秒数（>=1）
    - now 参数可注入（测试用），缺省用 time.monotonic()
    """

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]: ...


class TokenBucketLimiter:
    """per-key 令牌桶限流器：允许瞬时突发 burst，平均速率受 rpm 限制。

    状态：per-key [tokens, last_refill_t]，惰性填充（无后台线程），
    threading.Lock 保护（uvicorn 线程池并发写）。
    """

    def __init__(self, rpm: int, burst: int):
        if rpm <= 0:
            raise ValueError("rpm 必须为正数（0 表示禁用，不应实例化本类）")
        if burst <= 0:
            raise ValueError("burst 必须为正数")
        self._rpm = rpm
        self._rate = rpm / 60.0            # tokens/s 填充速率
        self._capacity = float(burst)      # 桶容量 = 最大突发
        self._lock = threading.Lock()
        # per-key 状态：[当前令牌数, 上次填充时刻]；首次 check 时满桶初始化
        self._buckets: dict[str, list[float]] = defaultdict(
            lambda: [self._capacity, 0.0]
        )

    @property
    def rpm(self) -> int:
        return self._rpm

    @property
    def burst(self) -> int:
        return int(self._capacity)

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """消耗 1 个令牌并判定是否放行（放行才扣令牌，拒绝不扣）。

        Returns:
            (allowed, retry_after_s)：
            allowed=True  放行（令牌已消耗）
            allowed=False 拒绝（不扣令牌），retry_after_s=攒够 1 个令牌的建议秒数（>=1）
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets[key]
            if bucket[1] == 0.0:          # 首次访问：满桶初始化，计时起点=now
                bucket[0] = self._capacity
                bucket[1] = now
            elapsed = max(0.0, now - bucket[1])
            # 惰性填充：按时间差补令牌，封顶桶容量
            bucket[0] = min(self._capacity, bucket[0] + elapsed * self._rate)
            bucket[1] = now
            if bucket[0] >= 1.0:
                bucket[0] -= 1.0
                return True, 0.0
            # 令牌不足：算出攒够 1 个令牌需要的时间
            deficit = 1.0 - bucket[0]
            retry_after = deficit / self._rate if self._rate > 0 else float("inf")
            return False, max(1.0, math.ceil(retry_after))


class SlidingWindowLimiter:
    """per-key 滑动窗口计数限流器。"""

    def __init__(self, limit: int, window_s: float = 60.0):
        if limit <= 0:
            raise ValueError("limit 必须为正数（0 表示禁用，不应实例化本类）")
        if window_s <= 0:
            raise ValueError("window_s 必须为正数")
        self._limit = limit
        self._window_s = float(window_s)
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window_s(self) -> float:
        return self._window_s

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """登记一次访问并判定是否放行。

        Returns:
            (allowed, retry_after_s)：
            allowed=True  放行（时间戳已入窗）
            allowed=False 拒绝（不占名额），retry_after_s=建议等待秒数（>=1）
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[key]
            # 滑出窗口的旧命中出队（这就是"恢复"：名额随时间自动释放）
            while hits and now - hits[0] >= self._window_s:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = self._window_s - (now - hits[0])
                return False, max(1.0, math.ceil(retry_after))
            hits.append(now)
            return True, 0.0


def from_env() -> Limiter | None:
    """从环境变量构建限流器；RATE_LIMIT_RPM 未设置或为 0 时返回 None（禁用）。"""
    limit = int(os.getenv("RATE_LIMIT_RPM", "0"))
    if limit <= 0:
        return None
    window = float(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
    return SlidingWindowLimiter(limit=limit, window_s=window)


def from_env_for_models() -> dict[str, Limiter]:
    """解析 MODEL_RATE_LIMITS JSON 构建 per-model 令牌桶字典（模型级限流）。

    格式：{"model_name": {"rpm": int, "burst": int}, ...}
    - 未设置/为空 -> {}（禁用模型限流）
    - burst 省略时默认等于 rpm（突发 = 平均速率，等效固定速率）
    - JSON 解析失败 -> 记 warning 并返回 {}（fail-open：避免误封全部模型，
      配置错误只降级为"无限流"，不阻断网关启动）
    - rpm/burst 非法（<=0 / 非数字）的条目跳过，其余条目照常生效
    """
    raw = os.getenv("MODEL_RATE_LIMITS", "").strip()
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).warning(
            "MODEL_RATE_LIMITS 不是合法 JSON，模型级限流禁用（fail-open）",
            extra={"error": str(e)},
        )
        return {}
    if not isinstance(cfg, dict):
        logging.getLogger(__name__).warning(
            "MODEL_RATE_LIMITS 顶层必须是 JSON 对象，模型级限流禁用（fail-open）",
        )
        return {}
    limiters: dict[str, Limiter] = {}
    for model, params in cfg.items():
        try:
            rpm = int(params.get("rpm", 0))
            burst = int(params.get("burst", rpm))
        except (AttributeError, TypeError, ValueError):
            logging.getLogger(__name__).warning(
                "MODEL_RATE_LIMITS 条目非法，已跳过该模型",
                extra={"model": model},
            )
            continue
        if rpm > 0 and burst > 0:
            limiters[model] = TokenBucketLimiter(rpm=rpm, burst=burst)
        else:
            logging.getLogger(__name__).warning(
                "MODEL_RATE_LIMITS 条目 rpm/burst 须为正数，已跳过该模型",
                extra={"model": model, "rpm": rpm, "burst": burst},
            )
    return limiters
