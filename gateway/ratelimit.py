"""滑动窗口限流器：按客户端标识限制时间窗口内的请求数，超出返回 429。

设计要点：
- 滑动窗口（非固定窗口）：窗口随时间滑动，无边界突刺；恢复机制 = 旧时间戳
  滑出窗口后名额自动释放，无需人工干预
- 被拒绝的请求不写入窗口（不占名额），Retry-After = 最旧命中滑出窗口的剩余秒数
- 内存态 per-client deque，threading.Lock 保护（uvicorn 线程池并发写）
- 环境变量：
    RATE_LIMIT_RPM       每窗口允许的请求数（0 = 禁用限流，默认）
    RATE_LIMIT_WINDOW_S  窗口长度（秒，默认 60）
  名义上 RPM = requests per minute；配合自定义窗口可表达
  "N 次 / M 秒"（验证脚本用短窗口加速恢复验证）
"""

import math
import os
import threading
import time
from collections import defaultdict, deque


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


def from_env() -> SlidingWindowLimiter | None:
    """从环境变量构建限流器；RATE_LIMIT_RPM 未设置或为 0 时返回 None（禁用）。"""
    limit = int(os.getenv("RATE_LIMIT_RPM", "0"))
    if limit <= 0:
        return None
    window = float(os.getenv("RATE_LIMIT_WINDOW_S", "60"))
    return SlidingWindowLimiter(limit=limit, window_s=window)
