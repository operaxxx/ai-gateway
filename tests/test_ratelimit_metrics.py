"""限流与指标测试：滑动窗口限流器 + 指标注册表 + server 中间件集成。

测试范围：
1. SlidingWindowLimiter：放行/拒绝/Retry-After/窗口滑动恢复/并发安全
2. metrics：计数器、状态码分布、延迟分位数、错误率
3. HTTP 层：限流中间件对 /v1/chat 返回 429 + Retry-After + 统一错误体；
   /v1/metrics 不限流不计入自身统计

运行：
  uv run python -m pytest test_ratelimit_metrics.py -v
"""

import threading

import pytest
from fastapi.testclient import TestClient

from gateway import metrics
from gateway.ratelimit import SlidingWindowLimiter, from_env
import server


@pytest.fixture(autouse=True)
def _clean_metrics():
    metrics.reset()
    yield
    metrics.reset()


# ---------- SlidingWindowLimiter ----------

class TestSlidingWindowLimiter:
    def test_allows_within_limit_then_rejects(self):
        lim = SlidingWindowLimiter(limit=3, window_s=60)
        t = 1000.0
        for i in range(3):
            allowed, _ = lim.check("ip1", now=t + i)
            assert allowed
        allowed, retry_after = lim.check("ip1", now=t + 10)
        assert not allowed
        assert retry_after >= 1

    def test_per_key_isolation(self):
        lim = SlidingWindowLimiter(limit=1, window_s=60)
        assert lim.check("a", now=0.0)[0]
        assert not lim.check("a", now=1.0)[0]
        assert lim.check("b", now=1.0)[0]   # 不同客户端互不影响

    def test_window_slide_recovery(self):
        """旧命中滑出窗口后名额释放（恢复机制）。"""
        lim = SlidingWindowLimiter(limit=2, window_s=5.0)
        assert lim.check("ip", now=0.0)[0]
        assert lim.check("ip", now=0.1)[0]
        allowed, retry_after = lim.check("ip", now=1.0)
        assert not allowed
        assert retry_after == 4   # ceil(window - (now - oldest)) = ceil(5 - 1) = 4
        # 旧时间戳滑出后恢复
        assert lim.check("ip", now=5.5)[0]

    def test_concurrent_access_thread_safe(self):
        lim = SlidingWindowLimiter(limit=10, window_s=60)
        allowed_count = []

        def worker():
            for _ in range(20):
                allowed_count.append(lim.check("ip", now=None)[0])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(allowed_count) == 10   # 并发下恰好放行 limit 个

    def test_from_env_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("RATE_LIMIT_RPM", raising=False)
        assert from_env() is None

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("RATE_LIMIT_RPM", "5")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "30")
        lim = from_env()
        assert lim is not None and lim.limit == 5 and lim.window_s == 30


# ---------- metrics ----------

class TestMetrics:
    def test_counters_and_error_rate(self):
        metrics.inc("requests_total")
        metrics.inc("requests_total")
        metrics.inc("errors_total")
        metrics.inc_status(200)
        metrics.inc_status(400)
        snap = metrics.snapshot()
        assert snap["counters"]["requests_total"] == 2
        assert snap["counters"]["errors_total"] == 1
        assert snap["status_counts"] == {"200": 1, "400": 1}
        assert snap["error_rate"] == 0.5

    def test_latency_percentiles(self):
        for ms in [10, 20, 30, 40, 100]:
            metrics.observe_http_latency(ms)
        snap = metrics.snapshot()
        st = snap["http_latency"]
        assert st["count"] == 5
        assert st["min_ms"] == 10
        assert st["max_ms"] == 100
        assert st["p50_ms"] == 30
        assert st["p99_ms"] == 100

    def test_unknown_counter_ignored(self):
        metrics.inc("no_such_counter")
        assert "no_such_counter" not in metrics.snapshot()["counters"]


# ---------- HTTP 层集成 ----------

class _FakeGateway:
    def complete(self, req):
        from gateway.types import ChatResponse, Usage
        return ChatResponse(text="ok", model=req.model,
                            usage=Usage(1, 1), stop_reason="stop")


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "store", server.SqlitePromptStore(str(tmp_path / "prompts.db")))
    monkeypatch.setattr(server, "gw", _FakeGateway())
    return TestClient(server.app)


class TestRateLimitHttp:
    def test_429_with_retry_after_and_unified_body(self, client, monkeypatch):
        monkeypatch.setattr(server, "limiter", SlidingWindowLimiter(limit=2, window_s=60))
        for _ in range(2):
            r = client.post("/v1/chat", json={"model": "deepseek-v4-flash",
                                              "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 200
        r = client.post("/v1/chat", json={"model": "deepseek-v4-flash",
                                          "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        body = r.json()
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["retryable"] is True

    def test_metrics_endpoint_exempt_and_not_counted(self, client, monkeypatch):
        """监控端点不限流、不计入请求统计（避免自噪声）。"""
        monkeypatch.setattr(server, "limiter", SlidingWindowLimiter(limit=1, window_s=60))
        client.post("/v1/chat", json={"model": "deepseek-v4-flash",
                                      "messages": [{"role": "user", "content": "hi"}]})
        # 连续 scrape 3 次：不受限流影响
        for _ in range(3):
            r = client.get("/v1/metrics")
            assert r.status_code == 200
        snap = client.get("/v1/metrics").json()
        assert snap["counters"]["requests_total"] == 1   # 只有那 1 次 chat

    def test_health_not_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(server, "limiter", SlidingWindowLimiter(limit=1, window_s=60))
        for _ in range(3):
            assert client.get("/health").status_code == 200
