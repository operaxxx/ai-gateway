"""限流与指标测试：滑动窗口/令牌桶限流器 + 指标注册表 + server 中间件集成。

测试范围：
1. SlidingWindowLimiter：放行/拒绝/Retry-After/窗口滑动恢复/并发安全
2. TokenBucketLimiter：突发消耗/耗尽拒绝/Retry-After/惰性填充恢复/key 隔离/并发安全
3. from_env_for_models：MODEL_RATE_LIMITS 解析（合法/非法 JSON/非法条目/默认禁用）
4. metrics：计数器、状态码分布、延迟分位数、错误率
5. HTTP 层：限流中间件对 /v1/chat 返回 429 + Retry-After + 统一错误体；
   /v1/metrics 不限流不计入自身统计
6. 多级限流协同：IP（scope=ip）与模型（scope=model）分级拒绝互不干扰

运行：
  uv run python -m pytest test_ratelimit_metrics.py -v
"""

import threading

import pytest
from fastapi.testclient import TestClient

from gateway import metrics
from gateway.ratelimit import (
    SlidingWindowLimiter,
    TokenBucketLimiter,
    from_env,
    from_env_for_models,
)
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


# ---------- TokenBucketLimiter ----------

class TestTokenBucketLimiter:
    def test_burst_consumption_then_reject(self):
        """初始满桶：连续请求放行 burst 个后拒绝。"""
        lim = TokenBucketLimiter(rpm=60, burst=3)
        t = 1000.0
        for i in range(3):
            allowed, _ = lim.check("m1", now=t + i * 0.01)
            assert allowed
        # 0.5s 只攒 0.5 个令牌（消耗后余 0.02 + 0.48 < 1），不足放行
        allowed, retry_after = lim.check("m1", now=t + 0.5)
        assert not allowed
        assert retry_after >= 1

    def test_retry_after_refill_math(self):
        """Retry-After = 攒够 1 个令牌所需秒数向上取整。

        rpm=60 -> rate=1 token/s；耗尽后 deficit=1.0 -> retry_after=1s。
        rpm=30 -> rate=0.5 token/s；deficit=1.0 -> retry_after=2s。
        """
        lim = TokenBucketLimiter(rpm=60, burst=1)
        t = 1000.0
        assert lim.check("m", now=t)[0]
        _, retry_after = lim.check("m", now=t + 0.1)
        assert retry_after == 1

        lim2 = TokenBucketLimiter(rpm=30, burst=1)
        assert lim2.check("m", now=t)[0]
        _, retry_after2 = lim2.check("m", now=t + 0.1)
        assert retry_after2 == 2

    def test_lazy_refill_recovers(self):
        """时间推进后惰性填充令牌，自动恢复放行（无后台线程）。"""
        lim = TokenBucketLimiter(rpm=60, burst=1)   # 1 token/s
        t = 1000.0
        assert lim.check("m", now=t)[0]
        assert not lim.check("m", now=t + 0.5)[0]   # 只攒了 0.5 个，不足 1
        assert lim.check("m", now=t + 1.5)[0]       # 攒够 1 个，恢复放行

    def test_capacity_caps_refill(self):
        """填充封顶桶容量：长时间空闲后也只回满到 burst，不无限累积。"""
        lim = TokenBucketLimiter(rpm=60, burst=2)
        t = 1000.0
        assert lim.check("m", now=t)[0]              # 满桶 2 → 消耗 → 1
        # 空闲 1 小时（可攒远超容量的令牌）也只回满到 burst=2
        assert lim.check("m", now=t + 3600)[0]       # 回满 2 → 消耗 → 1
        assert lim.check("m", now=t + 3600.01)[0]    # 1.01 → 消耗 → 0.01
        # 回满后最多再突发 burst 个：第 3 次（只攒 0.01 个）必须拒绝
        assert not lim.check("m", now=t + 3600.02)[0]

    def test_per_key_isolation(self):
        """不同 model 独立桶，互不影响。"""
        lim = TokenBucketLimiter(rpm=60, burst=1)
        t = 1000.0
        assert lim.check("model_a", now=t)[0]
        assert not lim.check("model_a", now=t + 0.1)[0]
        assert lim.check("model_b", now=t + 0.1)[0]   # model_b 满桶不受 model_a 影响

    def test_rejected_does_not_consume_token(self):
        """被拒绝的请求不扣令牌（不占名额）。"""
        lim = TokenBucketLimiter(rpm=60, burst=2)
        t = 1000.0
        assert lim.check("m", now=t)[0]
        assert lim.check("m", now=t + 0.1)[0]
        for i in range(5):   # 连续 5 次拒绝都不扣令牌
            assert not lim.check("m", now=t + 0.2 + i * 0.1)[0]
        # 若拒绝扣了令牌，这里就不可能立即恢复
        assert lim.check("m", now=t + 1.2)[0]

    def test_concurrent_access_thread_safe(self):
        """Lock 保护下并发放行数恰好 = burst。"""
        lim = TokenBucketLimiter(rpm=60, burst=10)
        allowed_count = []

        def worker():
            for _ in range(20):
                allowed_count.append(lim.check("m", now=None)[0])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(allowed_count) == 10   # 并发下恰好放行 burst 个

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            TokenBucketLimiter(rpm=0, burst=10)
        with pytest.raises(ValueError):
            TokenBucketLimiter(rpm=60, burst=0)


# ---------- from_env_for_models（MODEL_RATE_LIMITS 解析） ----------

class TestFromEnvForModels:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MODEL_RATE_LIMITS", raising=False)
        assert from_env_for_models() == {}

    def test_empty_string_disabled(self, monkeypatch):
        monkeypatch.setenv("MODEL_RATE_LIMITS", "  ")
        assert from_env_for_models() == {}

    def test_valid_config(self, monkeypatch):
        monkeypatch.setenv(
            "MODEL_RATE_LIMITS",
            '{"deepseek-v4-pro": {"rpm": 10, "burst": 20}, "deepseek-v4-flash": {"rpm": 60}}',
        )
        limiters = from_env_for_models()
        assert set(limiters.keys()) == {"deepseek-v4-pro", "deepseek-v4-flash"}
        assert limiters["deepseek-v4-pro"].rpm == 10
        assert limiters["deepseek-v4-pro"].burst == 20
        # burst 省略时默认 = rpm
        assert limiters["deepseek-v4-flash"].burst == 60

    def test_invalid_json_fail_open(self, monkeypatch):
        """非法 JSON：fail-open 返回 {}（禁用），不抛异常不阻断启动。"""
        monkeypatch.setenv("MODEL_RATE_LIMITS", "{not json")
        assert from_env_for_models() == {}

    def test_non_object_json_fail_open(self, monkeypatch):
        monkeypatch.setenv("MODEL_RATE_LIMITS", "[1,2,3]")
        assert from_env_for_models() == {}

    def test_invalid_entries_skipped(self, monkeypatch):
        """非法条目跳过，合法条目照常生效。"""
        monkeypatch.setenv(
            "MODEL_RATE_LIMITS",
            '{"bad1": {"rpm": 0, "burst": 5}, "bad2": {"rpm": "abc"}, "good": {"rpm": 10, "burst": 10}}',
        )
        limiters = from_env_for_models()
        assert set(limiters.keys()) == {"good"}


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


# ---------- 多级限流协同（IP 一级 + 模型二级） ----------

class TestMultiLevelRateLimit:
    """验证两级限流协同：IP 粗筛（中间件）在前，模型细筛（路由内）在后。"""

    def _chat(self, client, model="deepseek-v4-flash"):
        return client.post("/v1/chat", json={"model": model,
                                             "messages": [{"role": "user", "content": "hi"}]})

    def test_model_level_rejected_with_scope_model(self, client, monkeypatch):
        """(a) IP 未触限时模型桶耗尽 -> 429 + scope=model，IP 计数器不动。"""
        monkeypatch.setattr(server, "limiter", None)   # IP 限流禁用
        monkeypatch.setattr(server, "model_limiters",
                            {"deepseek-v4-flash": TokenBucketLimiter(rpm=60, burst=1)})
        assert self._chat(client).status_code == 200   # 消耗唯一令牌
        r = self._chat(client)
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        body = r.json()
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["scope"] == "model"
        assert body["error"]["model"] == "deepseek-v4-flash"
        assert body["error"]["retryable"] is True
        snap = client.get("/v1/metrics").json()
        assert snap["counters"]["model_rate_limited_total"] == 1
        assert snap["counters"]["rate_limited_total"] == 0   # IP 级未触发

    def test_ip_level_rejected_first_with_scope_ip(self, client, monkeypatch):
        """(b) IP 先触限 -> 中间件直接拒绝 scope=ip，模型级检查未执行。"""
        monkeypatch.setattr(server, "limiter", SlidingWindowLimiter(limit=1, window_s=60))
        monkeypatch.setattr(server, "model_limiters",
                            {"deepseek-v4-flash": TokenBucketLimiter(rpm=60, burst=1)})
        assert self._chat(client).status_code == 200   # IP 窗口占满 + 模型令牌扣 1
        r = self._chat(client)
        assert r.status_code == 429
        body = r.json()
        assert body["error"]["scope"] == "ip"
        snap = client.get("/v1/metrics").json()
        assert snap["counters"]["rate_limited_total"] == 1        # IP 级触发
        assert snap["counters"]["model_rate_limited_total"] == 0  # 模型级未触发

    def test_both_pass_then_allowed(self, client, monkeypatch):
        """(c) 两级都未触限 -> 正常放行，两个限流计数器均为 0。"""
        monkeypatch.setattr(server, "limiter", SlidingWindowLimiter(limit=10, window_s=60))
        monkeypatch.setattr(server, "model_limiters",
                            {"deepseek-v4-flash": TokenBucketLimiter(rpm=60, burst=10)})
        for _ in range(2):
            assert self._chat(client).status_code == 200
        snap = client.get("/v1/metrics").json()
        assert snap["counters"]["rate_limited_total"] == 0
        assert snap["counters"]["model_rate_limited_total"] == 0

    def test_other_model_unaffected(self, client, monkeypatch):
        """per-model 隔离：一个模型被拒不影响另一个模型。"""
        monkeypatch.setattr(server, "limiter", None)
        monkeypatch.setattr(server, "model_limiters",
                            {"deepseek-v4-flash": TokenBucketLimiter(rpm=60, burst=1)})
        assert self._chat(client, model="deepseek-v4-flash").status_code == 200
        assert self._chat(client, model="deepseek-v4-flash").status_code == 429
        # 未配置限流的模型不受影响
        assert self._chat(client, model="deepseek-v4-pro").status_code == 200
