"""重试机制测试：策略纯函数 + Gateway 层重试行为（MockTransport 故障注入）。

测试范围：
1. RetryPolicy.from_env 环境变量解析
2. should_retry / backoff_delay_s 纯函数（指数退避、Retry-After 优先、封顶）
3. complete()：可重试错误（500）退避后重试直至成功；不可重试（400）快速失败；
   重试耗尽后向上抛 GatewayError
4. stream()：首事件前失败可重试；start 之后中途失败不重试（直接透传 error 事件）

运行：
  uv run python -m pytest test_retry.py -v
"""

import json

import httpx
import pytest

from gateway.anthropic_adapter import AnthropicAdapter
from gateway.errors import GatewayError
from gateway.gateway import Gateway
from gateway.retry import RetryPolicy, backoff_delay_s, should_retry
from gateway.types import ChatRequest, Message


def _req() -> ChatRequest:
    return ChatRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="hi")])


def _ok_anthropic_json() -> dict:
    return {
        "content": [{"type": "text", "text": "ok"}],
        "model": "deepseek-v4-flash",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "stop_reason": "end_turn",
    }


_OK_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
)


class _MockUpstream:
    """可编程故障注入的上游替身：按脚本依次返回响应；脚本耗尽后按请求类型回正常响应。"""

    def __init__(self, script: list[httpx.Response]):
        self.script = list(script)
        self.attempts = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        if self.script:
            return self.script.pop(0)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=iter([_OK_SSE.encode()]),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=_ok_anthropic_json())

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def _gateway_with(upstream: _MockUpstream, policy: RetryPolicy | None = None) -> Gateway:
    adapter = AnthropicAdapter(api_key="test", base_url="https://mock")
    adapter.client = upstream.client()
    gw = Gateway()
    gw._adapters[AnthropicAdapter] = adapter
    if policy is not None:
        gw.retry_policy = policy
    return gw


# ---------- 策略纯函数 ----------

class TestRetryPolicy:
    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("RETRY_MAX_ATTEMPTS", raising=False)
        monkeypatch.delenv("RETRY_BACKOFF_BASE_S", raising=False)
        policy = RetryPolicy.from_env()
        assert policy.max_attempts == 3
        assert policy.backoff_base_s == 0.5
        assert policy.backoff_max_s == 8.0

    def test_from_env_overrides(self, monkeypatch):
        monkeypatch.setenv("RETRY_MAX_ATTEMPTS", "1")
        monkeypatch.setenv("RETRY_BACKOFF_BASE_S", "0.1")
        policy = RetryPolicy.from_env()
        assert policy.max_attempts == 1
        assert policy.backoff_base_s == 0.1

    def test_should_retry_respects_retryable_and_budget(self):
        policy = RetryPolicy(max_attempts=3)
        err = GatewayError("server", "api_error", "boom", retryable=True)
        assert should_retry(err, 0, policy)   # 第 1 次失败后还能重试
        assert should_retry(err, 1, policy)
        assert not should_retry(err, 2, policy)  # 已尝试 3 次，达上限
        fatal = GatewayError("client", "invalid_request_error", "bad", retryable=False)
        assert not should_retry(fatal, 0, policy)

    def test_backoff_exponential_with_cap(self):
        policy = RetryPolicy(backoff_base_s=0.5, backoff_max_s=2.0)
        err = GatewayError("server", "api_error", "boom", retryable=True)
        assert backoff_delay_s(err, 0, policy) == 0.5
        assert backoff_delay_s(err, 1, policy) == 1.0
        assert backoff_delay_s(err, 2, policy) == 2.0   # 封顶
        assert backoff_delay_s(err, 10, policy) == 2.0

    def test_backoff_retry_after_wins(self):
        policy = RetryPolicy(backoff_base_s=0.5, backoff_max_s=2.0)
        err = GatewayError("client", "rate_limit_error", "slow down",
                           retryable=True, retry_after="1.5")
        assert backoff_delay_s(err, 0, policy) == 1.5
        huge = GatewayError("client", "rate_limit_error", "slow",
                            retryable=True, retry_after="999")
        assert backoff_delay_s(huge, 0, policy) == 2.0  # 封顶保护
        weird = GatewayError("client", "rate_limit_error", "x",
                             retryable=True, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
        assert backoff_delay_s(weird, 0, policy) == 0.5  # HTTP-date 不可解析 -> 回退指数


# ---------- complete() 重试行为 ----------

class TestCompleteRetry:
    def test_retry_until_success(self):
        """前两次 500，第三次成功：上游收到 3 次请求。"""
        upstream = _MockUpstream([
            httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}}),
            httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}}),
        ])
        gw = _gateway_with(upstream, RetryPolicy(max_attempts=3, backoff_base_s=0.0))
        resp = gw.complete(_req())
        assert resp.text == "ok"
        assert upstream.attempts == 3

    def test_no_retry_on_client_error(self):
        """400 不可重试：仅 1 次请求，向上抛 GatewayError(client)。"""
        upstream = _MockUpstream([
            httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "bad"}}),
        ])
        gw = _gateway_with(upstream, RetryPolicy(max_attempts=3, backoff_base_s=0.0))
        with pytest.raises(GatewayError) as ei:
            gw.complete(_req())
        assert ei.value.category == "client"
        assert upstream.attempts == 1

    def test_retry_exhausted_raises(self):
        """持续 500：达 max_attempts 后抛 server 异常。"""
        upstream = _MockUpstream([
            httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}}),
        ] * 10)
        gw = _gateway_with(upstream, RetryPolicy(max_attempts=3, backoff_base_s=0.0))
        with pytest.raises(GatewayError) as ei:
            gw.complete(_req())
        assert ei.value.category == "server"
        assert upstream.attempts == 3   # 1 次首试 + 2 次重试


# ---------- stream() 重试行为 ----------

class TestStreamRetry:
    def test_retry_before_first_event(self):
        """首事件前上游 500：整段重试，下游只看到成功流。"""
        upstream = _MockUpstream([
            httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}}),
        ])
        gw = _gateway_with(upstream, RetryPolicy(max_attempts=3, backoff_base_s=0.0))
        events = list(gw.stream(_req()))
        assert [e.type for e in events] == ["start", "delta", "done"]
        assert upstream.attempts == 2

    def test_no_retry_after_first_event(self):
        """start/delta 已发出后中途断流：不重试，透传 error 事件（无法回退已发内容）。"""

        def broken_stream_gen():
            lines = _OK_SSE.split("\n\n")
            yield (lines[0] + "\n\n").encode()   # message_start
            yield (lines[1] + "\n\n").encode()   # content_block_delta
            raise httpx.ReadError("connection reset by peer")

        adapter = AnthropicAdapter(api_key="test", base_url="https://mock")
        adapter.client = httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=broken_stream_gen())
        ))
        gw = Gateway()
        gw._adapters[AnthropicAdapter] = adapter
        gw.retry_policy = RetryPolicy(max_attempts=3, backoff_base_s=0.0)

        events = list(gw.stream(_req()))
        assert [e.type for e in events] == ["start", "delta", "error"]
        err = events[-1].error
        assert isinstance(err, dict) and err["category"] == "network"

    def test_non_retryable_first_error_no_retry(self):
        """首事件前上游 401：不可重试，直接透传 error 事件。"""
        upstream = _MockUpstream([
            httpx.Response(401, json={"error": {"type": "authentication_error", "message": "bad key"}}),
        ])
        gw = _gateway_with(upstream, RetryPolicy(max_attempts=3, backoff_base_s=0.0))
        events = list(gw.stream(_req()))
        assert [e.type for e in events] == ["error"]
        assert upstream.attempts == 1
        err = events[0].error
        assert isinstance(err, dict) and err["retryable"] is False
