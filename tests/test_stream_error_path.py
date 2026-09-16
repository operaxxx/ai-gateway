"""流式错误路径回归：上游非 2xx 时 error 事件必须携带上游错误信息，而不是 ResponseNotRead。

背景 bug：
适配器在 `with client.stream()` 之外（except HTTPStatusError 里）调
from_http_response -> response.json()，此时流式响应体尚未 read，httpx 抛
ResponseNotRead("Attempted to access streaming response content, without having
called read()")，真实上游错误（4xx/5xx 的 body）被吞掉。

复现关键：
MockTransport handler 必须返回 content=迭代器 的响应（构造后 _content 未加载，
即真实 client.stream() 的"流式未读"状态）；text=/json= 参数会在构造时立即
设置 _content，无法复现本 bug（旧 test_stream_mock.error_client 因此漏测）。

运行：uv run python -m pytest tests/test_stream_error_path.py -v
"""

import httpx
import pytest

from gateway.anthropic_adapter import AnthropicAdapter
from gateway.responses_adapter import ResponsesAdapter
from gateway.types import ChatRequest, Message


def stream_error_client(status: int, body: bytes) -> httpx.Client:
    """返回处于"流式未读"状态的错误响应，复现真实 client.stream() 行为。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=iter([body]),
            headers={"content-type": "application/json"},
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("adapter_cls", [AnthropicAdapter, ResponsesAdapter])
@pytest.mark.parametrize("status,body,expect_type", [
    (400, b'{"error":{"type":"invalid_request_error","message":"temperature is not supported"}}',
     "invalid_request_error"),
    (500, b'{"error":{"type":"api_error","message":"internal upstream failure"}}',
     "api_error"),
])
def test_stream_upstream_error_becomes_error_event(adapter_cls, status, body, expect_type):
    """上游 4xx/5xx -> 单个 error 事件，且携带上游真实 type/message（而非 ResponseNotRead）。"""
    adapter = adapter_cls(api_key="test")
    adapter.client = stream_error_client(status, body)
    req = ChatRequest(model="x", messages=[Message(role="user", content="hi")])

    events = list(adapter.stream(req))

    assert len(events) == 1, f"应只有一个 error 事件: {events}"
    ev = events[0]
    assert ev.type == "error"
    assert ev.error["type"] == expect_type
    assert ev.error["status_code"] == status
    assert "ResponseNotRead" not in ev.error["message"]
    assert ("temperature is not supported" in ev.error["message"]
            or "internal upstream failure" in ev.error["message"])
