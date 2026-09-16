"""深度思考开关的 payload 翻译测试（纯函数，零网络）。

两个上游协议对思考模式的表达完全不同：
- Anthropic（deepseek-v4-flash）: "thinking": {"type": "enabled" | "disabled"}
  （DeepSeek 忽略 budget_tokens，不传）
- Responses（deepseek-v4-pro）: "reasoning": {"effort": "high" | "none"}（none=关闭）
thinking=None 表示跟随上游默认（DeepSeek 思考默认开启），不传任何字段。
"""

import pytest

from gateway.anthropic_adapter import AnthropicAdapter
from gateway.responses_adapter import ResponsesAdapter
from gateway.types import ChatRequest, Message


def make_request(model: str, thinking: bool | None) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[Message(role="user", content="你好")],
        thinking=thinking,
    )


@pytest.mark.parametrize("thinking, expected", [
    (True, {"type": "enabled"}),
    (False, {"type": "disabled"}),
])
def test_anthropic_thinking_translation(thinking, expected):
    adapter = AnthropicAdapter(api_key="test-key", base_url="https://mock")
    payload = adapter._build_payload(make_request("deepseek-v4-flash", thinking))
    assert payload["thinking"] == expected


def test_anthropic_thinking_none_omitted():
    adapter = AnthropicAdapter(api_key="test-key", base_url="https://mock")
    payload = adapter._build_payload(make_request("deepseek-v4-flash", None))
    assert "thinking" not in payload


@pytest.mark.parametrize("thinking, expected", [
    (True, {"effort": "high"}),
    (False, {"effort": "none"}),
])
def test_responses_thinking_translation(thinking, expected):
    adapter = ResponsesAdapter(api_key="test-key", base_url="https://mock")
    payload = adapter._build_payload(make_request("deepseek-v4-pro", thinking))
    assert payload["reasoning"] == expected


def test_responses_thinking_none_omitted():
    adapter = ResponsesAdapter(api_key="test-key", base_url="https://mock")
    payload = adapter._build_payload(make_request("deepseek-v4-pro", None))
    assert "reasoning" not in payload
