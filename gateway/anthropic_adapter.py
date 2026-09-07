import os
from collections.abc import Iterator

import httpx

from gateway.env import load_env
from gateway.sse import iter_sse
from gateway.types import ChatRequest, ChatResponse, StreamEvent, Usage

DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"

# 翻译表：Anthropic 私有词表 -> 统一词表
# 统一词表只有 4 个值：stop / max_tokens / content_filter / error
_STOP_REASON_MAP = {
    "end_turn": "stop",           # 自然生成完毕
    "stop_sequence": "stop",      # 命中自定义停止序列，也算正常结束
    "max_tokens": "max_tokens",   # 预算耗尽
    "refusal": "content_filter",  # 拒答/被过滤
}


def _normalize_stop_reason(raw: str | None) -> str:
    """查表翻译；认识的翻译，不认识的原样透传（向前兼容新值）。"""
    if raw is None:
        return "stop"
    return _STOP_REASON_MAP.get(raw, raw)


class AnthropicAdapter:
    """Anthropic Messages 协议适配器，可指向 DeepSeek 等兼容中转站。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_env()
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        base = base_url or os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("ANTHROPIC_API_PATH", "/v1/messages")
        self.api_url = base.rstrip("/") + path
        self.client = httpx.Client(timeout=60.0)

    def complete(self, request: ChatRequest) -> ChatResponse:
        response = self.client.post(
            self.api_url,
            headers=self._headers(),
            json=self._build_payload(request),
        )
        response.raise_for_status()
        return self._parse_response(request.model, response.json())

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """流式请求：把上游 SSE 翻译成统一 StreamEvent，边收边 yield（中继模式）。"""
        payload = self._build_payload(request)
        payload["stream"] = True
        input_tokens = 0
        output_tokens = 0
        stop_reason = "stop"
        try:
            with self.client.stream(
                "POST",
                self.api_url,
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                for event_name, data in iter_sse(response):
                    if event_name == "message_start":
                        input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                        yield StreamEvent(type="start")
                    elif event_name == "content_block_delta":
                        delta = data.get("delta", {})
                        dtype = delta.get("type")
                        if dtype == "text_delta":
                            yield StreamEvent(type="delta", text=delta.get("text", ""))
                        elif dtype == "thinking_delta":
                            # 扩展思考模式下的思考增量
                            yield StreamEvent(type="delta", text=delta.get("thinking", ""), channel="reasoning")
                    elif event_name == "message_delta":
                        stop_reason = data.get("delta", {}).get("stop_reason", stop_reason)
                        output_tokens = data.get("usage", {}).get("output_tokens", 0)
        except httpx.HTTPError as e:
            # HTTP 头可能已发出（200），流中途的错误只能以事件形式传递
            yield StreamEvent(type="error", error=str(e))
            return
        yield StreamEvent(
            type="done",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=_normalize_stop_reason(stop_reason),
        )

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(self, request: ChatRequest) -> dict:
        system_texts = [m.content for m in request.messages if m.role == "system"]
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        max_tokens = request.max_tokens if request.max_tokens is not None else 1024
        payload: dict = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_texts:
            payload["system"] = "\n\n".join(system_texts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _parse_response(self, fallback_model: str, data: dict) -> ChatResponse:
        text = ""
        for part in data.get("content", []):
            if part.get("type") == "text":
                text += part.get("text", "")
        usage_data = data.get("usage", {})
        return ChatResponse(
            text=text,
            model=data.get("model", fallback_model),
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
            stop_reason=_normalize_stop_reason(data.get("stop_reason")),
            raw=data,
        )
