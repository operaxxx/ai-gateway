import os
from collections.abc import Iterator

import httpx

from gateway.env import load_env
from gateway.errors import from_http_response, from_httpx_error
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
    "tool_use": "stop",           # 结构化输出场景，tool 调用完成即正常结束
}


def _normalize_stop_reason(raw: str | None) -> str:
    """查表翻译；认识的翻译，不认识的原样透传（向前兼容新值）。"""
    if raw is None:
        return "stop"
    return _STOP_REASON_MAP.get(raw, raw)


class AnthropicAdapter:
    """Anthropic Messages 协议适配器，指向 DeepSeek 官方 Anthropic 兼容端点。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_env()
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        base = base_url or os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("ANTHROPIC_API_PATH", "/v1/messages")
        self.api_url = base.rstrip("/") + path
        self.client = httpx.Client(timeout=60.0)

    def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            response = self.client.post(
                self.api_url,
                headers=self._headers(),
                json=self._build_payload(request),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise from_http_response(e.response, "anthropic", e) from e
        except httpx.HTTPError as e:
            raise from_httpx_error(e, "anthropic") from e
        return self._parse_response(
            request.model, response.json(),
            structured=request.response_format is not None,
        )

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
                if response.is_error:
                    # 流式响应体未读，必须在 with 块内 read 后才能解析错误 JSON
                    # （with 外访问 .json() 会抛 httpx.ResponseNotRead）
                    response.read()
                    yield StreamEvent(
                        type="error",
                        error=from_http_response(response, "anthropic").to_dict(),
                    )
                    return
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
            # 网络层错误（连接/超时）以事件形式传递
            err = from_httpx_error(e, "anthropic")
            yield StreamEvent(type="error", error=err.to_dict())
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
        # 深度思考开关：Anthropic 协议用 thinking.type（DeepSeek 忽略 budget_tokens，不传）
        if request.thinking is not None:
            payload["thinking"] = {"type": "enabled" if request.thinking else "disabled"}
        # 结构化输出：用 tool_use 模式（不带 tool_choice，DeepSeek thinking 模式不支持强制 tool_choice）
        if request.response_format is not None:
            payload["tools"] = [{
                "name": "structured_output",
                "description": "Return the result as a structured JSON object matching the schema",
                "input_schema": request.response_format,
            }]
        return payload

    def _parse_response(self, fallback_model: str, data: dict, *, structured: bool = False) -> ChatResponse:
        text = ""
        for part in data.get("content", []):
            if part.get("type") == "tool_use" and structured:
                # 结构化输出：tool_use 块的 input 就是解析好的 JSON 对象
                import json as _json
                text = _json.dumps(part.get("input", {}), ensure_ascii=False)
                break
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
