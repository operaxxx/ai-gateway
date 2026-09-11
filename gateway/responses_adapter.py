import os
from collections.abc import Iterator

import httpx

from gateway.env import load_env
from gateway.errors import from_http_response, from_httpx_error
from gateway.sse import iter_sse
from gateway.types import ChatRequest, ChatResponse, StreamEvent, Usage

DEFAULT_BASE_URL = "https://api.deepseek.com"


def _normalize_stop_reason(status: str, reason: str | None) -> str:
    """把 (status, incomplete_details.reason) 翻译成统一词表。

    统一词表只有 4 个值：stop / max_tokens / content_filter / error
    """
    if status == "completed":
        return "stop"
    if status == "failed":
        return "error"
    # incomplete：按截断原因细分；不认识的 reason 透传为 "incomplete"
    return {
        "max_output_tokens": "max_tokens",
        "content_filter": "content_filter",
    }.get(reason or "", "incomplete")


class ResponsesAdapter:
    """OpenAI Responses API 协议适配器。

    与 Chat Completions 的关键差异：
    - 请求用 input 数组，content 是带 type 的结构化数组
    - system 提示词放在顶层 instructions 字段
    - max_tokens 叫 max_output_tokens
    - 响应回复藏在 output[].content[].text（双层嵌套）
    - 流式增量: 正文 response.output_text.delta / 思考 response.reasoning_text.delta（delta 均为纯字符串）
    - 终止事件有三种: response.completed / response.incomplete / response.failed
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_env()
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        base = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("RESPONSES_API_PATH", "/v1/responses")
        self.api_url = base.rstrip("/") + path
        self.client = httpx.Client(timeout=60.0)

    def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            response = self.client.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self._build_payload(request),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise from_http_response(e.response, "openai", e) from e
        except httpx.HTTPError as e:
            raise from_httpx_error(e, "openai") from e
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
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as response:
                response.raise_for_status()
                for event_name, data in iter_sse(response):
                    if event_name == "response.created":
                        yield StreamEvent(type="start")
                    elif event_name == "response.reasoning_text.delta":
                        # 推理模型的思考增量（delta 为纯字符串）
                        chunk = data.get("delta", "")
                        if chunk:
                            yield StreamEvent(type="delta", text=chunk, channel="reasoning")
                    elif event_name == "response.output_text.delta":
                        # 正文增量（delta 为纯字符串）
                        chunk = data.get("delta", "")
                        if chunk:
                            yield StreamEvent(type="delta", text=chunk)
                    elif event_name in ("response.completed", "response.incomplete", "response.failed"):
                        # 终止事件有三种：正常完成 / 预算耗尽等截断 / 失败
                        resp = data.get("response", {})
                        usage_data = resp.get("usage", {})
                        input_tokens = usage_data.get("input_tokens", 0)
                        output_tokens = usage_data.get("output_tokens", 0)
                        stop_reason = _normalize_stop_reason(
                            resp.get("status", "completed"),
                            (resp.get("incomplete_details") or {}).get("reason"),
                        )
        except httpx.HTTPStatusError as e:
            # HTTP 头已发出（200），流中途的 HTTP 错误只能以事件形式传递
            err = from_http_response(e.response, "openai", e)
            yield StreamEvent(type="error", error=err.to_dict())
            return
        except httpx.HTTPError as e:
            # 网络层错误（连接/超时）同样以事件形式传递
            err = from_httpx_error(e, "openai")
            yield StreamEvent(type="error", error=err.to_dict())
            return
        yield StreamEvent(
            type="done",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=stop_reason,
        )

    def _build_payload(self, request: ChatRequest) -> dict:
        system_texts = [m.content for m in request.messages if m.role == "system"]
        input_messages = [
            {
                "role": m.role,
                "content": [{"type": "input_text", "text": m.content}],
            }
            for m in request.messages
            if m.role != "system"
        ]
        payload: dict = {"model": request.model, "input": input_messages}
        if system_texts:
            payload["instructions"] = "\n\n".join(system_texts)
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        # 结构化输出：用 text.format 指定 JSON Schema
        if request.response_format is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.response_format,
                }
            }
        return payload

    def _parse_response(self, fallback_model: str, data: dict) -> ChatResponse:
        text = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text += part.get("text", "")
                        break
        usage_data = data.get("usage", {})
        stop_reason = _normalize_stop_reason(
            data.get("status", "completed"),
            (data.get("incomplete_details") or {}).get("reason"),
        )
        return ChatResponse(
            text=text,
            model=data.get("model", fallback_model),
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
            ),
            stop_reason=stop_reason,
            raw=data,
        )
