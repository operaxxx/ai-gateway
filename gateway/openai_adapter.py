import os

import httpx

from gateway.env import load_env
from gateway.errors import from_http_response, from_httpx_error
from gateway.types import ChatRequest, ChatResponse, Usage

DEFAULT_BASE_URL = "https://api.deepseek.com"

# 翻译表：Chat Completions 的 finish_reason -> 统一词表
_FINISH_REASON_MAP = {
    "stop": "stop",               # 自然生成完毕
    "length": "max_tokens",       # 预算耗尽
    "content_filter": "content_filter",
}


def _normalize_stop_reason(raw: str | None) -> str:
    """查表翻译；认识的翻译，不认识的原样透传（向前兼容新值）。"""
    if raw is None:
        return "stop"
    return _FINISH_REASON_MAP.get(raw, raw)


class OpenAIAdapter:
    """OpenAI Chat Completions 协议适配器，可指向 DeepSeek 等兼容中转站。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_env()
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        base = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("OPENAI_API_PATH", "/v1/chat/completions")
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

    def _build_payload(self, request: ChatRequest) -> dict:
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ]
        payload: dict = {"model": request.model, "messages": messages}
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _parse_response(self, fallback_model: str, data: dict) -> ChatResponse:
        choices = data.get("choices", [])
        text = ""
        stop_reason = "stop"
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "")
            stop_reason = _normalize_stop_reason(choices[0].get("finish_reason"))
        usage_data = data.get("usage", {})
        return ChatResponse(
            text=text,
            model=data.get("model", fallback_model),
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
            ),
            stop_reason=stop_reason,
            raw=data,
        )
