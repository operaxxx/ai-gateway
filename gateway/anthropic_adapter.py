import os

import httpx

from gateway.types import ChatRequest, ChatResponse, Usage

DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"


class AnthropicAdapter:
    """Anthropic Messages 协议适配器，可指向 DeepSeek 等兼容中转站。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        base = base_url or os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("ANTHROPIC_API_PATH", "/v1/messages")
        self.api_url = base.rstrip("/") + path
        self.client = httpx.Client(timeout=60.0)

    def complete(self, request: ChatRequest) -> ChatResponse:
        response = self.client.post(
            self.api_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=self._build_payload(request),
        )
        response.raise_for_status()
        return self._parse_response(request.model, response.json())

    def _build_payload(self, request: ChatRequest) -> dict:
        # Anthropic 的 system 是顶层字段，不放进 messages 数组
        system_texts = [m.content for m in request.messages if m.role == "system"]
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        # max_tokens 是 Anthropic 的必填字段，统一层里是可选的，这里给默认值
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
            stop_reason=data.get("stop_reason", "stop"),
            raw=data,
        )
