import os

import httpx

from gateway.types import ChatRequest, ChatResponse, Usage

DEFAULT_BASE_URL = "https://api.deepseek.com"


class OpenAIAdapter:
    """OpenAI Chat Completions 协议适配器，可指向 DeepSeek 等兼容中转站。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        base = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("OPENAI_API_PATH", "/v1/chat/completions")
        self.api_url = base.rstrip("/") + path
        self.client = httpx.Client(timeout=60.0)

    def complete(self, request: ChatRequest) -> ChatResponse:
        response = self.client.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._build_payload(request),
        )
        response.raise_for_status()
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
            stop_reason = choices[0].get("finish_reason", "stop")
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
