import os

import httpx

from gateway.types import ChatRequest, ChatResponse, Usage

DEFAULT_BASE_URL = "https://api.deepseek.com"


class ResponsesAdapter:
    """OpenAI Responses API 协议适配器。

    与 Chat Completions 的关键差异：
    - 请求用 input 数组，content 是带 type 的结构化数组
    - system 提示词放在顶层 instructions 字段
    - max_tokens 叫 max_output_tokens
    - 响应回复藏在 output[].content[].text（双层嵌套）
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        base = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        path = os.getenv("RESPONSES_API_PATH", "/v1/responses")
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
        stop_reason = "stop"
        if data.get("status") == "incomplete":
            stop_reason = (data.get("incomplete_details") or {}).get("reason", "incomplete")
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
