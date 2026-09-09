import time
from collections.abc import Iterator

from gateway.structured_output import validate as validate_structured
from gateway.types import ChatRequest, ChatResponse, StreamEvent
from gateway.anthropic_adapter import AnthropicAdapter
from gateway.responses_adapter import ResponsesAdapter

# 模型名 → 适配器的映射表
# 只支持两个模型，分别走不同协议，方便对比
MODEL_ROUTES: dict[str, type] = {
    "deepseek-v4-flash": AnthropicAdapter,   # Anthropic Messages 协议
    "deepseek-v4-pro": ResponsesAdapter,     # OpenAI Responses API 协议
}


class StructuredOutputError(Exception):
    """结构化输出校验失败时抛出。"""

    def __init__(self, errors: list, raw_text: str = ""):
        self.errors = errors
        self.raw_text = raw_text
        messages = "; ".join(f"[{'.'.join(str(x) for x in e.loc) or 'root'}] {e.message}" for e in errors)
        super().__init__(f"结构化输出校验失败: {messages}")


class Gateway:
    """统一网关：根据 model 名字路由到对应的适配器。"""

    def __init__(self):
        self._adapters: dict[type, object] = {}

    def complete(self, request: ChatRequest) -> ChatResponse:
        adapter_cls = self._resolve_adapter(request.model)
        adapter = self._get_adapter(adapter_cls)
        started = time.monotonic()
        resp = adapter.complete(request)
        # 总耗时在适配器返回后立即打点：只统计上游往返，不含本地结构化校验
        resp.elapsed_ms = (time.monotonic() - started) * 1000.0
        # 结构化输出后处理：校验返回内容是否符合 schema
        if request.response_format is not None:
            result = validate_structured(resp.text, request.response_format)
            if not result.ok:
                raise StructuredOutputError(result.errors or [], result.raw_text)
            # 校验通过，把解析后的结构化数据附加到 raw
            resp.raw["structured_output"] = result.parsed
        return resp

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """流式接口：返回统一 StreamEvent 生成器，边收边转发（中继模式）。

        计时在本层统一打点，适配器不感知计时（口径不随厂商漂移）：
        - 起点 started：本生成器首次被迭代时，即适配器即将发起上游请求前
        - ttft_ms：第一个 delta 到达（text/reasoning 通道均算首 token）
        - elapsed_ms：done/error 事件时，即上游流结束
        """
        adapter_cls = self._resolve_adapter(request.model)
        adapter = self._get_adapter(adapter_cls)
        started = time.monotonic()
        ttft_ms: float | None = None
        for ev in adapter.stream(request):
            if ev.type == "delta" and ttft_ms is None:
                ttft_ms = (time.monotonic() - started) * 1000.0
            elif ev.type in ("done", "error"):
                ev.ttft_ms = ttft_ms
                ev.elapsed_ms = (time.monotonic() - started) * 1000.0
            yield ev

    def _resolve_adapter(self, model: str) -> type:
        if model not in MODEL_ROUTES:
            supported = ", ".join(MODEL_ROUTES.keys())
            raise ValueError(
                f"不支持的模型: {model!r}，当前仅支持: {supported}"
            )
        return MODEL_ROUTES[model]

    def _get_adapter(self, adapter_cls: type):
        # 适配器是无状态的（状态在环境变量里），可以缓存复用
        if adapter_cls not in self._adapters:
            self._adapters[adapter_cls] = adapter_cls()
        return self._adapters[adapter_cls]
