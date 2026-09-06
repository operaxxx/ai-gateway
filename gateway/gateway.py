from gateway.types import ChatRequest, ChatResponse
from gateway.anthropic_adapter import AnthropicAdapter
from gateway.responses_adapter import ResponsesAdapter

# 模型名 → 适配器的映射表
# 只支持两个模型，分别走不同协议，方便对比
MODEL_ROUTES: dict[str, type] = {
    "deepseek-v4-flash": AnthropicAdapter,   # Anthropic Messages 协议
    "deepseek-v4-pro": ResponsesAdapter,     # OpenAI Responses API 协议
}


class Gateway:
    """统一网关：根据 model 名字路由到对应的适配器。"""

    def __init__(self):
        self._adapters: dict[type, object] = {}

    def complete(self, request: ChatRequest) -> ChatResponse:
        adapter_cls = self._resolve_adapter(request.model)
        adapter = self._get_adapter(adapter_cls)
        return adapter.complete(request)

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
