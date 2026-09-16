import logging
import time
from collections.abc import Iterator

from gateway.errors import GatewayError, from_unexpected
from gateway.structured_output import validate as validate_structured
from gateway.types import ChatRequest, ChatResponse, StreamEvent
from gateway.anthropic_adapter import AnthropicAdapter
from gateway.responses_adapter import ResponsesAdapter

logger = logging.getLogger(__name__)

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

    @staticmethod
    def _provider_for(adapter_cls: type) -> str:
        name = adapter_cls.__name__
        if "Anthropic" in name:
            return "anthropic"
        if "Responses" in name or "OpenAI" in name:
            return "openai"
        return name

    def complete(self, request: ChatRequest) -> ChatResponse:
        adapter_cls = self._resolve_adapter(request.model)
        adapter = self._get_adapter(adapter_cls)
        provider = self._provider_for(adapter_cls)
        started = time.monotonic()
        logger.debug(
            "complete 开始",
            extra={
                "model": request.model,
                "provider": provider,
                "stream": False,
            },
        )
        try:
            resp = adapter.complete(request)
        except GatewayError as e:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            logger.warning(
                "complete 上游失败",
                extra={
                    "model": request.model,
                    "provider": provider,
                    "error_category": e.category,
                    "error_type": e.type,
                    "error_status": e.status_code,
                    "error_message": e.message,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
            raise
        except Exception as e:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            logger.exception(
                "complete 未预期异常",
                extra={
                    "model": request.model,
                    "provider": provider,
                    "error_message": str(e),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
            raise from_unexpected(e, request.model) from e
        # 总耗时在适配器返回后立即打点：只统计上游往返，不含本地结构化校验
        resp.elapsed_ms = (time.monotonic() - started) * 1000.0
        # 结构化输出后处理：校验返回内容是否符合 schema
        if request.response_format is not None:
            result = validate_structured(resp.text, request.response_format)
            if not result.ok:
                logger.warning(
                    "结构化输出校验失败",
                    extra={
                        "model": request.model,
                        "provider": provider,
                        "error_category": "structured_output",
                        "error_message": str(result.errors),
                    },
                )
                raise StructuredOutputError(result.errors or [], result.raw_text)
            # 校验通过，把解析后的结构化数据附加到 raw
            resp.raw["structured_output"] = result.parsed
        logger.info(
            "complete 成功",
            extra={
                "model": request.model,
                "provider": provider,
                "elapsed_ms": round(resp.elapsed_ms, 1),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "stop_reason": resp.stop_reason,
            },
        )
        return resp

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """流式接口：返回统一 StreamEvent 生成器，边收边转发（中继模式）。

        计时在本层统一打点，适配器不感知计时（口径不随厂商漂移）：
        - 起点 started：本生成器首次被迭代时，即适配器即将发起上游请求前
        - ttft_ms：第一个 delta 到达（text/reasoning 通道均算首 token）
        - elapsed_ms：done/error 事件时，即上游流结束

        异常兜底：adapter.stream() 抛出的 GatewayError/未预期异常，因 HTTP 200 头
        可能已发出，统一转成结构化 error 事件（携带 GatewayError.to_dict()）并打点。
        """
        adapter_cls = self._resolve_adapter(request.model)
        adapter = self._get_adapter(adapter_cls)
        provider = self._provider_for(adapter_cls)
        started = time.monotonic()
        ttft_ms: float | None = None
        logger.debug(
            "stream 开始",
            extra={
                "model": request.model,
                "provider": provider,
                "stream": True,
            },
        )
        try:
            for ev in adapter.stream(request):
                if ev.type == "delta" and ttft_ms is None:
                    ttft_ms = (time.monotonic() - started) * 1000.0
                elif ev.type in ("done", "error"):
                    ev.ttft_ms = ttft_ms
                    ev.elapsed_ms = (time.monotonic() - started) * 1000.0
                    if ev.type == "done":
                        logger.info(
                            "stream 成功结束",
                            extra={
                                "model": request.model,
                                "provider": provider,
                                "stream": True,
                                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                                "elapsed_ms": round(ev.elapsed_ms, 1),
                                "input_tokens": ev.usage.input_tokens if ev.usage else None,
                                "output_tokens": ev.usage.output_tokens if ev.usage else None,
                                "stop_reason": ev.stop_reason,
                            },
                        )
                    else:
                        err_dict = ev.error if isinstance(ev.error, dict) else {"message": str(ev.error)}
                        logger.warning(
                            "stream 中途失败",
                            extra={
                                "model": request.model,
                                "provider": provider,
                                "stream": True,
                                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                                "elapsed_ms": round(ev.elapsed_ms, 1),
                                "error_category": err_dict.get("category"),
                                "error_type": err_dict.get("type"),
                                "error_status": err_dict.get("status_code"),
                                "error_message": err_dict.get("message"),
                            },
                        )
                yield ev
        except GatewayError as e:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            logger.warning(
                "stream 上游失败",
                extra={
                    "model": request.model,
                    "provider": provider,
                    "stream": True,
                    "error_category": e.category,
                    "error_type": e.type,
                    "error_status": e.status_code,
                    "error_message": e.message,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
            yield self._error_event(e, started, ttft_ms)
        except Exception as e:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            logger.exception(
                "stream 未预期异常",
                extra={
                    "model": request.model,
                    "provider": provider,
                    "stream": True,
                    "error_message": str(e),
                    "elapsed_ms": round(elapsed_ms, 1),
                },
            )
            yield self._error_event(from_unexpected(e, request.model), started, ttft_ms)

    @staticmethod
    def _error_event(err: GatewayError, started: float, ttft_ms: float | None) -> StreamEvent:
        """把 GatewayError 转成 error 事件，补齐计时字段。"""
        ev = StreamEvent(type="error", error=err.to_dict())
        ev.ttft_ms = ttft_ms
        ev.elapsed_ms = (time.monotonic() - started) * 1000.0
        return ev

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
