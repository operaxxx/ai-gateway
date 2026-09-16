import logging
import time
from collections.abc import Iterator

from gateway import metrics
from gateway.env import load_env
from gateway.errors import GatewayError, from_unexpected
from gateway.retry import RetryPolicy, backoff_delay_s, should_retry, sleep_backoff
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
    """统一网关：根据 model 名字路由到对应的适配器。

    重试机制：对 retryable 错误（network/server/429/409）做有限次指数退避重试，
    策略由 RETRY_* 环境变量控制（RetryPolicy）。流式仅在首事件发出前可重试。
    """

    def __init__(self):
        self._adapters: dict[type, object] = {}
        load_env()   # 重试策略读环境变量，.env 兜底（与适配器同语义：显式 export 永远赢）
        self.retry_policy = RetryPolicy.from_env()

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
        attempt = 0   # 已完成的尝试次数 - 1（0 = 首次尝试）
        while True:
            try:
                resp = adapter.complete(request)
                break
            except GatewayError as e:
                if should_retry(e, attempt, self.retry_policy):
                    delay = backoff_delay_s(e, attempt, self.retry_policy)
                    metrics.inc("retries_total")
                    logger.warning(
                        "上游失败，指数退避后重试",
                        extra={
                            "model": request.model,
                            "provider": provider,
                            "attempt": attempt + 1,
                            "max_attempts": self.retry_policy.max_attempts,
                            "backoff_ms": round(delay * 1000, 1),
                            "retry_after": e.retry_after,
                            "error_category": e.category,
                            "error_type": e.type,
                            "error_status": e.status_code,
                            "error_message": e.message,
                        },
                    )
                    sleep_backoff(delay)
                    attempt += 1
                    continue
                elapsed_ms = (time.monotonic() - started) * 1000.0
                metrics.inc("llm_errors_total")
                metrics.observe_llm_latency(elapsed_ms)
                logger.warning(
                    "complete 上游失败",
                    extra={
                        "model": request.model,
                        "provider": provider,
                        "attempts": attempt + 1,
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
                metrics.inc("llm_errors_total")
                metrics.observe_llm_latency(elapsed_ms)
                logger.exception(
                    "complete 未预期异常",
                    extra={
                        "model": request.model,
                        "provider": provider,
                        "error_message": str(e),
                        "elapsed_ms": round(elapsed_ms, 1),
                    },
                )
                raise from_unexpected(e, provider) from e
        # 总耗时在适配器返回后立即打点：只统计上游往返，不含本地结构化校验
        resp.elapsed_ms = (time.monotonic() - started) * 1000.0
        metrics.inc("llm_calls_total")
        metrics.observe_llm_latency(resp.elapsed_ms)
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

        结构化输出（response_format 非空）：中继 delta 的同时累积正文通道，
        done 事件前用累积文本跑 JSON Schema 校验，结论挂在 done 事件上
        （structured_ok/structured_parsed/structured_errors）。流式下 HTTP 200
        已发出、无法 422，校验失败不改变 done 语义，由下游按结论自行展示。

        异常兜底：adapter.stream() 抛出的 GatewayError/未预期异常，因 HTTP 200 头
        可能已发出，统一转成结构化 error 事件（携带 GatewayError.to_dict()）并打点。

        重试机制：仅当上游失败发生在"首事件发出前"（下游还什么都没收到）才允许
        整段重试；一旦 start/delta 已发给下游，流无法回退，中途失败直接透传
        error 事件。计时起点含重试等待（ttft/elapsed 口径 = 含重试的总耗时）。
        """
        adapter_cls = self._resolve_adapter(request.model)
        adapter = self._get_adapter(adapter_cls)
        provider = self._provider_for(adapter_cls)
        started = time.monotonic()

        # —— 重试安全区：偷看首事件决定是否重试（首事件未发给下游，可安全重来）——
        attempt = 0
        first: StreamEvent | None = None
        gen: Iterator[StreamEvent] = iter(())
        while True:
            gen = adapter.stream(request)
            try:
                first = next(gen, None)
            except GatewayError as e:
                first = StreamEvent(type="error", error=e.to_dict())
            except Exception as e:
                first = StreamEvent(
                    type="error", error=from_unexpected(e, provider).to_dict()
                )
            if (
                first is not None
                and first.type == "error"
                and isinstance(first.error, dict)
                and first.error.get("retryable")
                and (attempt + 1) < self.retry_policy.max_attempts
            ):
                err = _gateway_error_from_dict(first.error)
                delay = backoff_delay_s(err, attempt, self.retry_policy)
                metrics.inc("retries_total")
                logger.warning(
                    "stream 上游失败，指数退避后重试（首事件前）",
                    extra={
                        "model": request.model,
                        "provider": provider,
                        "attempt": attempt + 1,
                        "max_attempts": self.retry_policy.max_attempts,
                        "backoff_ms": round(delay * 1000, 1),
                        "retry_after": err.retry_after,
                        "error_category": err.category,
                        "error_type": err.type,
                        "error_status": err.status_code,
                        "error_message": err.message,
                    },
                )
                sleep_backoff(delay)
                attempt += 1
                continue
            break

        ttft_ms: float | None = None
        text_parts: list[str] = []   # 结构化校验用：累积正文通道增量
        logger.debug(
            "stream 开始",
            extra={
                "model": request.model,
                "provider": provider,
                "stream": True,
                "attempts": attempt + 1,
            },
        )

        def _events() -> Iterator[StreamEvent]:
            """把偷看的首事件放回迭代最前面。"""
            if first is not None:
                yield first
            yield from gen

        try:
            for ev in _events():
                if ev.type == "delta":
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - started) * 1000.0
                    # 结构化校验用：累积正文通道增量（reasoning/思考不参与校验）
                    if ev.channel == "text" and request.response_format is not None:
                        text_parts.append(ev.text)
                elif ev.type in ("done", "error"):
                    ev.ttft_ms = ttft_ms
                    ev.elapsed_ms = (time.monotonic() - started) * 1000.0
                    if ev.type == "done":
                        structured_extra: dict = {}
                        # 流式结构化输出：流结束后对累积正文做 schema 校验
                        if request.response_format is not None:
                            result = validate_structured("".join(text_parts), request.response_format)
                            ev.structured_ok = result.ok
                            if result.ok:
                                ev.structured_parsed = result.parsed
                            else:
                                ev.structured_errors = [
                                    {"loc": e.loc, "type": e.type, "message": e.message}
                                    for e in (result.errors or [])
                                ]
                            structured_extra = {"structured_ok": result.ok}
                            if not result.ok:
                                logger.warning(
                                    "流式结构化输出校验失败",
                                    extra={
                                        "model": request.model,
                                        "provider": provider,
                                        "error_category": "structured_output",
                                        "error_message": str(result.errors),
                                    },
                                )
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
                                **structured_extra,
                            },
                        )
                        metrics.inc("llm_calls_total")
                        metrics.observe_llm_latency(ev.elapsed_ms)
                        if ttft_ms is not None:
                            metrics.observe_llm_ttft(ttft_ms)
                    else:
                        err_dict = ev.error if isinstance(ev.error, dict) else {"message": str(ev.error)}
                        metrics.inc("llm_errors_total")
                        metrics.observe_llm_latency(ev.elapsed_ms)
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
            metrics.inc("llm_errors_total")
            metrics.observe_llm_latency(elapsed_ms)
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
            metrics.inc("llm_errors_total")
            metrics.observe_llm_latency(elapsed_ms)
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
            yield self._error_event(from_unexpected(e, provider), started, ttft_ms)

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


def _gateway_error_from_dict(d: dict) -> GatewayError:
    """把 error 事件携带的 GatewayError.to_dict() 还原为异常对象。

    用于流式首事件重试判定与退避计算（dict 里没有 raw，重试不需要它）。
    """
    return GatewayError(
        d.get("category", "unexpected"),
        d.get("type", "unexpected_error"),
        d.get("message", ""),
        retryable=bool(d.get("retryable")),
        provider=d.get("provider", ""),
        status_code=d.get("status_code"),
        request_id=d.get("request_id"),
        retry_after=d.get("retry_after"),
    )
