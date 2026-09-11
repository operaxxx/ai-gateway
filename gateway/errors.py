"""统一异常分类体系（与厂商无关）。

调研依据：OpenAI / Anthropic / DeepSeek 三家错误码文档（HTTP 状态码 + JSON error body）。
设计目标：把上游厂商各异的错误表达，统一翻译成一套稳定的异常分类，
让上层（server.py / 重试器 / 监控）只依赖本模块的 category/type/retryable，
不感知具体厂商差异。

异常分类（category，4 大类）：
- network    网络层异常：连接失败 / DNS / 超时（httpx 抛出，未到 HTTP 层）
- client     客户端业务异常：4xx，上游拒绝请求，重试无意义（除非改请求）
- server     服务端异常：5xx，上游故障，可重试
- unexpected 未预期异常：代码 bug / 解析错误等兜底

错误码（type，细分语义，对齐三家用词）：
  network_timeout / network_connection / network_dns
  invalid_request_error(400) / authentication_error(401) / billing_error(402)
  permission_error(403) / not_found_error(404) / conflict_error(409)
  request_too_large(413) / invalid_parameters(422) / rate_limit_error(429)
  api_error(500) / service_unavailable(503) / timeout_error(504) / overloaded_error(529)

重试策略：retryable = (category in {network, server}) 或 type == rate_limit_error / conflict_error
"""

import json
from typing import Any

import httpx

CATEGORY_NETWORK = "network"
CATEGORY_CLIENT = "client"
CATEGORY_SERVER = "server"
CATEGORY_UNEXPECTED = "unexpected"

# type -> (category, retryable)
_TYPE_SPEC: dict[str, tuple[str, bool]] = {
    "network_timeout": (CATEGORY_NETWORK, True),
    "network_connection": (CATEGORY_NETWORK, True),
    "network_dns": (CATEGORY_NETWORK, True),
    "invalid_request_error": (CATEGORY_CLIENT, False),
    "authentication_error": (CATEGORY_CLIENT, False),
    "billing_error": (CATEGORY_CLIENT, False),
    "permission_error": (CATEGORY_CLIENT, False),
    "not_found_error": (CATEGORY_CLIENT, False),
    "conflict_error": (CATEGORY_CLIENT, True),
    "request_too_large": (CATEGORY_CLIENT, False),
    "invalid_parameters": (CATEGORY_CLIENT, False),
    "rate_limit_error": (CATEGORY_CLIENT, True),
    "api_error": (CATEGORY_SERVER, True),
    "service_unavailable": (CATEGORY_SERVER, True),
    "timeout_error": (CATEGORY_SERVER, True),
    "overloaded_error": (CATEGORY_SERVER, True),
    "unexpected_error": (CATEGORY_UNEXPECTED, False),
}

# HTTP 状态码 -> type（厂商 error.type 缺失时按状态码兜底）
_STATUS_TO_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "request_too_large",
    422: "invalid_parameters",
    429: "rate_limit_error",
    500: "api_error",
    502: "service_unavailable",
    503: "service_unavailable",
    504: "timeout_error",
    529: "overloaded_error",
}


class GatewayError(Exception):
    """LLM 调用统一异常。

    所有厂商各异的错误表达都翻译成本异常，上层只依赖 category/type/retryable。

    Attributes:
        category: 4 大类 network/client/server/unexpected
        type: 细分语义 type（对齐三家用词）
        message: 人类可读的错误描述
        retryable: 是否建议重试
        provider: 厂商标识 anthropic/openai（用于路由与日志）
        status_code: 上游 HTTP 状态码（网络异常为 None）
        request_id: 上游 request-id 头（追踪用）
        retry_after: 上游 Retry-After 头值（秒，限流时有用）
        raw: 上游响应体原文（调试用，不进对外响应）
    """

    def __init__(
        self,
        category: str,
        type: str,
        message: str,
        retryable: bool = False,
        provider: str = "",
        status_code: int | None = None,
        request_id: str | None = None,
        retry_after: str | None = None,
        raw: dict[str, Any] | None = None,
    ):
        self.category = category
        self.type = type
        self.message = message
        self.retryable = retryable
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id
        self.retry_after = retry_after
        self.raw = raw
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """统一错误响应体（脱敏后可返回给调用方）。raw 不对外。"""
        body: dict[str, Any] = {
            "category": self.category,
            "type": self.type,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.provider:
            body["provider"] = self.provider
        if self.status_code is not None:
            body["status_code"] = self.status_code
        if self.request_id:
            body["request_id"] = self.request_id
        if self.retry_after:
            body["retry_after"] = self.retry_after
        return body


def _spec(type_: str) -> tuple[str, bool]:
    return _TYPE_SPEC.get(type_, (CATEGORY_UNEXPECTED, False))


def from_httpx_error(exc: httpx.HTTPError, provider: str) -> GatewayError:
    """把 httpx 网络/超时异常翻译成 GatewayError(category=network)。

    httpx 异常继承链：HTTPError -> {TimeoutException, ConnectError, HTTPStatusError, ...}
    HTTPStatusError 转交 from_http_response 处理。
    """
    if isinstance(exc, httpx.TimeoutException):
        return GatewayError(
            CATEGORY_NETWORK, "network_timeout",
            f"上游请求超时: {exc}", True, provider,
        )
    if isinstance(exc, httpx.ConnectError):
        return GatewayError(
            CATEGORY_NETWORK, "network_connection",
            f"连接上游失败: {exc}", True, provider,
        )
    return GatewayError(
        CATEGORY_NETWORK, "network_connection",
        f"网络异常: {exc}", True, provider,
    )


def from_http_response(
    response: httpx.Response,
    provider: str,
    original_exc: Exception | None = None,
) -> GatewayError:
    """把上游 HTTP 非 2xx 响应翻译成 GatewayError。

    优先解析响应体里的 error.type（三家用词接近），缺失时按状态码兜底。
    同时提取 request-id / retry-after 头。
    """
    status = response.status_code
    body = _safe_json(response)
    error_obj = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error_obj, dict):
        error_obj = {}

    err_type = error_obj.get("type") or _STATUS_TO_TYPE.get(status, "unexpected_error")
    message = error_obj.get("message") or (
        str(original_exc) if original_exc else f"上游返回 {status}"
    )
    request_id = (
        response.headers.get("request-id")
        or response.headers.get("x-request-id")
    )
    retry_after = response.headers.get("retry-after")
    cat, retry = _spec(err_type)
    return GatewayError(
        cat, err_type, message, retry, provider, status, request_id, retry_after,
        body if isinstance(body, dict) else None,
    )


def from_unexpected(exc: Exception, provider: str = "") -> GatewayError:
    """把未预期异常包装成 GatewayError(category=unexpected)。"""
    return GatewayError(
        CATEGORY_UNEXPECTED, "unexpected_error",
        f"未预期异常: {type(exc).__name__}: {exc}", False, provider,
    )


def http_status_for(err: GatewayError) -> int:
    """GatewayError -> 下游返回给调用方的 HTTP 状态码。

    - client 且有上游 status -> 透传（401/404/429 等）
    - network -> 503（上游不可达）
    - server -> 502（上游故障）
    - unexpected -> 500
    """
    if err.category == CATEGORY_CLIENT and err.status_code is not None:
        return err.status_code
    if err.category == CATEGORY_NETWORK:
        return 503
    if err.category == CATEGORY_SERVER:
        return 502
    return 500


def _safe_json(response: httpx.Response) -> dict | None:
    try:
        return response.json()
    except (json.JSONDecodeError, ValueError):
        return None
