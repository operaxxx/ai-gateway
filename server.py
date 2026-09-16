"""FastAPI HTTP 服务器：把 Gateway 暴露成统一 HTTP API。

下游 SSE 格式（方案 A：自定义极简格式，和 StreamEvent 一一对应）：
  event: start
  data: {"type":"start"}

  event: delta
  data: {"type":"delta","text":"你好","channel":"text"}

  event: delta
  data: {"type":"delta","text":"思考中...","channel":"reasoning"}

  event: done
  data: {"type":"done","usage":{"input_tokens":15,"output_tokens":5},"stop_reason":"stop","ttft_ms":320.5,"elapsed_ms":1850.2}

  event: error
  data: {"type":"error","error":"消息","elapsed_ms":120.3}

  计时字段（毫秒，仅流结束事件携带）：
    ttft_ms    = 网关发起上游请求 → 收到第一个 delta（首 token 延迟）
    elapsed_ms = 网关发起上游请求 → 流结束；非流式 JSON 响应同名字段为总耗时

运行:
  uv run uvicorn server:app --reload --port 8000
  curl http://localhost:8000/v1/chat -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}' -H "Content-Type: application/json"
  curl http://localhost:8000/v1/prompts -d '{"id":"translator","name":"翻译","content":"把{{text}}从中文翻译成{{lang}}"}' -H "Content-Type: application/json"
  curl http://localhost:8000/v1/chat -d '{"model":"deepseek-v4-flash","prompt":{"id":"translator","variables":{"text":"你好","lang":"英文"}},"messages":[{"role":"user","content":"请开始"}]}' -H "Content-Type: application/json"
"""

import json
import os
import re
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gateway.env import load_env
from gateway.errors import GatewayError, from_unexpected, http_status_for
from gateway.gateway import Gateway, MODEL_ROUTES, StructuredOutputError
from gateway.logger_setup import new_request_id, setup_logging
from gateway.prompt_render import (
    InvalidTemplateError,
    MissingVariablesError,
    extract_variables,
    render,
)
from gateway.prompt_store import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
    PromptStore,
    PromptVersion,
    SqlitePromptStore,
)
from gateway.types import ChatRequest, Message, StreamEvent

app = FastAPI(title="AI Gateway", version="0.1.0")

# ---------- 日志 & request_id ----------
# setup_logging 配好 JsonFormatter + 可选文件轮转，根 logger 统一出口。
# middleware 在每个请求入口生成 request_id，通过 contextvar 让整条调用链
# （server → gateway → adapter）的日志自动带上同一个 request_id。
setup_logging()
import logging as _logging  # noqa: E402  — setup_logging 在顶层配完再取 logger
logger = _logging.getLogger(__name__)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    """每个请求生成 request_id，记录入口/出口日志。"""
    new_request_id()
    started = time.monotonic()
    logger.info(
        "请求进入",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None,
        },
    )
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.exception(
            "请求异常",
            extra={
                "path": request.url.path,
                "method": request.method,
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
        raise
    elapsed_ms = (time.monotonic() - started) * 1000.0
    logger.info(
        "请求完成",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "elapsed_ms": round(elapsed_ms, 1),
        },
    )
    return response

# CORS：允许浏览器前端直接调用（生产环境应限制 origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

gw = Gateway()


# ---------- 全局异常处理：统一错误响应 ----------

@app.exception_handler(GatewayError)
async def _handle_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
    """LLM 调用统一异常 -> HTTP 状态码 + 统一错误体。

    状态码映射见 errors.http_status_for：client 透传上游状态码，
    network->503，server->502，unexpected->500。
    上游 Retry-After 头透传（429 限流场景）。
    """
    status = http_status_for(exc)
    # retryable 分档：可重试的（限流/过载/5xx）用 WARNING，否则用 ERROR
    log_level = _logging.WARNING if exc.retryable else _logging.ERROR
    logger.log(
        log_level,
        "LLM 调用失败",
        extra={
            "path": request.url.path,
            "error_category": exc.category,
            "error_type": exc.type,
            "error_status": exc.status_code,
            "error_message": exc.message,
            "provider": exc.provider,
            "request_id_provider": exc.request_id,
            "retryable": exc.retryable,
            "retry_after": exc.retry_after,
        },
    )
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = exc.retry_after
    return JSONResponse(status_code=status, content={"error": exc.to_dict()}, headers=headers)


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """未预期异常兜底：包装成 unexpected GatewayError，返回 500 统一体。

    HTTPException/Pydantic 校验等 FastAPI 内置异常不被本 handler 接管
    （它们有更高优先级的内置 handler）。
    """
    logger.exception(
        "未预期异常",
        extra={"path": request.url.path, "error_message": str(exc)},
    )
    err = from_unexpected(exc)
    return JSONResponse(status_code=500, content={"error": err.to_dict()})


# ---------- Prompt 存储 ----------
# 介质由 env 决定（PROMPTS_DB_PATH），上层只依赖 PromptStore 协议，换介质换实现类即可
load_env()
store: PromptStore = SqlitePromptStore(os.environ.get("PROMPTS_DB_PATH", "prompts.db"))

# prompt id 规则：小写字母/数字开头，可含 - _（作为 slug 出现在 URL 和引用里）
_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


# ---------- API 层请求模型（Pydantic） ----------
# 和内部 ChatRequest 分开：API 层有 stream 字段，内部没有
# （"是否流式"是边界处的选择，不该渗入核心类型）

class MessageIn(BaseModel):
    role: str
    content: str


class PromptCreateIn(BaseModel):
    id: str                          # slug，作为引用锚点
    name: str
    description: str = ""
    content: str                     # Jinja2 模板：{{var}} 占位 + {% if %}/{% for %} 等


class PromptVersionCreateIn(BaseModel):
    content: str                     # 新版本模板内容（旧版本不可变）


class PromptRefIn(BaseModel):
    """调用方对模板的引用：gateway 服务端渲染，替代调用方自己拼 system prompt。"""
    id: str
    version: int | Literal["latest"] = "latest"
    variables: dict[str, Any] = {}   # 值任意 JSON 类型（list/dict 供循环使用）


class PromptRenderIn(BaseModel):
    """渲染预览请求（不调 LLM）。"""
    version: int | Literal["latest"] = "latest"
    variables: dict[str, Any] = {}


class ChatRequestIn(BaseModel):
    model: str
    messages: list[MessageIn]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    response_format: dict | None = None   # JSON Schema，None=自由输出
    prompt: PromptRefIn | None = None     # 提供时渲染为 system 消息插到最前
    thinking: bool | None = None          # 深度思考开关，None=跟随上游默认


def _to_internal(req: ChatRequestIn, messages: list[MessageIn]) -> ChatRequest:
    """API 层 Pydantic 模型 → 内部统一抽象 dataclass。"""
    return ChatRequest(
        model=req.model,
        messages=[Message(role=m.role, content=m.content) for m in messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        response_format=req.response_format,
        thinking=req.thinking,
    )


# ---------- StreamEvent → 下游 SSE 序列化 ----------

def _sse_line(event_type: str, data: dict) -> str:
    """生成一行 SSE 事件（两行 field + 空行分隔）。"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _timing_fields(ev: StreamEvent) -> dict:
    """提取 Gateway 打点的计时字段（毫秒，保留 1 位小数）；未打点则省略。"""
    fields = {}
    if ev.ttft_ms is not None:
        fields["ttft_ms"] = round(ev.ttft_ms, 1)
    if ev.elapsed_ms is not None:
        fields["elapsed_ms"] = round(ev.elapsed_ms, 1)
    return fields


def _event_to_sse(ev: StreamEvent) -> str:
    """把统一 StreamEvent 翻译成下游 SSE 行。"""
    if ev.type == "start":
        return _sse_line("start", {"type": "start"})
    if ev.type == "delta":
        return _sse_line("delta", {
            "type": "delta",
            "text": ev.text,
            "channel": ev.channel,
        })
    if ev.type == "done":
        usage = None
        if ev.usage:
            usage = {
                "input_tokens": ev.usage.input_tokens,
                "output_tokens": ev.usage.output_tokens,
            }
        return _sse_line("done", {
            "type": "done",
            "usage": usage,
            "stop_reason": ev.stop_reason,
            **_timing_fields(ev),
        })
    if ev.type == "error":
        return _sse_line("error", {
            "type": "error",
            "error": ev.error,
            **_timing_fields(ev),
        })
    return ""


# ---------- 路由 ----------

# 管理控制台（纯静态三件套，无构建步骤）；目录定位用 __file__，与启动 cwd 无关
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/", include_in_schema=False)
def index():
    """管理控制台首页。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    """列出网关支持的模型及其底层协议。"""
    return {
        "models": [
            {"id": model, "protocol": adapter.__name__}
            for model, adapter in MODEL_ROUTES.items()
        ]
    }


# ---------- Prompt 管理 ----------

def _resolve_version(prompt_id: str, version: int | Literal["latest"]) -> PromptVersion:
    """按版本号或 'latest' 解析模板版本；不存在直接转 HTTP 404。"""
    ver = (store.get_latest_version(prompt_id) if version == "latest"
           else store.get_version(prompt_id, version))
    if ver is None:
        raise HTTPException(404, f"prompt 版本不存在: {prompt_id}@{version}")
    return ver


@app.post("/v1/prompts", status_code=201)
def create_prompt(req: PromptCreateIn):
    """创建 prompt 模板（含版本 1）。变量从 content 提取存储，渲染时按声明校验。"""
    if not _SLUG_PATTERN.fullmatch(req.id):
        raise HTTPException(400, f"prompt id 不合法: {req.id!r}，需匹配 {_SLUG_PATTERN.pattern}")
    try:
        variables = extract_variables(req.content)   # 顺带完成语法校验
    except InvalidTemplateError as e:
        raise HTTPException(400, str(e))
    try:
        version = store.create_prompt(req.id, req.name, req.description, req.content, variables)
    except PromptAlreadyExistsError as e:
        raise HTTPException(409, str(e))
    return {
        "id": req.id, "name": req.name, "description": req.description,
        "version": version, "variables": variables,
    }


@app.get("/v1/prompts")
def list_prompts():
    """列出全部 prompt（含最新版本号）。"""
    return {"prompts": [
        {"id": p.id, "name": p.name, "description": p.description,
         "latest_version": p.latest_version, "created_at": p.created_at}
        for p in store.list_prompts()
    ]}


@app.get("/v1/prompts/{prompt_id}")
def get_prompt(prompt_id: str):
    """prompt 详情 + 版本历史（不含模板正文，正文按版本取）。"""
    p = store.get_prompt(prompt_id)
    if p is None:
        raise HTTPException(404, f"prompt 不存在: {prompt_id}")
    return {
        "id": p.id, "name": p.name, "description": p.description,
        "latest_version": p.latest_version, "created_at": p.created_at,
        "versions": [
            {"version": v.version, "variables": v.variables, "created_at": v.created_at}
            for v in store.list_versions(prompt_id)
        ],
    }


@app.post("/v1/prompts/{prompt_id}/versions", status_code=201)
def add_prompt_version(prompt_id: str, req: PromptVersionCreateIn):
    """追加新版本。版本不可变：没有修改端点，改内容 = 发新版本。"""
    try:
        variables = extract_variables(req.content)   # 顺带完成语法校验
    except InvalidTemplateError as e:
        raise HTTPException(400, str(e))
    try:
        version = store.add_version(prompt_id, req.content, variables)
    except PromptNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"id": prompt_id, "version": version, "variables": variables}


@app.get("/v1/prompts/{prompt_id}/versions/{version}")
def get_prompt_version(prompt_id: str, version: str):
    """取指定版本内容。version 支持整数或 'latest'。"""
    if version != "latest":
        try:
            version_num: int | Literal["latest"] = int(version)
        except ValueError:
            raise HTTPException(400, f"版本号不合法: {version!r}，应为整数或 'latest'")
    else:
        version_num = "latest"
    ver = _resolve_version(prompt_id, version_num)
    return {
        "id": ver.prompt_id, "version": ver.version, "content": ver.content,
        "variables": ver.variables, "created_at": ver.created_at,
    }


@app.post("/v1/prompts/{prompt_id}/render")
def render_prompt(prompt_id: str, req: PromptRenderIn):
    """渲染预览：不调用 LLM，用于调试模板与变量。"""
    ver = _resolve_version(prompt_id, req.version)
    try:
        rendered = render(ver.content, req.variables)
    except MissingVariablesError as e:
        raise HTTPException(400, {
            "error": "missing_variables",
            "message": str(e),
            "missing": e.missing,
        })
    return {"id": prompt_id, "version": ver.version, "rendered": rendered}


@app.delete("/v1/prompts/{prompt_id}", status_code=204)
def delete_prompt(prompt_id: str):
    """删除 prompt 及全部版本历史。不存在返回 404。"""
    try:
        store.delete_prompt(prompt_id)
    except PromptNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/v1/chat")
def chat(req: ChatRequestIn):
    """统一聊天接口。

    - stream=false（默认）：返回 JSON
    - stream=true：返回 text/event-stream，事件格式见文件头注释
    - 提供 prompt 引用时：gateway 服务端渲染模板为 system 消息插到最前，
      此时调用方 messages 不允许自带 system（渲染结果即 system）
    """
    if req.model not in MODEL_ROUTES:
        raise HTTPException(
            400,
            f"不支持的模型: {req.model}，支持: {list(MODEL_ROUTES.keys())}",
        )

    # v1 不支持流式 + 结构化输出组合（JSON 流片段无法解析）
    if req.stream and req.response_format is not None:
        raise HTTPException(
            400,
            "stream=true 与 response_format 不兼容：JSON 流的增量片段无法解析，请使用非流式模式",
        )

    # prompt 引用解析：渲染失败快速报错，不把残缺 prompt 发给 LLM
    messages_in = req.messages
    if req.prompt is not None:
        if any(m.role == "system" for m in messages_in):
            raise HTTPException(
                400,
                "使用 prompt 引用时 messages 不允许包含 system 消息：模板渲染结果将作为 system 消息",
            )
        ver = _resolve_version(req.prompt.id, req.prompt.version)
        try:
            system_text = render(ver.content, req.prompt.variables)
        except MissingVariablesError as e:
            raise HTTPException(400, {
                "error": "missing_variables",
                "message": str(e),
                "missing": e.missing,
            })
        messages_in = [MessageIn(role="system", content=system_text)] + list(messages_in)

    internal_req = _to_internal(req, messages_in)

    if req.stream:
        def generate():
            try:
                for ev in gw.stream(internal_req):
                    yield _event_to_sse(ev)
            except Exception as e:
                # 生成器内未预期异常（理论上 Gateway.stream 已兜底，此处为第二道防线）
                err = from_unexpected(e, req.model)
                yield _sse_line("error", {"type": "error", "error": err.to_dict()})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",        # 禁用缓存，流必须实时到达
                "X-Accel-Buffering": "no",          # nginx 反代时不缓冲
            },
        )

    try:
        resp = gw.complete(internal_req)
    except StructuredOutputError as e:
        # 422：输出内容不符合指定的 JSON Schema
        raise HTTPException(
            status_code=422,
            detail={
                "error": "structured_output_validation_failed",
                "message": str(e),
                "errors": [
                    {
                        "field": ".".join(str(x) for x in err.loc) or "root",
                        "type": err.type,
                        "message": err.message,
                    }
                    for err in e.errors
                ],
            },
        )

    # 非流式无 TTFT 概念，只返回上游往返总耗时
    elapsed_ms = round(resp.elapsed_ms, 1) if resp.elapsed_ms is not None else None

    # 结构化输出模式：返回解析后的对象而非纯文本
    if req.response_format is not None and "structured_output" in resp.raw:
        return {
            "structured_output": resp.raw["structured_output"],
            "model": resp.model,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            "stop_reason": resp.stop_reason,
            "elapsed_ms": elapsed_ms,
        }

    return {
        "text": resp.text,
        "model": resp.model,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
        "stop_reason": resp.stop_reason,
        "elapsed_ms": elapsed_ms,
    }


# 静态资源挂载放最后（所有 API 路由之后）：前缀 /static 提供 js/css，不与 /v1/*、/health 冲突
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
