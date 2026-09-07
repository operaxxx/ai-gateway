"""FastAPI HTTP 服务器：把 Gateway 暴露成统一 HTTP API。

下游 SSE 格式（方案 A：自定义极简格式，和 StreamEvent 一一对应）：
  event: start
  data: {"type":"start"}

  event: delta
  data: {"type":"delta","text":"你好","channel":"text"}

  event: delta
  data: {"type":"delta","text":"思考中...","channel":"reasoning"}

  event: done
  data: {"type":"done","usage":{"input_tokens":15,"output_tokens":5},"stop_reason":"stop"}

  event: error
  data: {"type":"error","error":"消息"}

运行:
  uv run uvicorn server:app --reload --port 8000
  curl http://localhost:8000/v1/chat -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}' -H "Content-Type: application/json"
"""

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gateway.gateway import Gateway, MODEL_ROUTES, StructuredOutputError
from gateway.types import ChatRequest, Message, StreamEvent

app = FastAPI(title="AI Gateway", version="0.1.0")

# CORS：允许浏览器前端直接调用（生产环境应限制 origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

gw = Gateway()


# ---------- API 层请求模型（Pydantic） ----------
# 和内部 ChatRequest 分开：API 层有 stream 字段，内部没有
# （"是否流式"是边界处的选择，不该渗入核心类型）

class MessageIn(BaseModel):
    role: str
    content: str


class ChatRequestIn(BaseModel):
    model: str
    messages: list[MessageIn]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    response_format: dict | None = None   # JSON Schema，None=自由输出


def _to_internal(req: ChatRequestIn) -> ChatRequest:
    """API 层 Pydantic 模型 → 内部统一抽象 dataclass。"""
    return ChatRequest(
        model=req.model,
        messages=[Message(role=m.role, content=m.content) for m in req.messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        response_format=req.response_format,
    )


# ---------- StreamEvent → 下游 SSE 序列化 ----------

def _sse_line(event_type: str, data: dict) -> str:
    """生成一行 SSE 事件（两行 field + 空行分隔）。"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
        })
    if ev.type == "error":
        return _sse_line("error", {"type": "error", "error": ev.error})
    return ""


# ---------- 路由 ----------

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


@app.post("/v1/chat")
def chat(req: ChatRequestIn):
    """统一聊天接口。

    - stream=false（默认）：返回 JSON
    - stream=true：返回 text/event-stream，事件格式见文件头注释
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

    internal_req = _to_internal(req)

    if req.stream:
        def generate():
            for ev in gw.stream(internal_req):
                yield _event_to_sse(ev)

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
        }

    return {
        "text": resp.text,
        "model": resp.model,
        "usage": {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
        "stop_reason": resp.stop_reason,
    }
