"""AI 网关 Mock 上游服务：双协议（Anthropic Messages / OpenAI Responses）+ 故障注入。

用途：为六大功能模块验证脚本提供可控上游。验证脚本通过管理端点注入故障，
网关（server.py）通过环境变量把上游地址指向本服务。

协议端点（与网关适配器的默认 API_PATH 对齐）:
  POST /v1/messages   Anthropic Messages 协议（流式 SSE / 非流式 JSON / tool_use 结构化）
  POST /v1/responses  OpenAI Responses 协议（response.output_text.delta / completed）

管理端点（验证脚本专用）:
  POST /_control   注入故障策略（详见 ControlModel 字段说明）
  POST /_reset     清空故障策略与计数器
  GET  /_counters  上游视角的请求计数（重试验证的证据来源：网关重试 N 次 = 上游收到 N 次请求）
  GET  /_health    就绪探针

运行:
  uv run python mock_upstream.py --port 8902
"""

import argparse
import json
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock LLM Upstream")

_LOCK = threading.Lock()


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------- 故障注入与计数（全局单进程状态，脚本串行使用） ----------

class ControlModel:
    """故障注入策略。字段语义：

    fail_remaining   后续 N 个业务请求直接失败（每请求递减，归零后恢复正 常）
    fail_status      失败时返回的 HTTP 状态码（500/503/429/400/401...）
    fail_body_type   失败体 error.type（默认按状态码映射，429 用 rate_limit_error）
    retry_after      失败响应携带的 Retry-After 头（秒，字符串）
    abort_before_response  处理器直接抛异常（模拟连接中断/服务崩溃）
    kill_midstream_after   流式响应发出 K 个 delta 后断流（模拟流中途网络故障）
    delay_s          响应前固定延迟（秒，模拟慢响应）
    sse_chunks       流式正文 delta 数量（默认 6）
    sse_chunk_delay_ms  相邻 delta 的发送间隔（毫秒，默认 60，供流式间隔验证）
    reasoning_chunks 思考通道 delta 数量（默认 0）
    structured_mode  结构化输出内容模式：valid / wrong_type / missing_field
    """

    def __init__(self):
        self.fail_remaining = 0
        self.fail_status = 500
        self.fail_body_type: str | None = None
        self.retry_after: str | None = None
        self.abort_before_response = False
        self.kill_midstream_after = 0
        self.delay_s = 0.0
        self.sse_chunks = 6
        self.sse_chunk_delay_ms = 60
        self.reasoning_chunks = 0
        self.structured_mode = "valid"


control = ControlModel()
counters = {
    "messages": {"total": 0, "by_status": {}},
    "responses": {"total": 0, "by_status": {}},
    "attempts": [],   # [{route, status, at}] 上游视角的每次请求落点
}

_STATUS_TO_TYPE = {
    400: "invalid_request_error", 401: "authentication_error",
    404: "not_found_error", 429: "rate_limit_error",
    500: "api_error", 503: "service_unavailable", 504: "timeout_error",
}


def _record(route: str, status: int) -> None:
    with _LOCK:
        c = counters[route]
        c["total"] += 1
        c["by_status"][str(status)] = c["by_status"].get(str(status), 0) + 1
        counters["attempts"].append({"route": route, "status": status, "at": time.time()})


def _take_fault() -> dict | None:
    """取出本次请求应注入的故障（无故障返回 None）。"""
    with _LOCK:
        if control.abort_before_response:
            return {"abort": True}
        if control.fail_remaining > 0:
            control.fail_remaining -= 1
            return {"status": control.fail_status, "type": control.fail_body_type,
                    "retry_after": control.retry_after}
    return None


def _fault_response(route: str, fault: dict) -> JSONResponse:
    status = fault["status"]
    err_type = fault.get("type") or _STATUS_TO_TYPE.get(status, "api_error")
    _record(route, status)
    headers = {}
    if fault.get("retry_after"):
        headers["Retry-After"] = str(fault["retry_after"])
    return JSONResponse(status_code=status, headers=headers, content={
        "error": {"type": err_type, "message": f"mock 上游注入故障 (HTTP {status})"}
    })


# ---------- 结构化输出内容生成 ----------

def _schema_example(schema: dict) -> object:
    """按 JSON Schema 生成最小合规示例值（enum 取第一项，required 全覆盖）。"""
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "string":
        n = schema.get("minLength", 1)
        return "x" * int(n)
    if t == "integer":
        return max(int(schema.get("minimum", 1)), 1)
    if t == "number":
        return max(float(schema.get("minimum", 1.0)), 1.0)
    if t == "boolean":
        return True
    if t == "array":
        return [_schema_example(schema.get("items", {"type": "string"}))]
    if t == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        obj = {k: _schema_example(v) for k, v in props.items() if k in required or True}
        return obj
    return "x"


def _structured_object(schema: dict, mode: str) -> object:
    """生成结构化输出内容；wrong_type / missing_field 模式产出不合规对象。"""
    obj = _schema_example(schema)
    if mode == "wrong_type" and isinstance(obj, dict):
        # 把第一个属性改成错误类型（string->int / int->string）
        for key in obj:
            if isinstance(obj[key], str):
                obj[key] = 12345
                break
            if isinstance(obj[key], int) and not isinstance(obj[key], bool):
                obj[key] = "not-an-int"
                break
    if mode == "missing_field" and isinstance(obj, dict):
        required = schema.get("required", [])
        if required:
            obj.pop(required[0], None)
    return obj


# ---------- Anthropic Messages 协议 ----------

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    fault = _take_fault()
    if fault and fault.get("abort"):
        _record("messages", 0)
        raise RuntimeError("mock: 模拟服务崩溃/连接中断")
    if fault:
        return _fault_response("messages", fault)

    payload = await request.json()
    _record("messages", 200)
    system = payload.get("system") or ""
    is_stream = bool(payload.get("stream"))
    structured = bool(payload.get("tools"))
    if control.delay_s > 0:
        time.sleep(control.delay_s)

    if is_stream:
        return StreamingResponse(
            _anthropic_sse(payload, system, structured),
            media_type="text/event-stream",
        )

    if structured:
        schema = payload["tools"][0]["input_schema"]
        obj = _structured_object(schema, control.structured_mode)
        content = [{"type": "tool_use", "id": "toolu_mock", "name": "structured_output", "input": obj}]
        stop = "tool_use"
    else:
        text = f"mock-reply[system={system}]"
        content = [{"type": "text", "text": text}]
        stop = "end_turn"
    return JSONResponse(content={
        "id": "msg_mock", "type": "message", "role": "assistant",
        "content": content, "model": payload.get("model", "mock"),
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def _split_chunks(text: str, n: int) -> list[str]:
    """把 text 精确等分为至多 n 个非空片段（不丢字）。"""
    n = max(1, min(n, len(text)))
    k, m = divmod(len(text), n)
    parts, idx = [], 0
    for i in range(n):
        step = k + (1 if i < m else 0)
        parts.append(text[idx:idx + step])
        idx += step
    return [p for p in parts if p]


def _anthropic_sse(payload: dict, system: str, structured: bool):
    """增量 yield 的 Anthropic SSE：先发 start，再逐块发 delta（kill_midstream 可中途断流）。"""
    yield _sse("message_start", {
        "type": "message_start", "message": {"id": "msg_mock", "usage": {"input_tokens": 10}},
    })
    block_index = 0
    if structured:
        obj = _structured_object(payload["tools"][0]["input_schema"], control.structured_mode)
        full = json.dumps(obj, ensure_ascii=False)
        delta_type = "input_json_delta"
        yield _sse("content_block_start", {
            "type": "content_block_start", "index": block_index,
            "content_block": {"type": "tool_use", "id": "toolu_mock", "name": "structured_output", "input": {}},
        })
    else:
        full = f"mock-reply[system={system}]"
        delta_type = "text_delta"
        if control.reasoning_chunks:
            for i in range(control.reasoning_chunks):
                if control.sse_chunk_delay_ms > 0:
                    time.sleep(control.sse_chunk_delay_ms / 1000.0)
                yield _sse("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": f"思考{i}。"},
                })
            block_index = 1
        yield _sse("content_block_start", {
            "type": "content_block_start", "index": block_index,
            "content_block": {"type": "text", "text": ""},
        })

    chunks = _split_chunks(full, control.sse_chunks)
    for i, chunk in enumerate(chunks):
        if control.kill_midstream_after and i >= control.kill_midstream_after:
            time.sleep(control.sse_chunk_delay_ms / 1000.0)
            raise RuntimeError("mock: 流中途断连（未发 done）")
        if control.sse_chunk_delay_ms > 0 and i > 0:
            time.sleep(control.sse_chunk_delay_ms / 1000.0)
        if delta_type == "text_delta":
            delta = {"type": "text_delta", "text": chunk}
        else:   # input_json_delta
            delta = {"type": "input_json_delta", "partial_json": chunk}
        yield _sse("content_block_delta", {
            "type": "content_block_delta", "index": block_index, "delta": delta,
        })
    if structured:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
        stop = "tool_use"
    else:
        stop = "end_turn"
    yield _sse("message_delta", {
        "type": "message_delta", "delta": {"stop_reason": stop},
        "usage": {"output_tokens": 5},
    })
    yield _sse("message_stop", {"type": "message_stop"})


# ---------- OpenAI Responses 协议 ----------

@app.post("/v1/responses")
async def openai_responses(request: Request):
    fault = _take_fault()
    if fault and fault.get("abort"):
        _record("responses", 0)
        raise RuntimeError("mock: 模拟服务崩溃/连接中断")
    if fault:
        return _fault_response("responses", fault)

    payload = await request.json()
    _record("responses", 200)
    instructions = payload.get("instructions") or ""
    is_stream = bool(payload.get("stream"))
    fmt = (payload.get("text") or {}).get("format") or {}
    structured = fmt.get("type") == "json_schema"
    if control.delay_s > 0:
        time.sleep(control.delay_s)

    if is_stream:
        return StreamingResponse(_responses_sse(instructions, structured, fmt),
                                 media_type="text/event-stream")

    if structured:
        text = json.dumps(_structured_object(fmt["schema"], control.structured_mode), ensure_ascii=False)
    else:
        text = f"mock-reply[instructions={instructions}]"
    return JSONResponse(content={
        "id": "resp_mock", "object": "response",
        "output": [{"type": "message", "id": "msg_mock", "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
        "model": payload.get("model", "mock"), "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def _responses_sse(instructions: str, structured: bool, fmt: dict):
    """增量 yield 的 Responses SSE。"""
    yield _sse("response.created", {
        "type": "response.created", "response": {"id": "resp_mock", "status": "in_progress"},
    })
    if structured:
        full = json.dumps(_structured_object(fmt["schema"], control.structured_mode), ensure_ascii=False)
    else:
        full = f"mock-reply[instructions={instructions}]"
    chunks = _split_chunks(full, control.sse_chunks)
    for i, chunk in enumerate(chunks):
        if control.kill_midstream_after and i >= control.kill_midstream_after:
            raise RuntimeError("mock: 流中途断连")
        if control.sse_chunk_delay_ms > 0 and i > 0:
            time.sleep(control.sse_chunk_delay_ms / 1000.0)
        yield _sse("response.output_text.delta",
                   {"type": "response.output_text.delta", "delta": chunk})
    yield _sse("response.completed", {
        "type": "response.completed",
        "response": {"id": "resp_mock", "status": "completed",
                     "usage": {"input_tokens": 10, "output_tokens": 5}},
    })


# ---------- 管理端点 ----------

@app.post("/_control")
async def set_control(request: Request):
    payload = await request.json()
    with _LOCK:
        for key, value in payload.items():
            if hasattr(control, key):
                setattr(control, key, value)
    return {"ok": True, "control": {k: v for k, v in vars(control).items()}}


@app.post("/_reset")
async def reset_all():
    global control
    with _LOCK:
        control = ControlModel()
        for c in (counters["messages"], counters["responses"]):
            c["total"] = 0
            c["by_status"] = {}
        counters["attempts"] = []
    return {"ok": True}


@app.get("/_counters")
def get_counters():
    return counters


@app.get("/_health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
