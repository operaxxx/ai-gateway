from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal['system', 'user', 'assistant']

@dataclass(slots=True)
class Message:
    role: Role
    content: str

@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    response_format: dict | None = None   # JSON Schema 字典，None=自由输出

@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass(slots=True)
class ChatResponse:
    text: str
    model: str
    usage: Usage
    stop_reason: str
    # stop_reason 统一词表（所有适配器入向翻译的目标）:
    #   stop           = 自然生成完毕
    #   max_tokens     = 预算耗尽截断
    #   content_filter = 内容被过滤/拒答
    #   error          = 生成失败
    #   其他            = 上游新词表值透传（向前兼容）
    raw: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float | None = None   # 非流式总耗时（毫秒），由 Gateway 打点；非流式无 TTFT 概念

# ---------- 统一流式事件（方案 A：自定义极简格式） ----------
# 适配器把各厂商的 SSE 事件翻译成下面 4 种事件，前端只需实现一个解析器：
#   start : 生成开始（HTTP 已就绪，上游确认接收）
#   delta : 增量文本片段
#   done  : 生成结束，带最终 usage 和 stop_reason
#   error : 流中途出错（此时 HTTP 200 已发出，只能用事件报错）

StreamEventType = Literal["start", "delta", "done", "error"]
StreamChannel = Literal["text", "reasoning"]

@dataclass(slots=True)
class StreamEvent:
    type: StreamEventType
    text: str = ""                      # delta: 增量文本
    channel: StreamChannel = "text"     # delta: text=正文, reasoning=思考过程（推理模型）
    usage: Usage | None = None          # done: 最终用量
    stop_reason: str | None = None      # done: 统一后的停止原因
    error: str | None = None            # error: 错误信息
    # 计时指标（毫秒），由 Gateway 统一打点，只出现在 done/error 事件上：
    #   ttft_ms    = 发起上游请求 → 收到第一个 delta（text/reasoning 均算首 token）
    #   elapsed_ms = 发起上游请求 → 流结束（done/error）
    ttft_ms: float | None = None
    elapsed_ms: float | None = None
