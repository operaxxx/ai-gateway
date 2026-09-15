"""离线验证：用 httpx.MockTransport 模拟两种上游 SSE，零 API 成本验证翻译正确性。

这是适配器模式的红利之一：stream() 的翻译逻辑完全不碰真实网络就能测试。
以后改了解析逻辑，跑一遍这个文件即可回归。

运行: uv run python test_stream_mock.py
"""

import httpx

from gateway.anthropic_adapter import AnthropicAdapter
from gateway.gateway import Gateway
from gateway.responses_adapter import ResponsesAdapter
from gateway.types import ChatRequest, Message

# ---------- 上游 SSE fixture（按两家官方文档的事件格式手写） ----------

ANTHROPIC_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":15}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"，世界"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
"""

# fixture 依据真实抓包（debug_dump_sse.py 的输出）手写，而非凭记忆：
# created -> reasoning item (reasoning_text.delta 思考) -> message item (output_text.delta 正文)
# -> completed；delta 字段均为纯字符串；
# 最后一个事件故意不带结尾空行，覆盖 iter_sse 的 EOF flush 路径
RESPONSES_SSE = """event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}

event: response.in_progress
data: {"type":"response.in_progress","response":{"id":"resp_1","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","status":"in_progress"}}

event: response.content_part.added
data: {"type":"response.content_part.added","content_index":0,"item_id":"rs_1","output_index":0,"part":{"type":"reasoning_text","text":""}}

event: response.reasoning_text.delta
data: {"type":"response.reasoning_text.delta","content_index":0,"delta":"让我们想想","item_id":"rs_1","output_index":0}

event: response.reasoning_text.done
data: {"type":"response.reasoning_text.done","content_index":0,"item_id":"rs_1","output_index":0,"text":"让我们想想"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","status":"completed"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","status":"in_progress"}}

event: response.content_part.added
data: {"type":"response.content_part.added","content_index":0,"item_id":"msg_1","output_index":1,"part":{"type":"output_text","text":""}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","content_index":0,"delta":"你好","item_id":"msg_1","output_index":1}

event: response.output_text.delta
data: {"type":"response.output_text.delta","content_index":0,"delta":"，世界","item_id":"msg_1","output_index":1}

event: response.output_text.done
data: {"type":"response.output_text.done","content_index":0,"item_id":"msg_1","output_index":1,"text":"你好，世界"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","status":"completed"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":15,"output_tokens":5}}}
"""


# ---------- mock 工具 ----------

def mock_client(body: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def error_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream exploded")
    return httpx.Client(transport=httpx.MockTransport(handler))


def run(adapter, model: str) -> list:
    req = ChatRequest(model=model, messages=[Message(role="user", content="hi")])
    return list(adapter.stream(req))


def main():
    a = AnthropicAdapter(api_key="test")
    a.client = mock_client(ANTHROPIC_SSE)
    r = ResponsesAdapter(api_key="test")
    r.client = mock_client(RESPONSES_SSE)

    a_events = run(a, "deepseek-v4-flash")
    r_events = run(r, "deepseek-v4-pro")

    print("Anthropic 上游 -> 统一事件流:")
    for e in a_events:
        print("  ", e)
    print("Responses 上游 -> 统一事件流:")
    for e in r_events:
        print("  ", e)

    # 1. 事件类型序列符合各自预期（responses 多一个思考 delta）
    a_types = [e.type for e in a_events]
    r_types = [e.type for e in r_events]
    assert a_types == ["start", "delta", "delta", "done"], f"Anthropic 事件序列异常: {a_types}"
    assert r_types == ["start", "delta", "delta", "delta", "done"], f"Responses 事件序列异常: {r_types}"

    # 2. 正文（text 通道）拼接一致；思考（reasoning 通道）只出现在 responses
    a_text = "".join(e.text for e in a_events if e.type == "delta" and e.channel == "text")
    r_text = "".join(e.text for e in r_events if e.type == "delta" and e.channel == "text")
    assert a_text == r_text == "你好，世界", f"正文不一致: {a_text!r} vs {r_text!r}"
    r_reason = "".join(e.text for e in r_events if e.type == "delta" and e.channel == "reasoning")
    assert r_reason == "让我们想想", f"思考通道异常: {r_reason!r}"
    assert all(e.channel == "text" for e in a_events if e.type == "delta"), "Anthropic 不应有思考通道"

    # 3. usage 已统一（Anthropic 分散两处 / Responses 只在最后，翻译后都在 done 里）
    a_done, r_done = a_events[-1], r_events[-1]
    assert a_done.usage == r_done.usage, f"usage 不一致: {a_done.usage} vs {r_done.usage}"
    assert (a_done.usage.input_tokens, a_done.usage.output_tokens) == (15, 5)

    print("[PASS] 两个协议被翻译成了同构的统一事件流")

    # 4. stop_reason 已统一：两种协议的正常结束都翻译成 "stop"
    assert a_done.stop_reason == r_done.stop_reason == "stop", (
        f"stop_reason 未统一: {a_done.stop_reason!r} vs {r_done.stop_reason!r}"
    )
    print(f"[PASS] stop_reason 已统一: 正常结束均为 {a_done.stop_reason!r}")

    # 4b. 翻译函数是纯函数，直接单测（无需任何 HTTP mock）
    from gateway.anthropic_adapter import _normalize_stop_reason as anth_norm
    from gateway.openai_adapter import _normalize_stop_reason as cc_norm
    from gateway.responses_adapter import _normalize_stop_reason as resp_norm

    assert anth_norm("end_turn") == "stop"
    assert anth_norm("stop_sequence") == "stop"
    assert anth_norm("max_tokens") == "max_tokens"
    assert anth_norm("refusal") == "content_filter"
    assert anth_norm(None) == "stop"
    assert anth_norm("brand_new_reason") == "brand_new_reason"  # 未知值透传

    assert resp_norm("completed", None) == "stop"
    assert resp_norm("incomplete", "max_output_tokens") == "max_tokens"
    assert resp_norm("incomplete", "content_filter") == "content_filter"
    assert resp_norm("failed", None) == "error"
    assert resp_norm("incomplete", "brand_new") == "incomplete"  # 未知 reason 兜底

    assert cc_norm("stop") == "stop"
    assert cc_norm("length") == "max_tokens"

    print("[PASS] stop_reason 翻译函数单测通过（3 个适配器 × 全词表）")

    # 5. 错误路径：上游 500 -> error 事件（而不是抛异常打断消费方）
    a_err = AnthropicAdapter(api_key="test")
    a_err.client = error_client()
    err_events = run(a_err, "deepseek-v4-flash")
    assert len(err_events) == 1 and err_events[0].type == "error", f"错误路径异常: {err_events}"
    print(f"[PASS] 上游 500 被翻译为 error 事件: {err_events[0].error!r}")

    # 6. Gateway 路由层：model -> 适配器 -> 统一事件流
    gw = Gateway()
    gw._adapters[AnthropicAdapter] = a
    gw._adapters[ResponsesAdapter] = r
    gw_types = [e.type for e in gw.stream(
        ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="hi")])
    )]
    assert gw_types == ["start", "delta", "delta", "delta", "done"], f"Gateway 路由异常: {gw_types}"
    print("[PASS] Gateway.stream 路由正常")


if __name__ == "__main__":
    main()
