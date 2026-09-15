"""测试脚本：对比两个模型流式输出，验证统一事件流的翻译正确性。

上游 SSE 事件序列完全不同（Anthropic 2 层嵌套 vs Responses 3 层嵌套），
但经过适配器翻译后，上层看到的都是同一种统一事件流（带 text/reasoning 通道）。
API key 从项目根目录 .env 读取，无需 export。

运行:
    uv run python test_streaming.py
"""

import time
from collections import Counter

from gateway.types import ChatRequest, Message
from gateway.gateway import Gateway


def stream_model(gw: Gateway, model: str) -> dict:
    """消费统一事件流，返回统计结果。"""
    req = ChatRequest(
        model=model,
        messages=[
            Message(role="system", content="你是一个简洁的助手"),
            Message(role="user", content="用一句话解释什么是API网关"),
        ],
        max_tokens=512,  # 推理模型（pro）要先"思考"再回答，预算给足
    )

    print(f"\n{'='*60}")
    print(f"模型: {model}")
    print(f"{'='*60}")

    start = time.time()
    first_token_at = None   # 首个"正文"delta 的延迟
    events: list = []
    rendered = []
    reasoning_chars = 0

    for ev in gw.stream(req):
        events.append(ev)
        if ev.type == "delta":
            if ev.channel == "text":
                if first_token_at is None:
                    first_token_at = time.time() - start  # 首字延迟
                rendered.append(ev.text)
            else:
                reasoning_chars += len(ev.text)
    elapsed = time.time() - start

    types = Counter(ev.type for ev in events)
    text_deltas = sum(1 for e in events if e.type == "delta" and e.channel == "text")
    reasoning_deltas = sum(1 for e in events if e.type == "delta" and e.channel == "reasoning")
    print(f"事件类型统计: {dict(types)}")
    print(f"delta 细分: 正文 {text_deltas} 个 / 思考 {reasoning_deltas} 个")
    if text_deltas == 0 and reasoning_deltas == 0:
        print("[WARN] 未收到任何 delta——多半是适配器监听的事件名与上游不符，"
              "可用 debug_dump_sse.py 核对真实事件名")
    print(f"首字延迟: {first_token_at:.2f}s" if first_token_at else "首字延迟: 无正文 delta")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"正文长度: {len(''.join(rendered))} 字符 / 思考长度: {reasoning_chars} 字符")

    done = next((e for e in events if e.type == "done"), None)
    errors = [e for e in events if e.type == "error"]
    if errors:
        print(f"错误: {errors[0].error}")
    if done:
        print(f"最终 usage: {done.usage}")
        print(f"最终 stop_reason: {done.stop_reason}")

    return {
        "model": model,
        "elapsed": elapsed,
        "ttft": first_token_at,
        "types": dict(types),
        "reasoning_chars": reasoning_chars,
        "usage": done.usage if done else None,
        "stop_reason": done.stop_reason if done else None,
    }


def _fmt_secs(v) -> str:
    return f"{v:.2f}s" if v is not None else "无"


def compare(results: list[dict]):
    """验证：两个协议翻译后的事件流是否同构。"""
    print(f"\n{'='*60}")
    print("【统一事件流同构性验证】")
    print(f"{'='*60}")

    if len(results) < 2:
        print("结果不足，无法对比")
        return

    a, b = results
    print(f"\n{'指标':<16} {a['model']:<22} {b['model']}")
    print("-" * 60)
    print(f"{'事件类型':<16} {a['types']} {b['types']}")
    print(f"{'首字延迟':<16} {_fmt_secs(a['ttft']):<16} {_fmt_secs(b['ttft'])}")
    print(f"{'总耗时':<16} {a['elapsed']:.2f}s{'':<14} {b['elapsed']:.2f}s")
    print(f"{'思考长度':<16} {a['reasoning_chars']} 字符{'':<14} {b['reasoning_chars']} 字符")
    print(f"{'usage':<16} {a['usage']} {b['usage']}")
    print(f"{'stop_reason':<16} {a['stop_reason']!s:<22} {b['stop_reason']}")

    # 同构性：两边的事件类型集合应该完全一致
    same = set(a["types"]) == set(b["types"])
    print(f"\n事件类型集合一致: {same}")
    if same:
        print("=> 两种上游协议被翻译成了同一种事件流，前端无需感知差异")


def main():
    gw = Gateway()
    results = []
    for model in ["deepseek-v4-flash", "deepseek-v4-pro"]:
        try:
            results.append(stream_model(gw, model))
        except Exception as e:
            print(f"\n{model} 调用失败: {e}")
    compare(results)


if __name__ == "__main__":
    main()
