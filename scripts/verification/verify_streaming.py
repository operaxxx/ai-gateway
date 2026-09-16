"""验证模块 1：流式输出。

验证目标：数据以流式方式持续返回（非一次性到达），并采集完整的时序证据。

执行步骤:
  1. 启动 mock 上游（配置 8 个 delta × 80ms 间隔）与网关
  2. 发送 stream=true 请求，逐事件记录接收时间戳
  3. 校验 SSE 头/事件顺序/块数/块间隔/计时字段/内容完整性
  4. 对照组：stream=false 单次 JSON 返回

预期结果:
  - Content-Type 为 text/event-stream，带 Cache-Control: no-cache / X-Accel-Buffering: no
  - 事件顺序 start -> delta* -> done，正文 delta 数 >= 上游块数
  - 相邻 delta 接收间隔 >= 上游发送间隔的 60%（证明逐块到达而非缓冲后一次性下发）
  - done 事件携带 ttft_ms / elapsed_ms / usage / stop_reason，且与实测时间吻合
  - 拼接后的正文与上游发送内容一致

运行:
  uv run python scripts/verification/verify_streaming.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Evidence, chat, chat_sse, mock_control, start_gateway, start_mock

MODEL = "deepseek-v4-flash"   # Anthropic Messages 协议
CHUNKS = 8
CHUNK_DELAY_MS = 80
EXPECTED_TEXT = f"mock-reply[system=]"


def main() -> int:
    ev = Evidence("streaming", "流式输出：SSE 持续返回 + 时序证据")

    ev.step("启动 mock 上游 + 网关（mock 配置 8 个 delta，间隔 80ms）")
    mock_cm, mock_port = start_mock()
    gw_cm, gw_port = start_gateway(mock_port=mock_port)
    with mock_cm, gw_cm:
        base = f"http://127.0.0.1:{gw_port}"
        mock_control(mock_port, sse_chunks=CHUNKS, sse_chunk_delay_ms=CHUNK_DELAY_MS)

        # ---- 流式请求 ----
        ev.step(f"发送 stream=true 请求并逐事件记录接收时间戳", f"{base}/v1/chat")
        t_start = time.time()
        result = chat_sse(base, {
            "model": MODEL, "stream": True,
            "messages": [{"role": "user", "content": "讲个故事"}],
        })
        t_end = time.time()
        events = result["events"]
        ev.evidence("stream_run", {
            "started_at": time.strftime("%H:%M:%S", time.localtime(t_start)),
            "ended_at": time.strftime("%H:%M:%S", time.localtime(t_end)),
            "http_status": result["status"],
            "first_byte_ms": result["first_byte_ms"],
            "events": events,
        })

        deltas = [e for e in events if e["event"] == "delta"]
        done = next((e for e in events if e["event"] == "done"), None)
        types = [e["event"] for e in events]
        gaps = [d["gap_ms"] for d in deltas[1:]]   # 首个 delta 的 gap 相对 start，不计入

        ev.step("校验 SSE 响应头")
        ev.check("Content-Type 为 text/event-stream",
                 "text/event-stream", result["headers"].get("content-type", "").split(";")[0])
        ev.check("Cache-Control: no-cache（流禁用缓存）",
                 "no-cache", result["headers"].get("cache-control", ""))
        ev.check("X-Accel-Buffering: no（反代不缓冲）",
                 "no", result["headers"].get("x-accel-buffering", ""))

        ev.step("校验事件序列与块数")
        ev.check("首事件为 start", True, bool(types) and types[0] == "start")
        ev.check("末事件为 done", True, bool(types) and types[-1] == "done")
        ev.check("事件仅含 start/delta/done（无 error）",
                 True, set(types) <= {"start", "delta", "done"})
        ev.check(f"正文 delta 块数 >= 上游块数({CHUNKS})",
                 True, len(deltas) >= CHUNKS,
                 detail=f"实际收到 {len(deltas)} 个 delta")
        ev.evidence("delta_gaps_ms", gaps)

        ev.step("校验流式特征：delta 逐块间隔到达")
        # mock 每块间隔 80ms；若网关缓冲后一次性下发，间隔会聚集成 0
        min_gap = min(gaps) if gaps else 0.0
        ev.check(f"相邻 delta 最小间隔 >= {CHUNK_DELAY_MS*0.6:.0f}ms",
                 True, min_gap >= CHUNK_DELAY_MS * 0.6,
                 detail=f"间隔记录(ms): {gaps}")
        ev.check("总流时长 >= 上游发送时长(块数-1)*间隔 的 80%",
                 True, (events[-1]["t_ms"] if events else 0)
                 >= (CHUNKS - 1) * CHUNK_DELAY_MS * 0.8,
                 detail=f"流总时长 {events[-1]['t_ms'] if events else 0}ms")

        ev.step("校验计时字段（网关打点 vs 实测）")
        if done:
            d = done["data"]
            ev.check("done 含 stop_reason", True, d.get("stop_reason") == "stop")
            ev.check("done 含 usage(input/output_tokens)",
                     True, isinstance((d.get("usage") or {}).get("output_tokens"), int))
            ev.check("done 含 ttft_ms > 0", True, (d.get("ttft_ms") or 0) > 0,
                     detail=f"ttft_ms={d.get('ttft_ms')}")
            ev.check("done 含 elapsed_ms > 0", True, (d.get("elapsed_ms") or 0) > 0,
                     detail=f"elapsed_ms={d.get('elapsed_ms')}")
            # ttft 应接近实测首个 delta 到达时间（容差 500ms：含 start 事件与网络抖动）
            first_delta = next((e for e in deltas), None)
            if first_delta:
                drift = abs((d.get("ttft_ms") or 0) - first_delta["t_ms"])
                ev.check("ttft_ms 与实测首 delta 到达时间吻合(±500ms)",
                         True, drift <= 500,
                         detail=f"ttft_ms={d.get('ttft_ms')}, 实测={first_delta['t_ms']}ms")
        else:
            ev.check("收到 done 事件", True, False)

        ev.step("校验内容完整性")
        text = "".join(e["data"].get("text", "") for e in deltas)
        ev.check("拼接正文 == 上游发送内容", EXPECTED_TEXT, text)

        # ---- 对照组：非流式 ----
        ev.step("对照组：stream=false 单次 JSON 返回")
        r = chat(base, {"model": MODEL, "stream": False,
                        "messages": [{"role": "user", "content": "hi"}]})
        ev.check("非流式返回 JSON（非 event-stream）",
                 "application/json", r.headers.get("content-type", "").split(";")[0])
        ev.check("非流式响应含 text 与 elapsed_ms",
                 True, "text" in r.json() and "elapsed_ms" in r.json())

    return 0 if ev.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
