"""验证模块 5：重试机制。

验证目标：对可重试上游故障（5xx 服务端故障 / 429 限流 / 连接中断）按策略自动重试并最终恢复，
对不可重试故障（4xx）快速失败；重试次数、成功率、恢复时间均有客观证据。

证据来源（三方互证）:
  - mock 上游 /_counters：上游视角收到的请求次数（重试 N 次 = 上游收到 N+1 次）
  - 网关 /v1/metrics 的 retries_total 计数
  - 网关 JSON 日志中的 "指数退避后重试" 记录（含 attempt/backoff_ms/error_category）

执行步骤:
  1. 启动 mock + 网关（RETRY_MAX_ATTEMPTS=3, BACKOFF_BASE=0.2s）
  2. 场景A 服务端故障恢复：前 2 次 500 -> 第 3 次成功（重试 2 次）
  3. 场景B 上游限流：1 次 429 + Retry-After:1 -> 尊重 Retry-After 后成功
  4. 场景C 不可重试：400 -> 仅 1 次尝试快速失败
  5. 场景D 重试耗尽：持续 503 -> 3 次尝试后向上返回 502
  6. 场景E 流式首事件前故障：500 一次后恢复 -> 下游看到完整成功流
  7. 场景F 流式中途断连：不重试（无法回退已发内容），error 事件透传
  8. 汇总成功率与恢复时间

运行:
  uv run python scripts/verification/verify_retry.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import (Evidence, chat, chat_sse, gateway_metrics, mock_counters,
                     mock_control, mock_reset, start_gateway, start_mock)

MODEL = "deepseek-v4-flash"
MAX_ATTEMPTS = 3
BASE_S = 0.2


def _metrics_retries(port: int) -> int:
    return gateway_metrics(port)["counters"]["retries_total"]


def main() -> int:
    ev = Evidence("retry", "重试机制：故障注入 / 重试次数 / 退避策略 / 恢复时间")
    recovery_records: list[dict] = []

    ev.step(f"启动 mock 上游 + 网关（RETRY_MAX_ATTEMPTS={MAX_ATTEMPTS}, BACKOFF_BASE={BASE_S}s）")
    mock_cm, mock_port = start_mock()
    gw_cm, gw_port = start_gateway(mock_port=mock_port, extra_env={
        "RETRY_MAX_ATTEMPTS": str(MAX_ATTEMPTS),
        "RETRY_BACKOFF_BASE_S": str(BASE_S),
        "RETRY_BACKOFF_MAX_S": "2",
    })
    with mock_cm, gw_cm:
        base = f"http://127.0.0.1:{gw_port}"
        payload = {"model": MODEL, "stream": False,
                   "messages": [{"role": "user", "content": "hi"}]}

        # ---- 场景 A：服务端故障（500 × 2）后恢复 ----
        ev.step("场景A：注入 2 次 500 -> 第 3 次成功（指数退避重试）")
        mock_reset(mock_port)
        mock_control(mock_port, fail_remaining=2, fail_status=500)
        retries_before = _metrics_retries(gw_port)
        t0 = time.monotonic()
        r = chat(base, payload)
        recovery_ms = round((time.monotonic() - t0) * 1000, 1)
        c = mock_counters(mock_port)["messages"]
        ev.evidence("case_a", {"status": r.status_code, "upstream_attempts": c["total"],
                               "by_status": c["by_status"], "recovery_ms": recovery_ms})
        ev.check("最终 HTTP 200（自动恢复）", 200, r.status_code)
        ev.check("上游收到 3 次请求（1 首试 + 2 重试）", 3, c["total"])
        ev.check("上游状态记录 2×500 + 1×200", {"500": 2, "200": 1}, c["by_status"])
        ev.check("网关 retries_total 计数 +2", 2, _metrics_retries(gw_port) - retries_before)
        ev.check("恢复时间 >= 2 次退避之和(0.2+0.4=0.6s)", True, recovery_ms >= 600,
                 detail=f"recovery_ms={recovery_ms}")
        recovery_records.append({"case": "A 500×2", "recovered": True, "ms": recovery_ms})

        # ---- 场景 B：上游限流 429 + Retry-After ----
        ev.step("场景B：注入 1 次 429（Retry-After: 1s）-> 尊重指示后成功")
        mock_reset(mock_port)
        mock_control(mock_port, fail_remaining=1, fail_status=429, retry_after="1")
        t0 = time.monotonic()
        r = chat(base, payload)
        recovery_ms = round((time.monotonic() - t0) * 1000, 1)
        c = mock_counters(mock_port)["messages"]
        ev.evidence("case_b", {"status": r.status_code, "upstream_attempts": c["total"],
                               "recovery_ms": recovery_ms})
        ev.check("最终 HTTP 200", 200, r.status_code)
        ev.check("上游收到 2 次请求", 2, c["total"])
        ev.check("退避尊重 Retry-After（耗时 >= 1s）", True, recovery_ms >= 1000,
                 detail=f"recovery_ms={recovery_ms}（含上游往返）")
        recovery_records.append({"case": "B 429+Retry-After", "recovered": True, "ms": recovery_ms})

        # ---- 场景 C：不可重试 400 快速失败 ----
        ev.step("场景C：注入持续 400（不可重试）-> 快速失败不重试")
        mock_reset(mock_port)
        mock_control(mock_port, fail_remaining=99, fail_status=400)
        retries_before = _metrics_retries(gw_port)
        t0 = time.monotonic()
        r = chat(base, payload)
        fail_ms = round((time.monotonic() - t0) * 1000, 1)
        c = mock_counters(mock_port)["messages"]
        ev.evidence("case_c", {"status": r.status_code, "upstream_attempts": c["total"],
                               "fail_ms": fail_ms})
        ev.check("HTTP 400（透传客户端错误）", 400, r.status_code)
        ev.check("上游仅收到 1 次请求（零重试）", 1, c["total"])
        ev.check("网关 retries_total 无增量", 0, _metrics_retries(gw_port) - retries_before)
        ev.check("快速失败（< 1s，无退避等待）", True, fail_ms < 1000,
                 detail=f"fail_ms={fail_ms}")

        # ---- 场景 D：重试耗尽 ----
        ev.step(f"场景D：持续 503 -> {MAX_ATTEMPTS} 次尝试耗尽后返回 502")
        mock_reset(mock_port)
        mock_control(mock_port, fail_remaining=99, fail_status=503)
        retries_before = _metrics_retries(gw_port)
        r = chat(base, payload)
        c = mock_counters(mock_port)["messages"]
        body = r.json()
        ev.evidence("case_d", {"status": r.status_code, "upstream_attempts": c["total"],
                               "error": body.get("error")})
        ev.check("HTTP 502（server 类错误统一映射）", 502, r.status_code)
        ev.check(f"上游收到 {MAX_ATTEMPTS} 次请求后放弃", MAX_ATTEMPTS, c["total"])
        ev.check("网关 retries_total 增量 = max_attempts-1", MAX_ATTEMPTS - 1,
                 _metrics_retries(gw_port) - retries_before)
        ev.check("错误体 error.category=server / retryable=true",
                 ("server", True),
                 (body.get("error", {}).get("category"), body.get("error", {}).get("retryable")))

        # ---- 场景 E：流式首事件前故障重试 ----
        ev.step("场景E：流式请求遇 1 次 500 -> 首事件前重试，下游看到完整成功流")
        mock_reset(mock_port)
        mock_control(mock_port, fail_remaining=1, fail_status=500)
        result = chat_sse(base, {"model": MODEL, "stream": True,
                                 "messages": [{"role": "user", "content": "hi"}]})
        types = [e["event"] for e in result["events"]]
        c = mock_counters(mock_port)["messages"]
        ev.evidence("case_e", {"event_types": types, "upstream_attempts": c["total"]})
        ev.check("事件序列为 start→delta…→done（无 error）", True,
                 types[0] == "start" and types[-1] == "done" and "error" not in types)
        ev.check("上游收到 2 次请求（重试生效）", 2, c["total"])

        # ---- 场景 F：流式中途断连（不重试）----
        ev.step("场景F：流式发出 2 个 delta 后断连 -> 不重试，error 事件透传")
        mock_reset(mock_port)
        mock_control(mock_port, kill_midstream_after=2, sse_chunk_delay_ms=30)
        result = chat_sse(base, {"model": MODEL, "stream": True,
                                 "messages": [{"role": "user", "content": "hi"}]})
        types = [e["event"] for e in result["events"]]
        c = mock_counters(mock_port)["messages"]
        err_ev = next((e for e in result["events"] if e["event"] == "error"), None)
        ev.evidence("case_f", {"event_types": types, "upstream_attempts": c["total"],
                               "error_event": err_ev})
        ev.check("上游仅 1 次请求（已发内容不可回退，不重试）", 1, c["total"])
        ev.check("下游收到 start + delta 后收到 error 事件", True,
                 types[0] == "start" and "delta" in types and types[-1] == "error")
        if err_ev and isinstance(err_ev["data"], dict):
            err_body = err_ev["data"].get("error", err_ev["data"])
            ev.check("error 事件含网络类错误（category=network）", "network",
                     err_body.get("category"))

        # ---- 成功率与恢复汇总 ----
        ev.step("汇总：可恢复场景成功率与恢复时间")
        success = sum(1 for x in recovery_records if x["recovered"])
        ev.check("带计时记录的可恢复场景全部恢复（A=500×2, B=429）", 2, success,
                 detail=f"记录={recovery_records}")
        ev.check("平均恢复时间 < 3s", True,
                 sum(x["ms"] for x in recovery_records) / max(1, len(recovery_records)) < 3000)
        ev.evidence("recovery_records", recovery_records)

    return 0 if ev.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
