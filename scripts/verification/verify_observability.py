"""验证模块 4：可观测数据（指标收集 + 可视化报告）。

验证目标：/v1/metrics 暴露的关键指标与结构化日志能真实反映系统行为，
并可生成 HTML 可视化报告。

执行步骤:
  1. 启动 mock 上游 + 网关（LOG_FILE 落盘 JSON 日志）
  2. 抓取基线指标 -> 混合负载（3 非流式 + 3 流式成功，2 非法模型 400）
  3. 负载期间采样网关进程资源占用（CPU% / RSS，ps 采集）
  4. 抓取负载后指标，校验计数/错误率/延迟分位数与实际负载一致
  5. 解析 JSON 日志，校验日志字段与指标互相印证
  6. 生成 HTML 可视化报告（内联 SVG 柱状图，零前端依赖）

预期结果:
  - requests_total 增量 == 8；llm_calls_total == 6；errors_total == 2；error_rate == 0.25
  - status_counts: 200x6, 400x2
  - llm_latency 样本 6 个且 p50 > 0；llm_ttft 样本 3 个（仅流式）
  - 日志含 3 条 "complete 成功" + 3 条 "stream 成功结束"；elapsed_ms/ttft_ms 字段存在
  - HTML 报告生成且包含 SVG 图表

运行:
  uv run python scripts/verification/verify_observability.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import (REPORT_DIR, Evidence, chat, chat_sse, gateway_metrics,
                     process_stats, start_gateway, start_mock)

MODEL = "deepseek-v4-flash"
LOG_FILE = REPORT_DIR / "gateway.log"
HTML_REPORT = REPORT_DIR / "observability_report.html"


def _run_workload(base: str, ev: Evidence) -> list[dict]:
    """混合负载：返回每请求的客户端观测耗时记录。"""
    records = []
    ev.step("负载：3 次非流式 + 3 次流式（成功路径）")
    for i in range(3):
        t0 = time.monotonic()
        r = chat(base, {"model": MODEL, "stream": False,
                        "messages": [{"role": "user", "content": f"hi {i}"}]})
        records.append({"kind": "non-stream", "ok": r.status_code == 200,
                        "client_ms": round((time.monotonic() - t0) * 1000, 1)})
    for i in range(3):
        t0 = time.monotonic()
        result = chat_sse(base, {"model": MODEL, "stream": True,
                                 "messages": [{"role": "user", "content": f"流 {i}"}]})
        records.append({"kind": "stream", "ok": any(e["event"] == "done" for e in result["events"]),
                        "client_ms": round((time.monotonic() - t0) * 1000, 1)})
    return records


def _run_error_workload(base: str, ev: Evidence) -> None:
    ev.step("负载：2 次非法模型（注入 400 错误）")
    for _ in range(2):
        chat(base, {"model": "no-such-model",
                    "messages": [{"role": "user", "content": "x"}]})


def _svg_bars(records: list[dict]) -> str:
    """客户端观测耗时柱状图（内联 SVG）。"""
    vals = [r["client_ms"] for r in records]
    if not vals:
        return ""
    w, h, pad = 560, 180, 30
    max_v = max(vals) or 1
    bw = (w - 2 * pad) / len(vals)
    bars = []
    for i, v in enumerate(vals):
        bh = (v / max_v) * (h - 60)
        color = "#4a90d9" if records[i]["kind"] == "non-stream" else "#3aa675"
        bars.append(
            f'<rect x="{pad + i * bw + 4:.0f}" y="{h - pad - bh:.0f}" '
            f'width="{bw - 8:.0f}" height="{bh:.0f}" fill="{color}" rx="2"/>'
            f'<text x="{pad + i * bw + bw / 2:.0f}" y="{h - pad - bh - 5:.0f}" '
            f'font-size="10" text-anchor="middle" fill="#333">{v:.0f}</text>'
            f'<text x="{pad + i * bw + bw / 2:.0f}" y="{h - 12:.0f}" '
            f'font-size="9" text-anchor="middle" fill="#888">#{i + 1}</text>'
        )
    return (f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{pad}" y="16" font-size="12" fill="#333">'
            f'每请求耗时 (ms)：蓝=非流式 绿=流式</text>{"".join(bars)}</svg>')


def _html_report(counters: dict, stats: dict | None, records: list[dict],
                 res_samples: list[dict]) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counters.items())
    res_rows = "".join(
        f"<tr><td>样本{i + 1}</td><td>{s['cpu_percent']}</td><td>{s['rss_mb']}</td></tr>"
        for i, s in enumerate(res_samples))
    lat = stats or {}
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>AI Gateway 可观测性报告</title>
<style>
body{{font-family:-apple-system,'PingFang SC',sans-serif;margin:24px;color:#222}}
table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ddd;padding:6px 14px;font-size:13px}}
th{{background:#f5f5f5}} h2{{margin-top:28px}} .note{{color:#777;font-size:12px}}
</style></head><body>
<h1>AI Gateway 可观测性验证报告</h1>
<p class="note">生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')} ·
数据源 /v1/metrics + JSON 结构化日志 + ps 进程采样</p>
<h2>指标计数</h2>
<table><tr><th>指标</th><th>值</th></tr>{rows}</table>
<h2>LLM 延迟分位数</h2>
<table><tr><th>样本数</th><th>avg</th><th>p50</th><th>p95</th><th>p99</th><th>max</th></tr>
<tr><td>{lat.get('count', 0)}</td><td>{lat.get('avg_ms')}</td><td>{lat.get('p50_ms')}</td>
<td>{lat.get('p95_ms')}</td><td>{lat.get('p99_ms')}</td><td>{lat.get('max_ms')}</td></tr></table>
<h2>每请求耗时</h2>{_svg_bars(records)}
<h2>网关进程资源采样</h2>
<table><tr><th>采样</th><th>CPU %</th><th>RSS MB</th></tr>{res_rows}</table>
</body></html>"""


def main() -> int:
    ev = Evidence("observability", "可观测数据：指标收集 / 日志印证 / 资源采样 / HTML 报告")

    ev.step("启动 mock 上游 + 网关（LOG_FILE 落盘 JSON 日志）")
    mock_cm, mock_port = start_mock()
    gw_cm, gw_port = start_gateway(mock_port=mock_port)
    with mock_cm, gw_cm as gw_proc:
        base = f"http://127.0.0.1:{gw_port}"
        import httpx
        httpx.post(f"http://127.0.0.1:{mock_port}/_control",
                   json={"sse_chunk_delay_ms": 20}, timeout=10)

        baseline = gateway_metrics(gw_port)
        ev.step("基线指标抓取", f"baseline requests_total={baseline['counters']['requests_total']}")

        records = _run_workload(base, ev)
        res_samples = []
        for _ in range(3):
            res_samples.append(process_stats(gw_proc.pid))
            time.sleep(0.05)
        _run_error_workload(base, ev)
        ev.evidence("client_records", records)
        ev.evidence("resource_samples", res_samples)

        after = gateway_metrics(gw_port)
        ev.step("负载后指标抓取与一致性校验")
        counters = after["counters"]
        d_requests = counters["requests_total"] - baseline["counters"]["requests_total"]
        ev.check("requests_total 增量 == 8（6 成功 + 2 失败）", 8, d_requests)
        ev.check("llm_calls_total == 6（全部成功）", 6, counters["llm_calls_total"])
        ev.check("llm_errors_total == 0（400 在路由层被拦，未到上游）",
                 0, counters["llm_errors_total"])
        ev.check("errors_total == 2", 2, counters["errors_total"])
        ev.check("error_rate == 0.25", 0.25, after["error_rate"])
        ev.check("status_counts 200:6 / 400:2",
                 {"200": 6, "400": 2}, {k: after["status_counts"].get(k, 0)
                                        for k in ("200", "400")})

        lat = after.get("llm_latency") or {}
        ev.check("llm_latency 样本 6 个", 6, lat.get("count"))
        ev.check("llm_latency p50 > 0", True, (lat.get("p50_ms") or 0) > 0,
                 detail=f"p50={lat.get('p50_ms')}ms p95={lat.get('p95_ms')}ms")
        ttft = after.get("llm_ttft") or {}
        ev.check("llm_ttft 样本 3 个（仅流式请求有首 token）", 3, ttft.get("count"))
        ev.evidence("metrics_after", after)

        ev.step("解析 JSON 结构化日志（与指标互相印证）")
        lines = [json.loads(l) for l in LOG_FILE.read_text().splitlines() if l.strip()]
        ok_complete = [l for l in lines if l.get("msg") == "complete 成功"]
        ok_stream = [l for l in lines if l.get("msg") == "stream 成功结束"]
        done_400 = [l for l in lines if l.get("msg") == "请求完成" and l.get("status") == 400]
        ev.check("日志 3 条 complete 成功", 3, len(ok_complete))
        ev.check("日志 3 条 stream 成功结束", 3, len(ok_stream))
        ev.check("日志 2 条 status=400 请求完成", 2, len(done_400))
        ev.check("成功日志带 elapsed_ms 计时字段",
                 True, all("elapsed_ms" in l for l in ok_complete + ok_stream))
        ev.check("成功日志带 request_id（全链路追踪）",
                 True, all(l.get("request_id") not in (None, "-", "") for l in ok_complete))
        ev.check("日志 JSON 含 service/env 标识",
                 True, all(l.get("service") == "ai-gateway" for l in lines[:5]))

        ev.step("生成 HTML 可视化报告（内联 SVG）")
        llm_stats = after.get("llm_latency")
        html = _html_report(counters, llm_stats, records, res_samples)
        HTML_REPORT.write_text(html)
        ev.check("HTML 报告已生成", True, HTML_REPORT.exists(),
                 detail=str(HTML_REPORT))
        ev.check("报告包含 SVG 图表", True, "<svg" in html)
        ev.check("报告包含指标计数表", True, "requests_total" in html)

    return 0 if ev.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
