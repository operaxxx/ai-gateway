"""六大功能模块验证总控 runner。

顺序执行 6 个验证脚本（每个脚本独立拉起 mock 上游 + 网关，互不干扰），
汇总 reports/*.json 生成 summary.json 与 summary.md。
任一模块非 PASS 时进程退出码为 1。

运行:
  uv run python scripts/verification/run_all.py
可选: uv run python scripts/verification/run_all.py streaming retry
  （只跑指定模块）
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REPORT_DIR = SCRIPT_DIR / "reports"

# 模块名 -> (脚本文件名, 中文说明)
MODULES = [
    ("streaming", "verify_streaming.py", "流式输出（时间戳/块数/接收间隔）"),
    ("structured_output", "verify_structured_output.py", "结构化输出（字段完整性/类型/格式 + 反例）"),
    ("prompt_template", "verify_prompt_template.py", "模板引用（变量替换/条件渲染/版本管理）"),
    ("observability", "verify_observability.py", "可观测数据（指标收集 + 可视化报告）"),
    ("retry", "verify_retry.py", "重试机制（故障注入/重试次数/恢复时间）"),
    ("rate_limit", "verify_rate_limit.py", "限流（阈值触发/429 策略/并发压力/窗口恢复）"),
]


def run_module(name: str, script: Path) -> dict:
    """运行单个验证脚本并读取其报告。脚本崩溃（无报告）时标记 ERROR。"""
    print(f"\n{'=' * 60}\n>>> [{name}] {script.name}\n{'=' * 60}")
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = round(time.monotonic() - t0, 1)

    report_path = REPORT_DIR / f"{name}.json"
    if proc.returncode == 0 and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))

    # 脚本未产出报告：构造 ERROR 占位记录（详细输出已流式打印，日志在 reports/）
    return {
        "module": name, "description": script.name,
        "duration_s": elapsed,
        "summary": {"total": 0, "passed": 0, "failed": 0, "status": "ERROR"},
        "checks": [], "steps": [], "evidence": {},
        "error": f"exit={proc.returncode}, 报告缺失或执行失败"
                 f"（日志: reports/{name}.log / {name}.proc）",
    }


def write_summary(results: list[dict]) -> tuple[bool, Path, Path]:
    """汇总为 summary.json + summary.md，返回 (是否全 PASS, json 路径, md 路径)。"""
    all_pass = all(r["summary"]["status"] == "PASS" for r in results)
    total_checks = sum(r["summary"]["total"] for r in results)
    passed_checks = sum(r["summary"]["passed"] for r in results)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "generated_at": now,
        "all_pass": all_pass,
        "modules_total": len(results),
        "modules_pass": sum(1 for r in results if r["summary"]["status"] == "PASS"),
        "checks_total": total_checks,
        "checks_passed": passed_checks,
        "total_duration_s": round(sum(r.get("duration_s", 0) for r in results), 1),
        "modules": results,
    }
    json_path = REPORT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}
    lines = [
        "# AI Gateway 六大功能模块验证报告",
        "",
        f"- 生成时间: {now}",
        f"- 结论: {'全部通过' if all_pass else '存在失败项'}"
        f"（模块 {summary['modules_pass']}/{summary['modules_total']}，"
        f"检查项 {passed_checks}/{total_checks}）",
        "",
        "| # | 模块 | 说明 | 检查项 | 结论 | 耗时 |",
        "|---|------|------|--------|------|------|",
    ]
    for i, r in enumerate(results, 1):
        s = r["summary"]
        lines.append(
            f"| {i} | {r['module']} | {r['description']} "
            f"| {s['passed']}/{s['total']} | {icon.get(s['status'], '?')} {s['status']} "
            f"| {r.get('duration_s', '-')}s |"
        )
    lines += ["", "详细证据见 reports/<module>.json 与 reports/observability_report.html"]
    md_path = REPORT_DIR / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_pass, json_path, md_path


def main() -> int:
    REPORT_DIR.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    selected = [(n, SCRIPT_DIR / f) for n, f, _ in MODULES if not only or n in only]
    if not selected:
        print(f"未知模块: {only}，可用: {[n for n, _, _ in MODULES]}")
        return 2

    print(f"AI Gateway 六大功能验证 · 共 {len(selected)} 个模块")
    t0 = time.monotonic()
    results = [run_module(n, p) for n, p in selected]
    all_pass, json_path, md_path = write_summary(results)

    print(f"\n{'=' * 60}\n总控汇总（总耗时 {round(time.monotonic() - t0, 1)}s）")
    for r in results:
        s = r["summary"]
        print(f"  [{'PASS' if s['status'] == 'PASS' else s['status']:>5}] "
              f"{r['module']:<20} {s['passed']}/{s['total']}  {r['description']}")
    print(f"{'=' * 60}")
    print(f"结论: {'✅ 六大功能全部验证通过' if all_pass else '❌ 存在未通过项'}")
    print(f"汇总: {json_path}\n      {md_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
