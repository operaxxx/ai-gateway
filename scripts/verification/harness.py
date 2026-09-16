"""验证脚本共享工具：进程编排、SSE 解析、证据记录与报告输出。

所有 verify_*.py 脚本通过本模块获得一致的结构：
- 每个脚本独立拉起自己的 mock 上游 + 网关服务（隔离环境变量，互不干扰）
- Evidence 类记录 执行步骤 / 预期结果 / 实际结果 / 自动判定(PASS|FAIL)
- 报告 JSON 落盘 scripts/verification/reports/<module>.json，run_all.py 汇总
"""

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
REPORT_DIR = SCRIPT_DIR / "reports"
GATEWAY_PORT = 8901
MOCK_PORT = 8902


# ---------- 进程编排 ----------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(url: str, timeout_s: float = 20.0) -> None:
    """轮询直到 HTTP 200（服务就绪），超时抛 RuntimeError。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"服务未就绪: {url}")


@contextlib.contextmanager
def service(cmd: list[str], env_extra: dict[str, str], ready_url: str, log_name: str):
    """启动子进程服务，就绪后交给调用方，退出时优雅终止。"""
    env = dict(os.environ)
    env.update(env_extra)
    REPORT_DIR.mkdir(exist_ok=True)
    log_path = REPORT_DIR / log_name
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(ready_url)
            yield proc
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def start_mock(port: int | None = None):
    """启动 mock 上游服务。返回 (context manager, port)。"""
    port = port or free_port()
    cm = service(
        [sys.executable, str(SCRIPT_DIR / "mock_upstream.py"), "--port", str(port)],
        env_extra={"MOCK_PORT": str(port)},
        ready_url=f"http://127.0.0.1:{port}/_health",
        log_name="mock_upstream.log",
    )
    return cm, port


def start_gateway(port: int | None = None, mock_port: int = MOCK_PORT,
                  extra_env: dict[str, str] | None = None, log_name: str = "gateway.log"):
    """启动网关服务（上游指向 mock）。返回 (context manager, port)。

    注意：进程 stdout/stderr 捕获到 <log_name>.proc（uvicorn 启动信息等），
    应用 JSON 日志经 LOG_FILE 写 <log_name>，二者分离避免双句柄交错撕裂行。
    应用日志文件按追加模式打开，启动前先删除旧文件保证本轮日志隔离。
    """
    port = port or free_port()
    log_path = REPORT_DIR / log_name
    log_path.unlink(missing_ok=True)
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{mock_port}",
        "ANTHROPIC_API_PATH": "/v1/messages",
        "OPENAI_BASE_URL": f"http://127.0.0.1:{mock_port}",
        "RESPONSES_API_PATH": "/v1/responses",
        "ANTHROPIC_API_KEY": "mock-key",
        "OPENAI_API_KEY": "mock-key",
        "PROMPTS_DB_PATH": str(REPORT_DIR / f"prompts_{port}.db"),
        "LOG_LEVEL": "INFO",
        "LOG_FILE": str(REPORT_DIR / log_name),
        "LOG_ENV": "verification",
    }
    env.update(extra_env or {})
    cm = service(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        env_extra=env,
        ready_url=f"http://127.0.0.1:{port}/health",
        log_name=f"{log_name}.proc",
    )
    return cm, port


def mock_control(mock_port: int, **kwargs) -> dict:
    """向 mock 上游注入故障策略。"""
    r = httpx.post(f"http://127.0.0.1:{mock_port}/_control", json=kwargs, timeout=10)
    r.raise_for_status()
    return r.json()


def mock_reset(mock_port: int) -> None:
    httpx.post(f"http://127.0.0.1:{mock_port}/_reset", timeout=10)


def mock_counters(mock_port: int) -> dict:
    r = httpx.get(f"http://127.0.0.1:{mock_port}/_counters", timeout=10)
    r.raise_for_status()
    return r.json()


def gateway_metrics(port: int) -> dict:
    r = httpx.get(f"http://127.0.0.1:{port}/v1/metrics", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------- HTTP / SSE 调用 ----------

def chat(base_url: str, payload: dict, timeout_s: float = 120.0) -> httpx.Response:
    """非流式聊天请求。"""
    return httpx.post(f"{base_url}/v1/chat", json=payload, timeout=timeout_s)


def chat_sse(base_url: str, payload: dict, timeout_s: float = 120.0) -> dict:
    """流式聊天请求：逐事件记录接收时间戳。

    Returns:
        {"status": int, "headers": dict, "events": [
            {"event": "start|delta|done|error", "data": dict,
             "t_ms": 距响应首字节的毫秒数, "gap_ms": 距上一事件的毫秒数}
        ], "first_byte_ms": 响应首字节耗时}
    """
    events: list[dict] = []
    with httpx.stream("POST", f"{base_url}/v1/chat", json=payload,
                      timeout=timeout_s) as resp:
        headers = dict(resp.headers)
        first_byte_ms: float | None = None
        t_prev = None
        t0 = time.monotonic()
        event_name = ""
        data_lines: list[str] = []
        for line in resp.iter_lines():
            now = time.monotonic()
            if first_byte_ms is None:
                first_byte_ms = (now - t0) * 1000
            if line == "":
                if data_lines:
                    try:
                        data = json.loads("\n".join(data_lines))
                    except json.JSONDecodeError:
                        data = {"_raw": "\n".join(data_lines)}
                    gap = 0.0 if t_prev is None else (now - t_prev) * 1000
                    events.append({
                        "event": event_name or "message", "data": data,
                        "t_ms": round((now - t0) * 1000, 1), "gap_ms": round(gap, 1),
                    })
                    t_prev = now
                event_name = ""
                data_lines = []
            elif line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
    return {"status": resp.status_code, "headers": headers,
            "events": events, "first_byte_ms": round(first_byte_ms or 0.0, 1)}


# ---------- 资源采样 ----------

def process_stats(pid: int) -> dict:
    """macOS/Linux 通用：ps 采集进程 CPU% 与 RSS。"""
    try:
        out = subprocess.run(
            ["ps", "-o", "%cpu=,rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        cpu_s, rss_s = out.split()
        return {"cpu_percent": float(cpu_s), "rss_mb": round(int(rss_s) / 1024, 1)}
    except Exception:
        return {"cpu_percent": None, "rss_mb": None}


# ---------- 证据记录与报告 ----------

class Evidence:
    """结构化验证证据：步骤、检查项（预期 vs 实际）、原始证据数据。

    check() 自动判定 PASS/FAIL 并打印；finish() 落盘报告并返回整体结论。
    """

    def __init__(self, module: str, description: str):
        self.module = module
        self.description = description
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.steps: list[dict] = []
        self.checks: list[dict] = []
        self.data: dict = {}
        self._t0 = time.monotonic()

    def step(self, action: str, detail: str | None = None) -> None:
        entry = {"n": len(self.steps) + 1, "action": action}
        if detail:
            entry["detail"] = detail
        self.steps.append(entry)
        print(f"  步骤 {entry['n']}: {action}" + (f" — {detail}" if detail else ""))

    def check(self, name: str, expected, actual, passed: bool | None = None,
              detail: str | None = None) -> bool:
        """记录一项检查；passed 缺省时按 expected == actual 自动判定。"""
        if passed is None:
            passed = expected == actual
        entry = {
            "name": name,
            "expected": _short(expected),
            "actual": _short(actual),
            "passed": bool(passed),
        }
        if detail:
            entry["detail"] = detail
        self.checks.append(entry)
        mark = "PASS" if passed else "FAIL"
        print(f"    [{mark}] {name}: 预期={_short(expected)} 实际={_short(actual)}")
        return passed

    def evidence(self, key: str, value) -> None:
        self.data[key] = value

    def finish(self) -> bool:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        ok = total > 0 and passed == total
        report = {
            "module": self.module,
            "description": self.description,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(time.monotonic() - self._t0, 1),
            "steps": self.steps,
            "checks": self.checks,
            "evidence": self.data,
            "summary": {"total": total, "passed": passed, "failed": total - passed,
                        "status": "PASS" if ok else "FAIL"},
        }
        REPORT_DIR.mkdir(exist_ok=True)
        out = REPORT_DIR / f"{self.module}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n{'='*60}")
        print(f"[{self.module}] 结论: {'PASS' if ok else 'FAIL'} "
              f"({passed}/{total} 项检查通过)  报告: {out}")
        print(f"{'='*60}")
        return ok


def _short(v, limit: int = 120) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]
