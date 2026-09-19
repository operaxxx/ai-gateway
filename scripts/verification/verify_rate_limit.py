"""验证模块 6：限流功能（多级：IP 滑动窗口 + 模型令牌桶）。

验证目标：
- 一级 IP 限流：滑动窗口在高并发/连续请求场景下按配置阈值精确触发，
  429 响应携带统一错误体（scope=ip）与 Retry-After，窗口滑动后自动恢复
- 二级模型限流：令牌桶按 burst 精确放行突发、耗尽后拒绝（scope=model），
  per-model 独立桶互不影响，令牌惰性填充后自动恢复

执行步骤:
  1. 启动 mock + 网关（RATE_LIMIT_RPM=6 / WINDOW_S=4s，重试关闭以免干扰计数）
  2. 阈值触发：连续 12 个请求 -> 前 6 个 200、后 6 个 429
  3. 429 响应体校验：error.type=rate_limit_error、scope=ip、retryable=true、Retry-After 头有效
  4. 限流期间 /v1/metrics 可正常抓取（监控不被限流）且 rate_limited_total 精确计数
  5. 并发压力：12 线程并发请求 -> 放行数 <= 阈值，其余 429（策略在高并发下仍精确）
  6. 恢复机制：等待窗口滑动（4s）-> 请求恢复 200
  7. 重启网关（IP 限流禁用，MODEL_RATE_LIMITS 配置令牌桶）
  8. 突发触发：连续 12 个请求 -> 前 burst(3) 个 200、其余 429（scope=model）
  9. per-model 隔离：flash 桶耗尽后 pro 独立桶不受影响
  10. 分级计数：model_rate_limited_total 精确、rate_limited_total 保持 0
  11. 恢复机制：等待令牌填充（0.2 token/s）-> 请求恢复 200

运行:
  uv run python scripts/verification/verify_rate_limit.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import (Evidence, chat, gateway_metrics, start_gateway, start_mock)

MODEL = "deepseek-v4-flash"
LIMIT = 6
WINDOW_S = 4
BURST = 12
# 模型级限流参数：rpm=12 -> 0.2 token/s 填充速率（连续请求期间攒不够 1 个，
# 保证放行数精确 = burst）；flash burst=3，pro burst=1
MODEL_FLASH_RPM = 12
MODEL_FLASH_BURST = 3
MODEL_LIMITS = {
    "deepseek-v4-flash": {"rpm": MODEL_FLASH_RPM, "burst": MODEL_FLASH_BURST},
    "deepseek-v4-pro": {"rpm": MODEL_FLASH_RPM, "burst": 1},
}


def _fire(base: str) -> dict:
    t0 = time.monotonic()
    r = chat(base, {"model": MODEL, "stream": False,
                    "messages": [{"role": "user", "content": "hi"}]}, timeout_s=30)
    body = {}
    try:
        body = r.json()
    except Exception:
        pass
    return {"status": r.status_code, "headers": dict(r.headers), "body": body,
            "ms": round((time.monotonic() - t0) * 1000, 1)}


def main() -> int:
    ev = Evidence("rate_limit", "限流：阈值触发 / 429 策略 / 并发压力 / 窗口恢复")

    ev.step(f"启动 mock 上游 + 网关（RATE_LIMIT_RPM={LIMIT}, WINDOW_S={WINDOW_S}s, 重试关闭）")
    mock_cm, mock_port = start_mock()
    gw_cm, gw_port = start_gateway(mock_port=mock_port, extra_env={
        "RATE_LIMIT_RPM": str(LIMIT),
        "RATE_LIMIT_WINDOW_S": str(WINDOW_S),
        "RETRY_MAX_ATTEMPTS": "1",   # 关闭重试，避免 429 触发上游重试干扰计数
    })
    with mock_cm, gw_cm:
        base = f"http://127.0.0.1:{gw_port}"

        # ---- 阈值触发 ----
        ev.step(f"连续发送 {BURST} 个请求（阈值 {LIMIT}）-> 前 {LIMIT} 个放行，其余拒绝")
        results = [_fire(base) for _ in range(BURST)]
        statuses = [r["status"] for r in results]
        first_reject = statuses.index(429) + 1 if 429 in statuses else -1
        allowed = statuses[:first_reject - 1] if first_reject > 0 else statuses   # 首个 429 不计入放行
        rejected = [r for r in results if r["status"] == 429]
        ev.evidence("sequential_results",
                    [{"status": r["status"], "ms": r["ms"]} for r in results])
        ev.check(f"前 {LIMIT} 个请求全部 200（阈值内放行）", True,
                 len(allowed) == LIMIT and all(s == 200 for s in allowed),
                 detail=f"状态序列={statuses}")
        ev.check(f"第 {LIMIT + 1} 个起全部 429（精确触发）",
                 True, first_reject == LIMIT + 1 and len(rejected) == BURST - LIMIT,
                 detail=f"首个 429 位于第 {first_reject} 个")
        ev.check("被拒请求不占窗口名额（放行数恰好 = 阈值）", LIMIT, len(allowed))

        # ---- 429 响应策略 ----
        ev.step("校验 429 响应：统一错误体 + Retry-After 头")
        sample = rejected[0]
        body = sample["body"].get("error", {})
        retry_after = sample["headers"].get("retry-after")
        ev.check("error.type = rate_limit_error", "rate_limit_error", body.get("type"))
        ev.check("error.scope = ip（一级限流标识）", "ip", body.get("scope"))
        ev.check("error.retryable = true（建议调用方稍后重试）", True, body.get("retryable") is True)
        ev.check("Retry-After 头存在且为 [1, 窗口] 内整数",
                 True, retry_after is not None and retry_after.isdigit()
                 and 1 <= int(retry_after) <= WINDOW_S,
                 detail=f"Retry-After={retry_after}")
        ev.evidence("rejected_response_sample",
                    {"status": sample["status"], "body": sample["body"],
                     "retry_after": retry_after})

        # ---- 限流期间监控可用 + 指标精确 ----
        ev.step("限流期间抓取 /v1/metrics（监控端点不受限流）")
        m = gateway_metrics(gw_port)
        ev.check("metrics 抓取成功且 rate_limited_total == 6", LIMIT,
                 m["counters"]["rate_limited_total"])
        ev.check("status_counts 含 429:6", 6, m["status_counts"].get("429", 0))
        ev.check("429 计入 errors_total（错误率口径完整）", LIMIT, m["counters"]["errors_total"])

        # ---- 并发压力 ----
        ev.step(f"等待窗口滑动 {WINDOW_S}s 后进行 {BURST} 线程并发压力测试")
        time.sleep(WINDOW_S + 0.3)
        with ThreadPoolExecutor(max_workers=BURST) as pool:
            concurrent_results = list(pool.map(lambda _: _fire(base), range(BURST)))
        c_statuses = [r["status"] for r in concurrent_results]
        c_allowed = sum(1 for s in c_statuses if s == 200)
        c_rejected = sum(1 for s in c_statuses if s == 429)
        ev.evidence("concurrent_results", [{"status": r["status"], "ms": r["ms"]}
                                           for r in concurrent_results])
        ev.check(f"并发放行数 <= 阈值（{LIMIT}）", True, c_allowed <= LIMIT,
                 detail=f"放行 {c_allowed} / 拒绝 {c_rejected} / 总 {BURST}")
        ev.check("并发放行 + 拒绝 == 总请求数（无丢失）", BURST, c_allowed + c_rejected)
        ev.check("并发下被拒请求全部拿到 429（策略执行无例外）", True, c_rejected == BURST - c_allowed)

        # ---- 恢复机制 ----
        ev.step(f"等待窗口滑动 {WINDOW_S + 0.5}s -> 验证限流自动恢复")
        t0 = time.monotonic()   # 恢复耗时 = 等待窗口滑动 + 首个成功请求，自发起总计时
        time.sleep(WINDOW_S + 0.5)
        r = _fire(base)
        recovery_s = round(time.monotonic() - t0, 1)
        ev.evidence("recovery", {"status": r["status"], "recovery_s": recovery_s})
        ev.check("窗口滑动后请求恢复 200（无需人工干预）", 200, r["status"])
        ev.check("恢复耗时符合预期（≈ 等待窗口时长）", True, 4.0 <= recovery_s <= 8.0,
                 detail=f"recovery_s={recovery_s}")

    # ---- 模型级限流（二级：令牌桶）----
    # IP 限流禁用（RATE_LIMIT_RPM 不设），两级互不干扰，单独验证模型级行为
    ev.step(f"重启网关（IP 限流禁用，MODEL_RATE_LIMITS 令牌桶："
            f"flash rpm={MODEL_FLASH_RPM}/burst={MODEL_FLASH_BURST}，pro burst=1）")
    ev.evidence("model_config", MODEL_LIMITS)
    mock_cm2, mock_port2 = start_mock()
    gw_cm2, gw_port2 = start_gateway(mock_port=mock_port2, extra_env={
        "MODEL_RATE_LIMITS": json.dumps(MODEL_LIMITS),
        "RETRY_MAX_ATTEMPTS": "1",   # 关闭重试，避免干扰计数
    })
    with mock_cm2, gw_cm2:
        base2 = f"http://127.0.0.1:{gw_port2}"

        # ---- 突发触发：初始满桶，前 burst 个放行，其余拒绝 ----
        ev.step(f"连续 {BURST} 个请求（flash burst={MODEL_FLASH_BURST}）"
                f"-> 前 {MODEL_FLASH_BURST} 个 200、其余 429")
        results2 = [_fire(base2) for _ in range(BURST)]
        statuses2 = [r["status"] for r in results2]
        allowed2 = sum(1 for s in statuses2 if s == 200)
        rejected2 = [r for r in results2 if r["status"] == 429]
        ev.evidence("model_sequential_results",
                    [{"status": r["status"], "ms": r["ms"]} for r in results2])
        ev.check(f"突发上限精确：放行数 == burst({MODEL_FLASH_BURST})",
                 MODEL_FLASH_BURST, allowed2)
        ev.check("桶耗尽后其余全部 429", BURST - MODEL_FLASH_BURST, len(rejected2))

        # ---- 429 响应策略：scope=model ----
        ev.step("校验模型级 429 响应：scope=model + Retry-After")
        sample2 = rejected2[0]
        body2 = sample2["body"].get("error", {})
        retry_after2 = sample2["headers"].get("retry-after")
        ev.check("error.type = rate_limit_error", "rate_limit_error", body2.get("type"))
        ev.check("error.scope = model（区分于 IP 级 scope=ip）", "model", body2.get("scope"))
        ev.check("error.model = deepseek-v4-flash", MODEL, body2.get("model"))
        ev.check("error.retryable = true", True, body2.get("retryable") is True)
        ev.check("Retry-After 头存在且为正整数",
                 True, retry_after2 is not None and retry_after2.isdigit()
                 and int(retry_after2) >= 1,
                 detail=f"Retry-After={retry_after2}")
        ev.evidence("model_rejected_response_sample",
                    {"status": sample2["status"], "body": sample2["body"],
                     "retry_after": retry_after2})

        # ---- per-model 隔离：flash 桶耗尽，pro 独立桶不受影响 ----
        ev.step("切换 model=deepseek-v4-pro（burst=1 独立桶）验证 per-model 隔离")
        r_pro = chat(base2, {"model": "deepseek-v4-pro", "stream": False,
                             "messages": [{"role": "user", "content": "hi"}]}, timeout_s=30)
        ev.check("flash 桶耗尽后 pro 首请求仍 200（独立桶互不影响）", 200, r_pro.status_code)

        # ---- 分级计数：模型级精确，IP 级保持 0 ----
        ev.step("抓取 /v1/metrics 验证分级计数器")
        m2 = gateway_metrics(gw_port2)
        ev.check(f"model_rate_limited_total == {len(rejected2)}（模型级拒绝数）",
                 len(rejected2), m2["counters"]["model_rate_limited_total"])
        ev.check("rate_limited_total == 0（IP 级禁用未触发）", 0,
                 m2["counters"]["rate_limited_total"])
        ev.check("status_counts 429 计数与模型级拒绝数一致", len(rejected2),
                 m2["status_counts"].get("429", 0))

        # ---- 令牌桶恢复：惰性填充（0.2 token/s），攒够 1 个即放行 ----
        ev.step("等待 5.5s 令牌桶惰性填充（0.2 token/s 攒 1 个需 5s）-> 验证恢复")
        t02 = time.monotonic()
        time.sleep(5.5)
        r_rec = _fire(base2)
        recovery2_s = round(time.monotonic() - t02, 1)
        ev.evidence("model_recovery", {"status": r_rec["status"], "recovery_s": recovery2_s})
        ev.check("令牌攒够后请求恢复 200（无需人工干预）", 200, r_rec["status"])

    return 0 if ev.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
