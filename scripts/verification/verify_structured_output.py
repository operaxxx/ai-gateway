"""验证模块 2：结构化输出（JSON Schema 约束）。

验证目标：输出数据严格符合预设 JSON Schema —— 字段完整性、数据类型准确性、格式正确性，
且失败路径（类型错误/缺字段）有明确可观测的报错证据。

执行步骤:
  1. 启动 mock 上游 + 网关
  2. 正例（非流式）：带嵌套 schema 的 response_format 请求 -> structured_output 逐项校验
  3. 正例（流式）：done 事件携带 validation 结论
  4. 反例 A（非流式）：上游返回类型错误 -> 422 + errors 明细
  5. 反例 B（流式）：上游返回缺字段 -> done.validation.ok=false + errors
  6. 独立校验器复核：gateway.structured_output.validate 对实际输出再验一遍

预期结果:
  - 正例：structured_output 与 schema 完全一致（required 全在、类型正确、无额外字段）
  - 流式正例：validation.ok=true 且 parsed 与非流式一致；delta 为 JSON 增量
  - 反例 A：HTTP 422，detail.errors 指明 field/type/message
  - 反例 B：done.validation.ok=false，errors 非空且含缺失字段路径

运行:
  uv run python scripts/verification/verify_structured_output.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
from harness import Evidence, chat, chat_sse, mock_control, start_gateway, start_mock

from gateway.structured_output import validate as schema_validate

FLASH = "deepseek-v4-flash"    # Anthropic tool_use 路径
PRO = "deepseek-v4-pro"        # Responses json_schema 路径

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "age": {"type": "integer", "minimum": 0},
        "tags": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["name", "age", "tags"],
    "additionalProperties": False,
}


def main() -> int:
    ev = Evidence("structured_output", "结构化输出：JSON Schema 字段/类型/格式校验")

    ev.step("启动 mock 上游 + 网关")
    mock_cm, mock_port = start_mock()
    gw_cm, gw_port = start_gateway(mock_port=mock_port)
    with mock_cm, gw_cm:
        base = f"http://127.0.0.1:{gw_port}"

        # ---- 正例 A：非流式（Anthropic tool_use 路径）----
        ev.step("正例A 非流式：flash + 嵌套 schema（上游 tool_use 模式）")
        mock_control(mock_port, structured_mode="valid")
        r = chat(base, {"model": FLASH, "stream": False, "response_format": SCHEMA,
                        "messages": [{"role": "user", "content": "给个结构化结果"}]})
        ev.check("HTTP 200", 200, r.status_code)
        body = r.json()
        obj = body.get("structured_output")
        ev.evidence("valid_response", body)

        ev.step("正例A 校验：字段完整性 / 类型准确性 / 格式正确性")
        ev.check("响应含 structured_output 对象", True, isinstance(obj, dict))
        if isinstance(obj, dict):
            ev.check("required 字段全部存在（name/age/tags）",
                     True, all(k in obj for k in ["name", "age", "tags"]))
            ev.check("name 类型为 string", "str", type(obj.get("name")).__name__)
            ev.check("age 类型为 integer（非 bool/float）", "int", type(obj.get("age")).__name__)
            ev.check("tags 为 string 数组", True,
                     isinstance(obj.get("tags"), list)
                     and all(isinstance(t, str) for t in obj["tags"]))
            ev.check("无额外字段（additionalProperties=false）",
                     True, set(obj) <= set(SCHEMA["properties"]))
            verdict = schema_validate(__import__("json").dumps(obj, ensure_ascii=False), SCHEMA)
            ev.check("独立 schema 校验器复核通过", True, verdict.ok,
                     detail=f"errors={verdict.errors}")
        ev.check("响应含 elapsed_ms（可观测耗时）", True, "elapsed_ms" in body)

        # ---- 正例 B：流式（done 事件携带校验结论）----
        ev.step("正例B 流式：stream + response_format 同开（delta 为 JSON 增量）")
        result = chat_sse(base, {"model": FLASH, "stream": True, "response_format": SCHEMA,
                                 "messages": [{"role": "user", "content": "结构化"}]})
        events = result["events"]
        deltas = [e for e in events if e["event"] == "delta"]
        done = next((e for e in events if e["event"] == "done"), None)
        ev.check("delta 数量 > 1（JSON 增量实时透传）", True, len(deltas) > 1,
                 detail=f"{len(deltas)} 个增量")
        if done:
            v = done["data"].get("validation") or {}
            ev.check("done.validation.ok = true", True, v.get("ok") is True)
            ev.check("validation.parsed 为完整对象（含全部 required 字段）",
                     True, isinstance(v.get("parsed"), dict)
                     and all(k in v["parsed"] for k in ["name", "age", "tags"]))
            ev.evidence("stream_validation", v)
            # 流式拼接结果与非流式一致
            import json as _json
            stream_text = "".join(e["data"].get("text", "") for e in deltas)
            if v.get("parsed") is not None and stream_text:
                ev.check("流式拼接 JSON 与 parsed 一致",
                         True, _json.loads(stream_text) == v["parsed"])
        else:
            ev.check("收到 done 事件", True, False)

        # ---- 正例 C：Responses 协议（json_schema 路径，覆盖 name 字段回归）----
        ev.step("正例C 非流式：pro 走 Responses json_schema 协议")
        r2 = chat(base, {"model": PRO, "stream": False, "response_format": SCHEMA,
                         "messages": [{"role": "user", "content": "结构化"}]})
        ev.check("HTTP 200", 200, r2.status_code)
        obj2 = r2.json().get("structured_output")
        ev.check("structured_output 符合 schema（responses 协议）", True,
                 isinstance(obj2, dict) and all(k in obj2 for k in ["name", "age", "tags"]))

        # ---- 反例 A：类型错误 -> 422 ----
        ev.step("反例A 非流式：上游返回类型错误（age: string）-> 期望 422")
        mock_control(mock_port, structured_mode="wrong_type")
        r3 = chat(base, {"model": FLASH, "stream": False, "response_format": SCHEMA,
                         "messages": [{"role": "user", "content": "结构化"}]})
        ev.evidence("invalid_response_422", {"status": r3.status_code, "body": r3.json()})
        ev.check("HTTP 422", 422, r3.status_code)
        detail = r3.json().get("detail", {})
        ev.check("错误码为 structured_output_validation_failed",
                 "structured_output_validation_failed", detail.get("error"))
        errs = detail.get("errors", [])
        ev.check("errors 明细含 field/type/message", True,
                 bool(errs) and all(k in errs[0] for k in ["field", "type", "message"]),
                 detail=f"errors={errs}")

        # ---- 反例 B：流式缺字段 -> done.validation.ok=false ----
        ev.step("反例B 流式：上游返回缺字段对象 -> done.validation.ok=false")
        mock_control(mock_port, structured_mode="missing_field")
        result2 = chat_sse(base, {"model": FLASH, "stream": True, "response_format": SCHEMA,
                                  "messages": [{"role": "user", "content": "结构化"}]})
        done2 = next((e for e in result2["events"] if e["event"] == "done"), None)
        if done2:
            v2 = done2["data"].get("validation") or {}
            ev.check("done.validation.ok = false", True, v2.get("ok") is False)
            ev.check("validation.errors 非空（含缺失字段路径）", True, bool(v2.get("errors")),
                     detail=f"errors={v2.get('errors')}")
            ev.evidence("stream_invalid_validation", v2)
        else:
            ev.check("收到 done 事件", True, False)

    return 0 if ev.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
