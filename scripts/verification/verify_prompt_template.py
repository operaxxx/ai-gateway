"""验证模块 3：模板引用（Prompt 模板机制）。

验证目标：模板加载、参数替换、条件渲染、循环渲染、版本引用与 chat 内联引用全链路有效。

执行步骤:
  1. 启动 mock 上游 + 网关（mock 会把收到的 system prompt 回显到正文，作为证据）
  2. 创建含 变量占位 + {% if %} 条件 + {% for %} 循环 的模板 -> 校验变量提取
  3. 渲染预览：参数替换正确、条件两个分支各自生效、循环逐项渲染
  4. 错误路径：缺变量 400（含 missing 明细）、语法错误 400
  5. 版本机制：追加 v2，v1 内容不可变；按版本渲染 v1 / latest 各自正确
  6. chat 内联引用：prompt.id+variables 渲染为 system 消息（mock 回显核对）；
     版本钉死引用（version=1）实际生效；与自带 system 消息互斥 400

预期结果: 每步与预期一致；任何一条不满足即 FAIL。

运行:
  uv run python scripts/verification/verify_prompt_template.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Evidence, chat, mock_control, start_gateway, start_mock

import httpx

MODEL = "deepseek-v4-flash"

TEMPLATE_V1 = (
    "你是{{role}}助手。\n"
    "{% if formal %}请使用敬语。{% else %}轻松一点。{% endif %}\n"
    "写作要点：\n"
    "{% for rule in rules %}- {{rule}}\n{% endfor %}"
)
TEMPLATE_V2 = "你是{{role}}助手。（第二版）\n{% if formal %}敬语。{% else %}随意。{% endif %}"


def main() -> int:
    ev = Evidence("prompt_template", "模板引用：加载/参数替换/条件渲染/循环/版本/chat 引用")

    ev.step("启动 mock 上游 + 网关")
    mock_cm, mock_port = start_mock()
    gw_cm, gw_port = start_gateway(mock_port=mock_port)
    with mock_cm, gw_cm:
        base = f"http://127.0.0.1:{gw_port}"
        mock_control(mock_port, sse_chunk_delay_ms=0)

        # ---- 模板创建与变量提取 ----
        ev.step("创建模板（变量 + 条件 + 循环），校验变量自动提取")
        r = httpx.post(f"{base}/v1/prompts", json={
            "id": "style-guide", "name": "写作风格", "description": "验证模板",
            "content": TEMPLATE_V1,
        }, timeout=30)
        ev.check("创建成功 201", 201, r.status_code)
        created = r.json()
        ev.check("变量提取完整且按出现顺序 [role, formal, rules]",
                 ["role", "formal", "rules"], created.get("variables"))
        ev.check("初始版本号为 1", 1, created.get("version"))

        # ---- 渲染预览：参数替换 ----
        ev.step("渲染预览：参数替换 + 条件分支(true) + 循环")
        r = httpx.post(f"{base}/v1/prompts/style-guide/render", json={
            "version": 1,
            "variables": {"role": "翻译", "formal": True, "rules": ["简洁", "准确"]},
        }, timeout=30)
        ev.check("渲染 200", 200, r.status_code)
        rendered = r.json().get("rendered", "")
        ev.evidence("rendered_formal", rendered)
        ev.check("参数替换生效（含『翻译』）", True, "翻译" in rendered)
        ev.check("条件分支 formal=true 生效（含『请使用敬语』）", True, "请使用敬语" in rendered)
        ev.check("条件另一分支未渲染（不含『轻松一点』）", True, "轻松一点" not in rendered)
        ev.check("循环逐项渲染（含 - 简洁 / - 准确）", True,
                 "- 简洁" in rendered and "- 准确" in rendered)

        ev.step("渲染预览：条件分支(false)")
        r = httpx.post(f"{base}/v1/prompts/style-guide/render", json={
            "variables": {"role": "翻译", "formal": False, "rules": []},
        }, timeout=30)
        rendered_false = r.json().get("rendered", "")
        ev.check("formal=false 走 else 分支（含『轻松一点』）", True, "轻松一点" in rendered_false)
        ev.check("空列表循环零项（不含『- 』）", True, "- " not in rendered_false)
        ev.evidence("rendered_casual", rendered_false)

        # ---- 错误路径 ----
        ev.step("错误路径：缺变量 -> 400 + missing 明细")
        r = httpx.post(f"{base}/v1/prompts/style-guide/render", json={
            "variables": {"role": "翻译"},
        }, timeout=30)
        ev.check("HTTP 400", 400, r.status_code)
        body = r.json().get("detail", {})
        missing = body.get("missing", []) if isinstance(body, dict) else []
        ev.check("error=missing_variables", "missing_variables", body.get("error"))
        # 语义约定：StrictUndefined 逐个报错，missing 指明渲染期遇到的第一个缺失变量
        ev.check("missing 指明第一个缺失变量（formal 先于 rules 被渲染）",
                 True, "formal" in missing,
                 detail=f"missing={missing}")

        ev.step("错误路径：模板语法错误 -> 400")
        r = httpx.post(f"{base}/v1/prompts", json={
            "id": "bad-syntax", "name": "坏模板", "content": "{% if %}",
        }, timeout=30)
        ev.check("HTTP 400", 400, r.status_code)
        ev.check("报错为模板语法错误", True, "模板语法错误" in r.text)

        # ---- 版本机制 ----
        ev.step("版本机制：追加 v2，v1 不可变，按版本渲染")
        r = httpx.post(f"{base}/v1/prompts/style-guide/versions",
                       json={"content": TEMPLATE_V2}, timeout=30)
        ev.check("追加版本 201 且版本号为 2", (201, 2), (r.status_code, r.json().get("version")))
        r = httpx.get(f"{base}/v1/prompts/style-guide/versions/1", timeout=30)
        ev.check("v1 内容不可变（仍为第一版）", True, TEMPLATE_V1 == r.json().get("content"))
        r = httpx.post(f"{base}/v1/prompts/style-guide/render", json={
            "version": 1, "variables": {"role": "X", "formal": True, "rules": []},
        }, timeout=30)
        ev.check("按 version=1 渲染得到 v1 内容", True, "请使用敬语" in r.json().get("rendered", ""))
        r = httpx.post(f"{base}/v1/prompts/style-guide/render", json={
            "version": "latest", "variables": {"role": "X", "formal": True},
        }, timeout=30)
        ev.check("latest 渲染得到 v2 内容", True, "敬语。" in r.json().get("rendered", "")
                 and "请使用" not in r.json().get("rendered", ""))
        r = httpx.get(f"{base}/v1/prompts/style-guide/versions/999", timeout=30)
        ev.check("不存在版本 -> 404", 404, r.status_code)

        # ---- chat 内联引用 ----
        ev.step("chat 内联引用：prompt+variables 渲染为 system（mock 回显核对）")
        payload = {
            "model": MODEL, "stream": False,
            "prompt": {"id": "style-guide", "version": 1,
                       "variables": {"role": "翻译", "formal": True, "rules": ["简洁", "准确"]}},
            "messages": [{"role": "user", "content": "开始"}],
        }
        r = chat(base, payload)
        ev.check("HTTP 200", 200, r.status_code)
        text = r.json().get("text", "")
        ev.evidence("chat_text_with_prompt", text)
        ev.check("上游收到的 system == v1 渲染结果（mock 回显含敬语分支与循环项）",
                 True, "请使用敬语" in text and "- 简洁" in text and "翻译" in text,
                 detail=f"text={text[:80]}…")

        ev.step("chat 引用版本钉死：version=1 时上游收到 v1（非 latest）")
        payload["prompt"]["id"] = "style-guide"
        r = chat(base, payload)
        ev.check("再次调用上游回显仍为 v1 内容", True, "请使用敬语" in r.json().get("text", ""))

        ev.step("互斥校验：prompt 引用与自带 system 消息不可同用 -> 400")
        payload["messages"] = [{"role": "system", "content": "x"},
                               {"role": "user", "content": "y"}]
        r = chat(base, payload)
        ev.check("HTTP 400", 400, r.status_code)

        ev.step("引用不存在的模板 -> 404")
        payload["messages"] = [{"role": "user", "content": "y"}]
        payload["prompt"] = {"id": "no-such-prompt", "variables": {}}
        r = chat(base, payload)
        ev.check("HTTP 404", 404, r.status_code)

    return 0 if ev.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
