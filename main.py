"""演示：同一个统一事件流（StreamEvent），两个模型、两种上游协议；
以及 Prompt 模板应用闭环——同一模板 v1/v2 版本渲染后对比生成效果。

前端只需要实现一个解析器，处理 4 种事件：start / delta / done / error。

运行:
    export OPENAI_API_KEY="你的key"
    export ANTHROPIC_API_KEY="你的key"
    uv run python main.py
"""

import os

from gateway.env import load_env
from gateway.prompt_render import render
from gateway.prompt_store import PromptAlreadyExistsError, SqlitePromptStore
from gateway.types import ChatRequest, Message
from gateway.gateway import Gateway


def make_request(model: str) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=[
            Message(role="system", content="你是一个简洁的助手"),
            Message(role="user", content="用一句话解释什么是API网关"),
        ],
        max_tokens=512,  # 推理模型（pro）要先"思考"再回答，预算给足
    )


def demo_stream(gw: Gateway, request: ChatRequest, label: str) -> str:
    """消费统一事件流：模拟前端的'事件状态机 + 渲染'。返回正文全文。"""
    print(f"\n{'='*60}")
    print(label)
    print(f"{'='*60}")

    rendered = []   # 前端的"已渲染正文"
    thinking = []   # 前端的"思考过程面板"
    try:
        for ev in gw.stream(request):
            if ev.type == "start":
                print("[start]  生成开始，UI 可以显示'生成中...'")
            elif ev.type == "delta":
                if ev.channel == "reasoning":
                    thinking.append(ev.text)   # 思考内容通常渲染到可折叠面板
                else:
                    rendered.append(ev.text)   # 渲染器只管追加正文
            elif ev.type == "done":
                print(f"[done]   usage={ev.usage}  stop_reason={ev.stop_reason}")
            elif ev.type == "error":
                print(f"[error]  {ev.error}")
    except Exception as e:
        print(f"调用失败: {e}")
        return ""

    full = "".join(rendered)
    if thinking:
        print(f"思考过程: {''.join(thinking)[:120]}...")
    print(f"渲染结果: {full}")
    return full


def ensure_translator(store: SqlitePromptStore) -> None:
    """幂等自举 demo 模板（与 server 共用同一个 prompts.db）：不存在才创建。

    v1：基础版；v2：精修版（加角色设定与风格约束）——用于演示版本对比。
    """
    try:
        store.create_prompt(
            "translator", "翻译", "中译英 demo 模板",
            "把{{text}}从中文翻译成{{lang}}，只输出译文。",
            ["text", "lang"],
        )
        store.add_version(
            "translator",
            "你是专业译者。把{{text}}从中文翻译成{{lang}}，只输出译文，"
            "保留原文的标点与语气。",
            ["text", "lang"],
        )
    except PromptAlreadyExistsError:
        pass   # 已存在（server 或上次 demo 建过），直接用


def demo_prompt_versions(gw: Gateway) -> None:
    """应用闭环演示：从存储读模板 → 渲染 → 流式调用，同一问题对比 v1/v2 效果。"""
    load_env()
    store = SqlitePromptStore(os.environ.get("PROMPTS_DB_PATH", "prompts.db"))
    ensure_translator(store)

    variables = {"text": "API 网关是微服务架构的统一入口。", "lang": "英文"}
    for version in (1, 2):
        ver = store.get_version("translator", version)
        system_text = render(ver.content, variables)
        request = ChatRequest(
            model="deepseek-v4-flash",
            messages=[
                Message(role="system", content=system_text),   # 渲染结果作为 system
                Message(role="user", content="请开始"),
            ],
            max_tokens=512,
        )
        label = f"Prompt 模板 translator@v{ver.version}\n--- system: {system_text}"
        demo_stream(gw, request, label)


def main():
    gw = Gateway()
    demo_stream(gw, make_request("deepseek-v4-flash"), "模型: deepseek-v4-flash（Anthropic Messages 协议）")
    demo_stream(gw, make_request("deepseek-v4-pro"), "模型: deepseek-v4-pro（OpenAI Responses 协议）")
    demo_prompt_versions(gw)   # Prompt 模板版本对比


if __name__ == "__main__":
    main()
