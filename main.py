"""演示：同一个统一事件流（StreamEvent），两个模型、两种上游协议。

前端只需要实现一个解析器，处理 4 种事件：start / delta / done / error。

运行:
    export OPENAI_API_KEY="你的key"
    export ANTHROPIC_API_KEY="你的key"
    uv run python main.py
"""

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


def demo_stream(gw: Gateway, model: str):
    """消费统一事件流：模拟前端的'事件状态机 + 渲染'。"""
    print(f"\n{'='*60}")
    print(f"模型: {model}（统一事件流）")
    print(f"{'='*60}")

    rendered = []   # 前端的"已渲染正文"
    thinking = []   # 前端的"思考过程面板"
    try:
        for ev in gw.stream(make_request(model)):
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
        return

    print(f"\n思考过程: {''.join(thinking)[:120]}...")
    print(f"渲染结果: {''.join(rendered)}")


def main():
    gw = Gateway()
    demo_stream(gw, "deepseek-v4-flash")   # Anthropic Messages 协议
    demo_stream(gw, "deepseek-v4-pro")     # OpenAI Responses 协议


if __name__ == "__main__":
    main()
