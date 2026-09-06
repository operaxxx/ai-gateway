import json

from gateway.types import ChatRequest, Message
from gateway.gateway import Gateway


def run_gateway(gw: Gateway, model: str):
    req = ChatRequest(
        model=model,
        messages=[
            Message(role="system", content="你是一个简洁的助手"),
            Message(role="user", content="用一句话解释什么是API网关"),
        ],
        max_tokens=100,
    )
    try:
        r = gw.complete(req)
        print(f"回复: {r.text}")
        print(f"统一用量: {r.usage}")
        print(f"统一 stop_reason: {r.stop_reason}")
        print("--- 原始响应 ---")
        print(json.dumps(r.raw, ensure_ascii=False, indent=2)[:1000])
    except Exception as e:
        print(f"调用失败: {e}")


def main():
    gw = Gateway()

    print("=" * 60)
    print("【模型 1：deepseek-v4-flash → Anthropic Messages 协议】")
    print("=" * 60)
    run_gateway(gw, "deepseek-v4-flash")

    print()
    print("=" * 60)
    print("【模型 2：deepseek-v4-pro → OpenAI Responses API 协议】")
    print("=" * 60)
    run_gateway(gw, "deepseek-v4-pro")


if __name__ == "__main__":
    main()
