"""诊断脚本：dump DeepSeek responses 流的真实事件名和数据样本。

用途：当适配器的 delta 过滤不匹配真实上游时，用这个看线上字节，不猜协议。
运行（在项目根目录）: uv run python -m scripts.debug_dump_sse
"""

from gateway.responses_adapter import ResponsesAdapter
from gateway.types import ChatRequest, Message

a = ResponsesAdapter()
payload = a._build_payload(ChatRequest(
    model="deepseek-v4-pro",
    messages=[Message(role="user", content="用一句话解释什么是API网关")],
    max_tokens=60,
))
payload["stream"] = True

seen: dict[str, str] = {}  # 事件名 -> 首个 data 样本
total_events = 0
with a.client.stream(
    "POST", a.api_url,
    headers={"Authorization": f"Bearer {a.api_key}"},
    json=payload,
) as r:
    print("status:", r.status_code)
    print("content-type:", r.headers.get("content-type"))
    ev = None
    for line in r.iter_lines():
        if line.startswith("event:"):
            ev = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            total_events += 1
            if ev not in seen:
                seen[ev] = data[:250]

print("URL:", a.api_url)
print(f"事件总数: {total_events}")
print("每个事件名 -> 首个 data 样本:")
for k, v in seen.items():
    print(f"  {k}: {v}")
