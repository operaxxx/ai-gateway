"""SSE 解析：把响应流翻译成 (event_name, data_dict) 序列。

SSE 是两家厂商共用的传输外壳，与具体业务协议无关，
所以作为公共模块被两个适配器复用，而不是各自复制一份。
"""

import json
from collections.abc import Iterator


def iter_sse(response) -> Iterator[tuple[str, dict]]:
    """逐事件 yield (event_name, data_dict)。

    SSE 规则：空行 = 一个事件结束；event: 行是事件名；data: 行是数据。
    兼容上游省略结尾空行的情况：流结束时补一次 flush。
    """
    event_name = ""
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            event = _parse_event(event_name, data_lines)
            if event is not None:
                yield event
            event_name = ""
            data_lines = []
        elif line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    # 流结束但缓冲区还有未刷新的事件（上游没发结尾空行）
    event = _parse_event(event_name, data_lines)
    if event is not None:
        yield event


def _parse_event(event_name: str, data_lines: list[str]) -> tuple[str, dict] | None:
    if not data_lines:
        return None
    try:
        return event_name, json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return None
