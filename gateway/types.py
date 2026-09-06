from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal['system', 'user', 'assistant']

@dataclass(slots=True)
class Message:
    role: Role
    content: str

@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None

@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass(slots=True)
class ChatResponse:
    text: str
    model: str
    usage: Usage
    stop_reason: str
    raw: dict[str, Any] = field(default_factory=dict)
