# 结构化输出功能实施计划

## Context

当前 AI Gateway 已实现统一聊天接口（非流式 + 流式），支持 Anthropic Messages 和 OpenAI Responses 两个适配器。用户需要新增结构化输出支持，确保 LLM 返回内容严格符合指定的 JSON Schema 格式。

核心架构决策：**约束（adapter 层）与校验（gateway 层）分离**。adapter 层负责透传上游 API 的原生结构化输出机制，gateway 层做与厂商无关的独立校验。这样即使中转站的原生约束不可靠，gateway 层仍能保证输出质量。

已确认的决策：A①(仅 JSON Schema) B①(422 报错) C①(流式拒绝) D(v1 不做自动修复) E①(单文件)。

## 改造步骤

### Step 1: types.py — 加 response_format 字段

在 `ChatRequest` dataclass 末尾加一个可选字段：

```python
response_format: dict | None = None   # JSON Schema 字典，None=自由输出
```

### Step 2: 新建 gateway/structured_output.py — 核心校验模块

单文件，包含以下组件：

1. **`ValidationError` dataclass**：结构化错误，字段含 `reason`(原因)、`field_path`(受影响字段路径)、`suggestion`(建议)
2. **`StructuredOutputError(Exception)`**：包装 ValidationError，供 gateway 层抛出
3. **`extract_json(text) -> str`**：容错提取 JSON 文本
   - 先尝试直接 `json.loads(text)`
   - 失败则正则提取 ` ```json ... ``` ` 代码块
   - 再失败则定位第一个 `{` 到最后一个 `}`
   - 全部失败返回 None
4. **`parse_and_validate(text, schema) -> dict | ValidationError`**：主入口
   - 调 extract_json 提取 JSON 字符串
   - 用 pydantic `TypeAdapter(dict)` 配合 JSON Schema 做校验
   - 校验成功返回解析后的 dict
   - 校验失败返回带字段路径的 ValidationError

### Step 3: anthropic_adapter.py — 适配器透传原生约束

在 `_build_payload` 中：当 `request.response_format` 不为 None 时，使用 tool_use 模式（中转站兼容性优于最新的 `output_config.format`）：

```python
if request.response_format:
    payload["tools"] = [{
        "name": "structured_output",
        "description": "Return the result as a structured JSON object",
        "input_schema": request.response_format,
    }]
    payload["tool_choice"] = {"type": "tool", "name": "structured_output"}
```

在 `_parse_response` 中：当 `request.response_format` 不为 None 时，从 `content` 数组中找 `type: "tool_use"` 的块，取其 `input` 字段作为 text 返回。

### Step 4: responses_adapter.py — 适配器透传原生约束

在 `_build_payload` 中：当 `request.response_format` 不为 None 时，设置 `text.format`：

```python
if request.response_format:
    payload["text"] = {
        "format": {
            "type": "json_schema",
            "schema": request.response_format,
        }
    }
```

`_parse_response` 无需大改：Responses API 的正文仍在 `output[].content[].text`，只是内容变为 JSON 字符串。

### Step 5: gateway.py — 加后处理校验钩子

在 `complete()` 方法中，adapter 返回后检查 `request.response_format`：
- 不为 None 则调用 `structured_output.parse_and_validate(resp.text, request.response_format)`
- 返回 `ValidationError` 则抛 `StructuredOutputError`
- 成功则把解析后的 dict 放入 `resp.raw["structured_output"]`

### Step 6: server.py — API 层映射 + 流式拒绝

1. `ChatRequestIn` 加 `response_format: dict | None = None`
2. `_to_internal()` 映射该字段
3. `chat()` 端点：当 `stream=true` 且 `response_format` 不为 None 时返回 400
4. `chat()` 端点：捕获 `StructuredOutputError` 返回 422 + 结构化错误体

### Step 7: 测试

新建 `test_structured_output.py`，测试用例包括：
- 标准 JSON Schema 校验（简单对象）
- 复杂嵌套结构（对象内含数组、嵌套对象）
- 必填字段缺失 → ValidationError
- 类型不匹配（string 传成 int）→ ValidationError
- 空/缺失字段拒绝
- extract_json 容错（带 ```json 代码块、带前后多余文本）
- 流式 + response_format 被拒绝

## 关键文件

| 文件 | 改动 |
|------|------|
| `gateway/types.py` | 加 `response_format` 字段 |
| `gateway/structured_output.py` | **新建**：校验模块 |
| `gateway/anthropic_adapter.py` | `_build_payload` + `_parse_response` 加结构化输出分支 |
| `gateway/responses_adapter.py` | `_build_payload` 加结构化输出分支 |
| `gateway/gateway.py` | `complete()` 加后处理校验 |
| `server.py` | API 模型 + 路由 + 错误处理 |
| `test_structured_output.py` | **新建**：测试用例 |

## 验证方式

1. 运行单元测试：`uv run python -m pytest test_structured_output.py -v`
2. 启动服务后 curl 测试：
   ```bash
   # 非结构化（回归测试）
   curl http://localhost:8000/v1/chat -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}' -H "Content-Type: application/json"

   # 结构化输出
   curl http://localhost:8000/v1/chat -d '{
     "model":"deepseek-v4-flash",
     "messages":[{"role":"user","content":"Extract name and age from: John is 25 years old"}],
     "response_format":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
   }' -H "Content-Type: application/json"

   # 流式 + 结构化（应返回 400）
   curl http://localhost:8000/v1/chat -d '{"model":"deepseek-v4-flash","messages":[...],"stream":true,"response_format":{...}}' -H "Content-Type: application/json"
   ```
