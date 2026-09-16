"""结构化输出功能测试。

测试范围：
1. extract_json 容错提取
2. validate Schema 校验（标准、嵌套、错误类型）
3. 网关层 StructuredOutputError
4. HTTP 层 422 / 400 响应
5. 流式 + 结构化输出（JSON 增量实时透传，done 事件携带校验结论）

运行：
  uv run python -m pytest test_structured_output.py -v
"""

import json

import httpx
from fastapi.testclient import TestClient

from gateway.structured_output import extract_json, find_unsupported_keywords, validate
from gateway.anthropic_adapter import AnthropicAdapter
from gateway.responses_adapter import ResponsesAdapter
from gateway.gateway import Gateway
from gateway.types import ChatRequest, Message
import server
from server import app


# ---------- extract_json 容错测试 ----------

class TestExtractJson:
    def test_pure_json_object(self):
        text = '{"name": "张三", "age": 25}'
        assert json.loads(extract_json(text)) == {"name": "张三", "age": 25}

    def test_pure_json_array(self):
        text = '[1, 2, 3]'
        assert json.loads(extract_json(text)) == [1, 2, 3]

    def test_json_with_prose_prefix(self):
        text = '这是结果：\n{"name": "李四", "age": 30}'
        assert json.loads(extract_json(text)) == {"name": "李四", "age": 30}

    def test_json_code_block(self):
        text = '好的，结果如下：\n```json\n{"name": "王五", "age": 40}\n```\n以上。'
        assert json.loads(extract_json(text)) == {"name": "王五", "age": 40}

    def test_json_code_block_no_lang(self):
        text = '```\n{"name": "赵六"}\n```'
        assert json.loads(extract_json(text)) == {"name": "赵六"}

    def test_invalid_text_returns_none(self):
        assert extract_json("这不是 JSON") is None

    def test_empty_string(self):
        assert extract_json("") is None


# ---------- validate Schema 校验测试 ----------

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "tags"],
        },
        "score": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "required": ["user", "score"],
}


class TestValidate:
    def test_valid_simple(self):
        text = '{"name": "张三", "age": 25}'
        result = validate(text, SIMPLE_SCHEMA)
        assert result.ok
        assert result.parsed == {"name": "张三", "age": 25}

    def test_valid_with_prose(self):
        text = '结果是 {"name": "李四", "age": 30} 请查收'
        result = validate(text, SIMPLE_SCHEMA)
        assert result.ok
        assert result.parsed["name"] == "李四"

    def test_valid_code_block(self):
        text = '```json\n{"name": "王五", "age": 40}\n```'
        result = validate(text, SIMPLE_SCHEMA)
        assert result.ok
        assert result.parsed["age"] == 40

    def test_missing_required_field(self):
        text = '{"name": "张三"}'
        result = validate(text, SIMPLE_SCHEMA)
        assert not result.ok
        assert len(result.errors) >= 1
        locs = [e.loc for e in result.errors]
        assert ("age",) in locs

    def test_type_mismatch(self):
        text = '{"name": 123, "age": "二十五"}'
        result = validate(text, SIMPLE_SCHEMA)
        assert not result.ok
        type_errors = [e for e in result.errors if e.type == "type_error"]
        assert len(type_errors) >= 2

    def test_additional_property_rejected(self):
        text = '{"name": "张三", "age": 25, "extra": "nope"}'
        result = validate(text, SIMPLE_SCHEMA)
        assert not result.ok
        extra_errors = [e for e in result.errors if e.type == "unexpected_property"]
        assert len(extra_errors) == 1
        assert "extra" in extra_errors[0].loc

    def test_valid_nested(self):
        text = '{"user": {"name": "test", "tags": ["a", "b"]}, "score": 85.5}'
        result = validate(text, NESTED_SCHEMA)
        assert result.ok
        assert result.parsed["user"]["tags"] == ["a", "b"]

    def test_nested_missing_field(self):
        text = '{"user": {"name": "test"}, "score": 50}'
        result = validate(text, NESTED_SCHEMA)
        assert not result.ok
        missing = [e for e in result.errors if e.type == "missing_field"]
        assert any("tags" in e.loc for e in missing)

    def test_nested_wrong_item_type(self):
        text = '{"user": {"name": "test", "tags": [1, 2]}, "score": 50}'
        result = validate(text, NESTED_SCHEMA)
        assert not result.ok
        type_errors = [e for e in result.errors if e.type == "type_error"]
        assert len(type_errors) >= 1

    def test_number_range_violation(self):
        text = '{"user": {"name": "test", "tags": ["a"]}, "score": 150}'
        result = validate(text, NESTED_SCHEMA)
        assert not result.ok
        max_errors = [e for e in result.errors if e.type == "value_error.maximum"]
        assert len(max_errors) == 1

    def test_empty_string_rejected_when_min_length(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
            },
            "required": ["name"],
        }
        text = '{"name": ""}'
        result = validate(text, schema)
        assert not result.ok
        min_errors = [e for e in result.errors if e.type == "value_error.minLength"]
        assert len(min_errors) == 1

    def test_null_value_in_required_field(self):
        text = '{"name": null, "age": 25}'
        result = validate(text, SIMPLE_SCHEMA)
        assert not result.ok
        # null 不符合 string 类型
        type_errors = [e for e in result.errors if e.type == "type_error"]
        assert len(type_errors) >= 1

    def test_enum_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["level"],
        }
        text = '{"level": "extreme"}'
        result = validate(text, schema)
        assert not result.ok
        enum_errors = [e for e in result.errors if e.type == "value_error.enum"]
        assert len(enum_errors) == 1

    def test_invalid_json(self):
        result = validate("not json at all", SIMPLE_SCHEMA)
        assert not result.ok
        assert result.errors[0].type == "json_parse_error"

    def test_array_min_items(self):
        schema = {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
        }
        text = '[1]'
        result = validate(text, schema)
        assert not result.ok
        min_errors = [e for e in result.errors if e.type == "value_error.minItems"]
        assert len(min_errors) == 1

    def test_boolean_not_integer(self):
        """JSON 的 true/false 在 Python 里是 bool（int 子类），不应被当作 integer"""
        text = '{"name": "test", "age": true}'
        result = validate(text, SIMPLE_SCHEMA)
        assert not result.ok
        type_errors = [e for e in result.errors if e.type == "type_error.integer"]
        assert len(type_errors) == 1


# ---------- Schema 边界 fail-fast 检测（不支持的关键字绝不静默漏校验） ----------

class TestSchemaBoundary:
    def test_unsupported_keyword_rejected(self):
        """顶层出现不支持关键字（oneOf）：validate 直接失败，type=unsupported_schema_keyword"""
        schema = {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
        }
        result = validate('"anything"', schema)
        assert not result.ok
        assert result.errors[0].type == "unsupported_schema_keyword"
        assert "oneOf" in result.errors[0].message

    def test_unsupported_keyword_in_nested_properties(self):
        """嵌套 properties 深处的不支持关键字也能被定位"""
        schema = {
            "type": "object",
            "properties": {
                "ok_field": {"type": "string"},
                "bad_field": {"type": "string", "pattern": "^a"},
            },
        }
        errors = find_unsupported_keywords(schema)
        assert len(errors) == 1
        assert errors[0].loc == ("properties", "bad_field", "pattern")

    def test_type_array_form_rejected(self):
        """type 数组形式（type: ["string", "null"]）会静默漏校验，必须拦截"""
        schema = {"type": ["string", "null"]}
        errors = find_unsupported_keywords(schema)
        assert len(errors) == 1
        assert errors[0].loc == ("type",)

    def test_items_array_form_rejected(self):
        """items 数组形式（按位置元组校验）未实现，必须拦截"""
        schema = {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]}
        errors = find_unsupported_keywords(schema)
        assert len(errors) == 1
        assert errors[0].loc == ("items",)

    def test_supported_schema_passes_boundary(self):
        """全部在支持范围内的 schema 不报任何边界错误"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        assert find_unsupported_keywords(schema) == []

    def test_validate_preflight_blocks_before_json_check(self):
        """validate() 入口先查边界：即便正文是合法 JSON，坏 schema 也报 unsupported 而非解析成功"""
        schema = {"type": "object", "format": "uri", "properties": {}}
        result = validate("{}", schema)
        assert not result.ok
        assert all(e.type == "unsupported_schema_keyword" for e in result.errors)


# ---------- 适配器 payload 翻译测试（纯函数，防上游协议字段回归） ----------

class TestAdapterPayload:
    SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}}

    def _request(self, model):
        return ChatRequest(
            model=model,
            messages=[Message(role="user", content="hi")],
            response_format=self.SCHEMA,
        )

    def test_anthropic_uses_tool_use_mode(self):
        """Anthropic 协议：response_format 翻译成 tools[0].input_schema"""
        adapter = AnthropicAdapter(api_key="test-key", base_url="https://mock")
        payload = adapter._build_payload(self._request("deepseek-v4-flash"))
        assert payload["tools"][0]["input_schema"] == self.SCHEMA
        assert "text" not in payload

    def test_responses_uses_json_schema_format_with_name(self):
        """Responses 协议：text.format.type=json_schema 且必须带 name（上游 400 教训）"""
        adapter = ResponsesAdapter(api_key="test-key", base_url="https://mock")
        payload = adapter._build_payload(self._request("deepseek-v4-pro"))
        fmt = payload["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["name"]
        assert fmt["schema"] == self.SCHEMA


# ---------- HTTP 层测试（用 TestClient mock） ----------

class TestHttpLayer:
    def setup_method(self):
        self.client = TestClient(app)

    def test_unsupported_model_returns_400(self):
        resp = self.client.post("/v1/chat", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400

    def test_unsupported_schema_preflight_returns_400(self):
        """schema 含不支持关键字：入口 400（fail fast），不发起 LLM 调用"""
        resp = self.client.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "object",
                "properties": {"q": {"type": "string", "pattern": "^a"}},
            },
        })
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "unsupported_schema"
        assert body["detail"]["unsupported"][0]["field"] == "properties.q.pattern"


# ---------- 流式 + 结构化输出：JSON 增量实时透传，流结束后校验 ----------

STREAM_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
}


def _anthropic_structured_sse(json_chunks: list[str], with_preamble: bool = False) -> str:
    """构造 Anthropic 结构化输出上游 SSE：正文块 + tool_use 块（input_json_delta 增量）。"""
    events = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":10}}}',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    ]
    if with_preamble:
        events.append('event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"让我想想"}}')
    events.append('event: content_block_stop\ndata: {"type":"content_block_stop","index":0}')
    events.append('event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"structured_output","input":{}}}')
    for chunk in json_chunks:
        events.append("event: content_block_delta\ndata: " + json.dumps({
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": chunk},
        }, ensure_ascii=False))
    events.append('event: content_block_stop\ndata: {"type":"content_block_stop","index":1}')
    events.append('event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}')
    events.append('event: message_stop\ndata: {"type":"message_stop"}')
    return "\n\n".join(events) + "\n\n"


def _responses_structured_sse(json_chunks: list[str]) -> str:
    """构造 Responses 协议上游 SSE：output_text.delta 携带 JSON 增量。"""
    events = [
        'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}',
    ]
    for chunk in json_chunks:
        events.append("event: response.output_text.delta\ndata: " + json.dumps({
            "type": "response.output_text.delta", "delta": chunk,
        }, ensure_ascii=False))
    events.append("event: response.completed\ndata: " + json.dumps({
        "type": "response.completed",
        "response": {"id": "resp_1", "status": "completed",
                     "usage": {"input_tokens": 10, "output_tokens": 5}},
    }))
    return "\n\n".join(events) + "\n\n"


def _mock_stream_adapter(adapter_cls, sse_body: str):
    """适配器实例 + MockTransport（流式未读状态，与真实 client.stream 行为一致）。"""
    adapter = adapter_cls(api_key="test", base_url="https://mock")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, content=iter([sse_body.encode("utf-8")]),
            headers={"content-type": "text/event-stream"},
        )

    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter.captured = captured
    return adapter


def _gateway_with(adapter_cls, sse_body: str) -> Gateway:
    adapter = _mock_stream_adapter(adapter_cls, sse_body)
    gw = Gateway()
    gw._adapters[adapter_cls] = adapter
    gw.captured_adapter = adapter
    return gw


def _stream_request(response_format=None) -> ChatRequest:
    return ChatRequest(
        model="deepseek-v4-flash",
        messages=[Message(role="user", content="hi")],
        response_format=response_format,
    )


class TestAdapterStreamStructured:
    """适配器层：结构化流式翻译（Anthropic tool 入参 JSON 增量 -> 正文 delta）。"""

    def test_anthropic_input_json_delta_streams_as_text(self):
        """input_json_delta 实时透传为正文 delta；前导文本抑制；tool_use->stop。"""
        adapter = _mock_stream_adapter(
            AnthropicAdapter, _anthropic_structured_sse(['{"answer": ', "42}"], with_preamble=True))
        events = list(adapter.stream(_stream_request(response_format=STREAM_SCHEMA)))

        assert [e.type for e in events] == ["start", "delta", "delta", "done"]
        text = "".join(e.text for e in events if e.type == "delta")
        assert text == '{"answer": 42}'
        assert "让我想想" not in text          # 与 complete() 口径一致：只取 tool 入参
        assert events[-1].stop_reason == "stop"  # tool_use 统一为 stop
        # 上游 payload：stream=true 且带 tools 约束
        assert adapter.captured["payload"]["stream"] is True
        assert adapter.captured["payload"]["tools"][0]["input_schema"] == STREAM_SCHEMA

    def test_anthropic_plain_stream_unaffected(self):
        """非结构化流式行为不变：text_delta 透传、input_json_delta 不透传。"""
        adapter = _mock_stream_adapter(
            AnthropicAdapter, _anthropic_structured_sse(['{"answer": 42}'], with_preamble=True))
        events = list(adapter.stream(_stream_request()))

        text = "".join(e.text for e in events if e.type == "delta")
        assert text == "让我想想"

    def test_responses_json_deltas_stream_as_text(self):
        """Responses 协议：output_text.delta 本就是 JSON 增量，透传不变。"""
        adapter = _mock_stream_adapter(
            ResponsesAdapter, _responses_structured_sse(['{"answer": ', "42}"]))
        req = ChatRequest(model="deepseek-v4-pro",
                          messages=[Message(role="user", content="hi")],
                          response_format=STREAM_SCHEMA)
        events = list(adapter.stream(req))

        text = "".join(e.text for e in events if e.type == "delta")
        assert text == '{"answer": 42}'
        assert adapter.captured["payload"]["stream"] is True
        assert adapter.captured["payload"]["text"]["format"]["schema"] == STREAM_SCHEMA


class TestGatewayStreamValidation:
    """网关层：流结束后对累积正文跑 schema 校验，结论挂 done 事件。"""

    def _done(self, adapter_cls, chunks, response_format=STREAM_SCHEMA):
        gw = _gateway_with(adapter_cls, _anthropic_structured_sse(chunks))
        events = list(gw.stream(_stream_request(response_format=response_format)))
        return events[-1]

    def test_valid_output_attaches_parsed(self):
        done = self._done(AnthropicAdapter, ['{"answer": ', "42}"])
        assert done.type == "done"
        assert done.structured_ok is True
        assert done.structured_parsed == {"answer": 42}
        assert done.structured_errors is None

    def test_schema_violation_attaches_field_errors(self):
        """缺必填/类型错：ok=false，errors 带 loc/type/message 明细。"""
        done = self._done(AnthropicAdapter, ['{"answer": "42"}'])
        assert done.structured_ok is False
        assert done.structured_parsed is None
        errs = done.structured_errors
        assert any(e["type"] == "type_error" and e["loc"] == ("answer",) for e in errs)

    def test_invalid_json_reports_parse_error(self):
        done = self._done(AnthropicAdapter, ["这不是 JSON"])
        assert done.structured_ok is False
        assert done.structured_errors[0]["type"] == "json_parse_error"

    def test_first_delta_is_accumulated(self):
        """回归：第一个正文 delta 也要进累积文本（ttft 分支不得吞掉增量）。"""
        done = self._done(AnthropicAdapter, ['{"answer"', ": 42}"])
        assert done.structured_parsed == {"answer": 42}

    def test_no_schema_keeps_validation_none(self):
        done = self._done(AnthropicAdapter, ['{"answer": 42}'], response_format=None)
        assert done.structured_ok is None
        assert done.structured_parsed is None
        assert done.structured_errors is None

    def test_responses_protocol_validation(self):
        gw = _gateway_with(ResponsesAdapter, _responses_structured_sse(['{"answer": ', "42}"]))
        req = ChatRequest(model="deepseek-v4-pro",
                          messages=[Message(role="user", content="hi")],
                          response_format=STREAM_SCHEMA)
        done = list(gw.stream(req))[-1]
        assert done.structured_ok is True
        assert done.structured_parsed == {"answer": 42}


class TestStreamStructuredHttp:
    """HTTP SSE 层：stream + response_format 返回 200，done 事件携带 validation。"""

    def _client(self, monkeypatch, adapter_cls, sse_body) -> TestClient:
        monkeypatch.setattr(server, "gw", _gateway_with(adapter_cls, sse_body))
        return TestClient(app)

    @staticmethod
    def _parse_sse(body: str) -> list[tuple[str, dict]]:
        out = []
        for block in body.split("\n\n"):
            if not block.strip():
                continue
            event, data = None, None
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
            out.append((event, data))
        return out

    def _post(self, client: TestClient, schema) -> httpx.Response:
        return client.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "response_format": schema,
        })

    def test_stream_with_schema_returns_200_and_validation_ok(self, monkeypatch):
        """流式 + 结构化不再 400：delta 实时透传，done 带 ok=true + parsed。"""
        client = self._client(monkeypatch, AnthropicAdapter,
                              _anthropic_structured_sse(['{"answer": ', "42}"]))
        resp = self._post(client, STREAM_SCHEMA)

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = self._parse_sse(resp.text)
        delta_text = "".join(d["text"] for e, d in events if e == "delta")
        assert delta_text == '{"answer": 42}'   # 增量实时透传，不是流完才给
        done = [d for e, d in events if e == "done"][-1]
        assert done["validation"] == {"ok": True, "parsed": {"answer": 42}}

    def test_stream_schema_violation_reports_errors(self, monkeypatch):
        """校验失败：done.validation.ok=false，errors 明确指出字段与原因。"""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["answer", "reason"],
        }
        client = self._client(monkeypatch, AnthropicAdapter,
                              _anthropic_structured_sse(['{"answer": 42}']))
        resp = self._post(client, schema)

        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        done = [d for e, d in events if e == "done"][-1]
        v = done["validation"]
        assert v["ok"] is False
        assert len(v["errors"]) >= 1
        missing = [e for e in v["errors"] if e["field"] == "reason"]
        assert missing and missing[0]["type"] == "missing_field"
        assert missing[0]["message"]

    def test_stream_without_schema_done_has_no_validation(self, monkeypatch):
        """未开结构化：done 不带 validation 字段（向后兼容）。"""
        client = self._client(monkeypatch, AnthropicAdapter,
                              _anthropic_structured_sse(['{"answer": 42}']))
        resp = client.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        done = [d for e, d in self._parse_sse(resp.text) if e == "done"][-1]
        assert "validation" not in done
