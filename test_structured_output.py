"""结构化输出功能测试。

测试范围：
1. extract_json 容错提取
2. validate Schema 校验（标准、嵌套、错误类型）
3. 网关层 StructuredOutputError
4. HTTP 层 422 / 400 响应

运行：
  uv run python -m pytest test_structured_output.py -v
"""

import json

from fastapi.testclient import TestClient

from gateway.structured_output import extract_json, validate
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


# ---------- HTTP 层测试（用 TestClient mock） ----------

class TestHttpLayer:
    def setup_method(self):
        self.client = TestClient(app)

    def test_stream_with_response_format_rejected(self):
        """stream=true + response_format 应返回 400"""
        resp = self.client.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "response_format": {"type": "object", "properties": {}},
        })
        assert resp.status_code == 400

    def test_unsupported_model_returns_400(self):
        resp = self.client.post("/v1/chat", json={
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 400
