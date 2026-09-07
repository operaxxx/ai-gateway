"""结构化输出校验模块（与厂商无关）。

职责：
1. 从 LLM 返回的文本中容错提取 JSON
2. 用 pydantic 按 JSON Schema 校验
3. 返回结构化错误信息（原因 + 字段路径 + 建议）

不负责约束 LLM 输出（那是 adapter 层的事），只做解析和校验。
"""

import json
import re
from dataclasses import dataclass


@dataclass(slots=True)
class FieldError:
    """单个字段的校验错误。"""
    loc: tuple[str, ...]   # 字段路径，如 ("name",) 或 ("items", 0, "price")
    type: str              # 错误类型，如 "missing", "string_type", "value_error"
    message: str           # 人类可读的错误描述


@dataclass(slots=True)
class ValidationResult:
    """校验结果：成功返回 parsed（dict），失败返回 errors 列表。"""
    ok: bool
    parsed: dict | list | None = None
    errors: list[FieldError] | None = None
    raw_text: str = ""     # 原始输入文本（供错误信息引用）


# ---------- 容错 JSON 提取 ----------

# 匹配 ```json ... ``` 代码块
_RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def extract_json(text: str) -> str | None:
    """从可能带噪声的文本中提取 JSON 字符串。

    策略（逐级降级）：
    1. 直接 json.loads — 最理想情况
    2. 提取 ```json ... ``` 代码块
    3. 定位第一个 { 或 [ 到最后一个 } 或 ]
    4. 全部失败返回 None
    """
    text = text.strip()

    # 策略 1：直接解析
    if _try_json(text) is not None:
        return text

    # 策略 2：代码块
    m = _RE_JSON_BLOCK.search(text)
    if m:
        candidate = m.group(1).strip()
        if _try_json(candidate) is not None:
            return candidate

    # 策略 3：花括号/方括号定位
    candidate = _extract_by_brackets(text)
    if candidate is not None:
        return candidate

    return None


def _try_json(text: str) -> object | None:
    """尝试 json.loads，成功返回解析值，失败返回 None。"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_by_brackets(text: str) -> str | None:
    """定位第一个 { 到最后一个 }（或 [ 到 ]）的子串。"""
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            if _try_json(candidate) is not None:
                return candidate
    return None


# ---------- Schema 校验 ----------

def validate(json_str: str, schema: dict) -> ValidationResult:
    """按 JSON Schema 校验 JSON 字符串。

    Args:
        json_str: LLM 返回的文本（可以是纯 JSON 也可以带噪声）
        schema:  JSON Schema 字典

    Returns:
        ValidationResult: ok=True 时 parsed 为解析后的 dict/list
    """
    extracted = extract_json(json_str)
    if extracted is None:
        return ValidationResult(
            ok=False,
            errors=[FieldError(
                loc=(),
                type="json_parse_error",
                message="无法从输出中提取有效 JSON。输出可能不是 JSON 格式，或包含无法解析的语法错误。",
            )],
            raw_text=json_str,
        )

    # 先解析成 Python 对象
    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as e:
        return ValidationResult(
            ok=False,
            errors=[FieldError(
                loc=(),
                type="json_decode_error",
                message=f"JSON 解码失败：{e.msg}（行 {e.lineno} 列 {e.colno}）",
            )],
            raw_text=json_str,
        )

    # 用轻量 JSON Schema 校验器校验（覆盖常用关键字，不引入额外依赖）
    schema_errors = _validate_against_schema(data, schema)
    if schema_errors:
        return ValidationResult(ok=False, errors=schema_errors, raw_text=json_str)

    return ValidationResult(ok=True, parsed=data, raw_text=json_str)


def _validate_against_schema(data: object, schema: dict, path: tuple = ()) -> list[FieldError]:
    """轻量 JSON Schema 校验器（覆盖常用关键字）。

    支持的 JSON Schema 关键字：
    - type, properties, required, items
    - enum, minimum, maximum, minLength, maxLength
    - additionalProperties (false 时禁止额外字段)
    """
    errors: list[FieldError] = []

    expected_type = schema.get("type")
    if expected_type:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
            "null": type(None),
        }
        # JSON 解析后 true/false 变成 Python bool，是 int 的子类，需排除
        if expected_type == "integer" and isinstance(data, bool):
            errors.append(FieldError(
                loc=path, type="type_error.integer",
                message=f"期望 integer，得到 boolean",
            ))
            return errors
        if expected_type == "number" and isinstance(data, bool):
            errors.append(FieldError(
                loc=path, type="type_error.number",
                message=f"期望 number，得到 boolean",
            ))
            return errors

        py_type = type_map.get(expected_type)
        if py_type and not isinstance(data, py_type):
            errors.append(FieldError(
                loc=path, type="type_error",
                message=f"期望 {expected_type}，得到 {_type_name(data)}",
            ))
            return errors  # 类型不对就不继续往下校验了

    # enum 校验
    if "enum" in schema and data not in schema["enum"]:
        allowed = ", ".join(repr(v) for v in schema["enum"])
        errors.append(FieldError(
            loc=path, type="value_error.enum",
            message=f"值 {data!r} 不在允许列表中（允许：{allowed}）",
        ))

    # 数值范围
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(FieldError(
                loc=path, type="value_error.minimum",
                message=f"值 {data} 小于最小值 {schema['minimum']}",
            ))
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(FieldError(
                loc=path, type="value_error.maximum",
                message=f"值 {data} 大于最大值 {schema['maximum']}",
            ))

    # 字符串长度
    if isinstance(data, str):
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(FieldError(
                loc=path, type="value_error.minLength",
                message=f"字符串长度 {len(data)} 小于最小长度 {schema['minLength']}",
            ))
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(FieldError(
                loc=path, type="value_error.maxLength",
                message=f"字符串长度 {len(data)} 大于最大长度 {schema['maxLength']}",
            ))

    # 对象校验
    if isinstance(data, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # 必填字段检查
        for req_field in required:
            if req_field not in data:
                errors.append(FieldError(
                    loc=path + (req_field,),
                    type="missing_field",
                    message=f"缺少必填字段：{req_field}",
                ))

        # 非空检查（minLength=1 已覆盖字符串，这里补 None 检查）
        for key, value in data.items():
            if value is None:
                field_schema = properties.get(key, {})
                if field_schema.get("type") != "null" and key in required:
                    errors.append(FieldError(
                        loc=path + (key,),
                        type="null_value",
                        message=f"必填字段 {key} 的值为 null",
                    ))

        # 递归校验 properties
        for key, value in data.items():
            if key in properties:
                errors.extend(_validate_against_schema(value, properties[key], path + (key,)))

        # additionalProperties: false
        if schema.get("additionalProperties") is False:
            extra = set(data.keys()) - set(properties.keys())
            for key in extra:
                errors.append(FieldError(
                    loc=path + (key,),
                    type="unexpected_property",
                    message=f"不允许的额外字段：{key}",
                ))

    # 数组校验
    if isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                errors.extend(_validate_against_schema(item, items_schema, path + (i,)))

        # minItems / maxItems
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(FieldError(
                loc=path, type="value_error.minItems",
                message=f"数组长度 {len(data)} 小于最小 {schema['minItems']}",
            ))
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(FieldError(
                loc=path, type="value_error.maxItems",
                message=f"数组长度 {len(data)} 大于最大 {schema['maxItems']}",
            ))

    return errors


def _type_name(data: object) -> str:
    """返回数据的类型名（人类可读）。"""
    if isinstance(data, bool):
        return "boolean"
    if isinstance(data, int):
        return "integer"
    if isinstance(data, float):
        return "number"
    if isinstance(data, str):
        return "string"
    if isinstance(data, dict):
        return "object"
    if isinstance(data, list):
        return "array"
    if data is None:
        return "null"
    return type(data).__name__
