"""Prompt 模板渲染：基于 Jinja2（严格模式）。

为什么用 Jinja2 而不是手写正则：
- 业界事实标准（LangChain 等同款），除变量占位外还支持条件 / 循环 / 过滤器，
  模板表达能力随需求增长无需更换引擎
- 语法错误在创建/渲染时即被捕获（TemplateSyntaxError → InvalidTemplateError），
  而不是静默输出错误内容

语义约定：
- 变量占位 {{ var }}，值可以是任意 JSON 类型（str/list/dict，供循环使用）
- 缺失变量：渲染期抛 UndefinedError → MissingVariablesError（Jinja2 逐个报，
  第一个缺的先报）。不做静态全量拦截，以保留 {{ x | default("y") }}、
  {% if x is defined %} 等惯用法
- 多余变量：忽略（调用方可传超集，便于多场景复用同一模板）
- StrictUndefined：任何未定义引用都是错误，绝不静默输出空串

Environment 配置（针对 prompt 场景）：
- autoescape 关闭（默认）：prompt 是纯文本，不做 HTML 转义
- trim_blocks / lstrip_blocks：{% %} 控制标签独占一行时不留多余空行（多行 prompt 友好）
- keep_trailing_newline：保留模板末尾换行（Jinja2 默认会吃掉）
- 无沙箱：模板由存储库管理（管理员创建），不是终端用户输入
"""

import re
from typing import Any

from jinja2 import Environment, StrictUndefined
from jinja2 import meta as jinja_meta
from jinja2.exceptions import TemplateSyntaxError, UndefinedError

_env = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


class MissingVariablesError(Exception):
    """渲染时缺少模板引用的必需变量。"""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"缺少变量: {', '.join(missing)}")


class InvalidTemplateError(Exception):
    """模板语法错误（Jinja2 无法解析）。"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(f"模板语法错误: {message}")


def _first_pos(name: str, template: str) -> int:
    """变量名在模板文本中首次出现的位置（仅用于排序）。"""
    m = re.search(rf"\b{re.escape(name)}\b", template)
    return m.start() if m else 0


def extract_variables(template: str) -> list[str]:
    """解析模板 AST，提取所有引用的变量名，按首次出现顺序去重。

    语法错误抛 InvalidTemplateError。循环变量 / {% set %} / macro 参数
    属于已声明绑定，不算引用变量（Jinja2 作用域分析保证）。
    """
    try:
        ast = _env.parse(template)
    except TemplateSyntaxError as e:
        raise InvalidTemplateError(f"第 {e.lineno} 行: {e.message}") from e
    # find_undeclared_variables 返回集合（无序），按模板中首次出现位置排序
    names = jinja_meta.find_undeclared_variables(ast)
    return sorted(names, key=lambda n: _first_pos(n, template))


def render(template: str, variables: dict[str, Any] | None) -> str:
    """用 variables 渲染模板（支持 Jinja2 全部语法），返回渲染后的文本。

    - 缺失引用变量 → MissingVariablesError（报告第一个缺失项）
    - 多余变量忽略；语法错误 → InvalidTemplateError
    """
    variables = variables or {}
    try:
        return _env.from_string(template).render(**variables)
    except UndefinedError as e:
        # StrictUndefined 报错消息形如 "'lang' is undefined"，提取变量名
        m = re.match(r"'([^']+)' is undefined", str(e))
        raise MissingVariablesError([m.group(1) if m else str(e)]) from e
    except TemplateSyntaxError as e:
        raise InvalidTemplateError(f"第 {e.lineno} 行: {e.message}") from e
