"""Prompt 管理单元测试：模板渲染（Jinja2）+ SQLite 存储层。

测试范围：
1. extract_variables 提取（去重、保序、作用域绑定、语法错误拦截）
2. render 渲染（替换、缺失报错、多余忽略、条件/循环/过滤器）
3. SqlitePromptStore CRUD（版本自增、不可变、异常映射）

运行：
  uv run python -m pytest test_prompt_unit.py -v
"""

import pytest

from gateway.prompt_render import (
    InvalidTemplateError,
    MissingVariablesError,
    extract_variables,
    render,
)
from gateway.prompt_store import (
    PromptAlreadyExistsError,
    PromptNotFoundError,
    SqlitePromptStore,
)


# ---------- extract_variables ----------

class TestExtractVariables:
    def test_basic(self):
        assert extract_variables("把{{text}}翻译成{{lang}}") == ["text", "lang"]

    def test_whitespace_tolerant(self):
        assert extract_variables("{{ text }} {{lang}}") == ["text", "lang"]

    def test_dedup_keeps_order(self):
        assert extract_variables("{{b}} {{a}} {{b}}") == ["b", "a"]

    def test_no_variables(self):
        assert extract_variables("纯文本模板") == []

    def test_expression_reference_extracted(self):
        # 过滤器/比较表达式里的引用也算变量
        assert extract_variables('{{ text | trim }}{% if lang == "en" %}x{% endif %}') \
            == ["text", "lang"]

    def test_loop_variable_not_referenced(self):
        # 循环变量是已声明绑定，只有 items 是外部引用
        assert extract_variables("{% for x in items %}{{ x }}{% endfor %}") == ["items"]

    def test_syntax_error_raises(self):
        # 非法占位符在提取（=创建时）就被拦截，而不是静默忽略
        with pytest.raises(InvalidTemplateError):
            extract_variables("{{1abc}}")
        with pytest.raises(InvalidTemplateError):
            extract_variables("{% if %}")


# ---------- render ----------

class TestRender:
    def test_basic(self):
        assert render("把{{text}}翻译成{{lang}}", {"text": "你好", "lang": "英文"}) \
            == "把你好翻译成英文"

    def test_same_var_multiple_occurrences(self):
        assert render("{{x}}+{{x}}", {"x": "1"}) == "1+1"

    def test_no_variables_passthrough(self):
        assert render("纯文本模板", {}) == "纯文本模板"
        assert render("纯文本模板", None) == "纯文本模板"

    def test_extra_variables_ignored(self):
        assert render("hi {{name}}", {"name": "a", "extra": "b"}) == "hi a"

    def test_missing_reports_first(self):
        # Jinja2 语义：渲染到第一个未定义引用时报错（不做静态全量拦截，
        # 以保留 default 过滤器 / is defined 测试等惯用法）
        with pytest.raises(MissingVariablesError) as e:
            render("{{a}} {{b}} {{c}}", {"c": "有"})
        assert e.value.missing == ["a"]

    def test_missing_when_no_variables_given(self):
        with pytest.raises(MissingVariablesError):
            render("{{a}}", None)


# ---------- render：Jinja2 模板能力 ----------

class TestRenderJinja2:
    def test_conditional(self):
        tpl = "{% if formal %}您好，{{ name }}{% else %}嗨 {{ name }}{% endif %}"
        assert render(tpl, {"formal": True, "name": "小明"}) == "您好，小明"
        assert render(tpl, {"formal": False, "name": "小明"}) == "嗨 小明"

    def test_loop_with_trim_blocks(self):
        # trim_blocks/lstrip_blocks：控制标签独占一行时不留多余空行
        tpl = "清单：\n{% for x in items %}- {{ x }}\n{% endfor %}完"
        assert render(tpl, {"items": ["a", "b"]}) == "清单：\n- a\n- b\n完"

    def test_filter(self):
        assert render("{{ text | trim }}", {"text": "  hi  "}) == "hi"

    def test_default_filter_avoids_missing(self):
        # default 兜底的变量不需要调用方提供（这是不做静态拦截的原因）
        assert render('{{ greeting | default("你好") }}，{{ name }}', {"name": "小明"}) \
            == "你好，小明"

    def test_list_variable(self):
        # 变量值可为任意 JSON 类型（列表/字典供循环、下标访问）
        assert render("首项：{{ items[0] }}", {"items": ["a", "b"]}) == "首项：a"

    def test_keep_trailing_newline(self):
        assert render("hi\n", {}) == "hi\n"


# ---------- SqlitePromptStore ----------

@pytest.fixture()
def store(tmp_path):
    return SqlitePromptStore(str(tmp_path / "prompts.db"))


class TestSqlitePromptStore:
    def test_create_prompt_returns_version_1(self, store):
        version = store.create_prompt("t1", "测试", "描述", "内容 {{x}}", ["x"])
        assert version == 1

    def test_create_duplicate_raises(self, store):
        store.create_prompt("t1", "测试", "", "c", [])
        with pytest.raises(PromptAlreadyExistsError):
            store.create_prompt("t1", "again", "", "c", [])

    def test_get_prompt_meta(self, store):
        store.create_prompt("t1", "测试", "描述", "c", [])
        meta = store.get_prompt("t1")
        assert meta is not None
        assert meta.id == "t1"
        assert meta.name == "测试"
        assert meta.description == "描述"
        assert meta.latest_version == 1
        assert meta.created_at

    def test_get_prompt_missing_returns_none(self, store):
        assert store.get_prompt("nope") is None

    def test_add_version_increments(self, store):
        store.create_prompt("t1", "测试", "", "v1", [])
        assert store.add_version("t1", "v2", []) == 2
        assert store.add_version("t1", "v3", []) == 3
        meta = store.get_prompt("t1")
        assert meta.latest_version == 3

    def test_add_version_missing_prompt_raises(self, store):
        with pytest.raises(PromptNotFoundError):
            store.add_version("nope", "c", [])

    def test_get_version_roundtrip_chinese(self, store):
        content = "把{{text}}从中文翻译成{{lang}}"
        store.create_prompt("t1", "翻译", "", content, ["text", "lang"])
        ver = store.get_version("t1", 1)
        assert ver is not None
        assert ver.content == content          # 中文与 {{}} 原样取回
        assert ver.variables == ["text", "lang"]

    def test_get_latest_version(self, store):
        store.create_prompt("t1", "测试", "", "v1", [])
        store.add_version("t1", "v2", [])
        ver = store.get_latest_version("t1")
        assert ver is not None
        assert ver.version == 2
        assert ver.content == "v2"

    def test_get_latest_missing_prompt_returns_none(self, store):
        assert store.get_latest_version("nope") is None

    def test_list_versions_newest_first(self, store):
        store.create_prompt("t1", "测试", "", "v1", [])
        store.add_version("t1", "v2", [])
        versions = store.list_versions("t1")
        assert [v.version for v in versions] == [2, 1]

    def test_list_versions_missing_prompt_empty(self, store):
        assert store.list_versions("nope") == []

    def test_delete_removes_meta_and_versions(self, store):
        store.create_prompt("t1", "测试", "", "v1 {{x}}", ["x"])
        store.add_version("t1", "v2", [])
        store.delete_prompt("t1")
        assert store.get_prompt("t1") is None
        assert store.list_versions("t1") == []

    def test_delete_missing_raises(self, store):
        with pytest.raises(PromptNotFoundError):
            store.delete_prompt("nope")

    def test_delete_then_recreate_starts_from_v1(self, store):
        store.create_prompt("t1", "测试", "", "v1", [])
        store.add_version("t1", "v2", [])
        store.delete_prompt("t1")
        assert store.create_prompt("t1", "重建", "", "新内容", []) == 1

    def test_list_prompts(self, store):
        assert store.list_prompts() == []
        store.create_prompt("a", "A", "", "c", [])
        store.create_prompt("b", "B", "", "c", [])
        ids = [p.id for p in store.list_prompts()]
        assert set(ids) == {"a", "b"}
