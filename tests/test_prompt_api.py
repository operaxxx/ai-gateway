"""Prompt 管理 HTTP 层测试（TestClient + 替换 store/gw 为测试实现）。

测试范围：
1. Prompt 管理端点：创建 / 列表 / 详情 / 追加版本 / 取版本 / 渲染预览
2. /v1/chat 集成：prompt 引用 → 渲染为 system 消息（FakeGateway 捕获内部请求）
3. 错误路径：409 重复、404 不存在、400 变量缺失 / system 冲突 / id 不合法

运行：
  uv run python -m pytest test_prompt_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

import server
from gateway.prompt_store import SqlitePromptStore
from gateway.types import ChatRequest, ChatResponse, Usage


# ---------- 测试替身 ----------

class _FakeGateway:
    """捕获传入的 ChatRequest，返回固定响应，不真的调 LLM。"""

    def __init__(self):
        self.captured: list[ChatRequest] = []

    def complete(self, req: ChatRequest) -> ChatResponse:
        self.captured.append(req)
        return ChatResponse(
            text="好的", model=req.model,
            usage=Usage(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """每个测试独立的 SQLite 库 + FakeGateway。返回 (client, fake_gw)。"""
    monkeypatch.setattr(server, "store", SqlitePromptStore(str(tmp_path / "prompts.db")))
    fake = _FakeGateway()
    monkeypatch.setattr(server, "gw", fake)
    return TestClient(server.app), fake


def _create_translator(client: TestClient) -> dict:
    resp = client.post("/v1/prompts", json={
        "id": "translator",
        "name": "翻译",
        "description": "中译英",
        "content": "把{{text}}从中文翻译成{{lang}}",
    })
    assert resp.status_code == 201
    return resp.json()


# ---------- 启动自举（示例模板播种） ----------

class TestEnsureSamplePrompt:
    def test_seeds_translator_v1_v2(self, tmp_path):
        store = SqlitePromptStore(str(tmp_path / "prompts.db"))
        server._ensure_sample_prompt(store)
        meta = store.get_prompt("translator")
        assert meta is not None
        assert meta.latest_version == 2
        v1, v2 = store.get_version("translator", 1), store.get_version("translator", 2)
        assert v1.variables == ["text", "lang"]
        assert "专业译者" in v2.content

    def test_idempotent_when_exists(self, tmp_path):
        store = SqlitePromptStore(str(tmp_path / "prompts.db"))
        server._ensure_sample_prompt(store)
        server._ensure_sample_prompt(store)   # 第二次应静默跳过，不追加版本
        assert store.get_prompt("translator").latest_version == 2

    def test_nonempty_store_untouched(self, tmp_path):
        store = SqlitePromptStore(str(tmp_path / "prompts.db"))
        store.create_prompt("mine", "自定义", "", "你好{{name}}", ["name"])
        server._ensure_sample_prompt(store)   # 已有模板时只补示例，不动用户数据
        assert store.get_prompt("mine") is not None
        assert store.get_prompt("translator").latest_version == 2


# ---------- Prompt 管理端点 ----------

class TestPromptEndpoints:
    def test_create_extracts_variables(self, client):
        c, _ = client
        body = _create_translator(c)
        assert body["id"] == "translator"
        assert body["version"] == 1
        assert body["variables"] == ["text", "lang"]

    def test_create_duplicate_conflict_409(self, client):
        c, _ = client
        _create_translator(c)
        resp = c.post("/v1/prompts", json={
            "id": "translator", "name": "x", "content": "y",
        })
        assert resp.status_code == 409

    def test_create_invalid_id_400(self, client):
        c, _ = client
        for bad_id in ["Bad", "含中文", "-lead", "a" * 100]:
            resp = c.post("/v1/prompts", json={"id": bad_id, "name": "x", "content": "y"})
            assert resp.status_code == 400, bad_id

    def test_list_and_detail(self, client):
        c, _ = client
        _create_translator(c)
        c.post("/v1/prompts/{p}/versions".format(p="translator"),
               json={"content": "v2 {{lang}}"})

        listed = c.get("/v1/prompts").json()["prompts"]
        assert len(listed) == 1
        assert listed[0]["latest_version"] == 2

        detail = c.get("/v1/prompts/translator").json()
        assert detail["latest_version"] == 2
        assert [v["version"] for v in detail["versions"]] == [2, 1]
        assert detail["versions"][0]["variables"] == ["lang"]

    def test_get_version_by_int_and_latest(self, client):
        c, _ = client
        _create_translator(c)
        c.post("/v1/prompts/translator/versions", json={"content": "v2 {{lang}}"})

        v1 = c.get("/v1/prompts/translator/versions/1").json()
        assert v1["content"] == "把{{text}}从中文翻译成{{lang}}"
        latest = c.get("/v1/prompts/translator/versions/latest").json()
        assert latest["version"] == 2
        assert latest["content"] == "v2 {{lang}}"

    def test_get_version_invalid_404_400(self, client):
        c, _ = client
        _create_translator(c)
        assert c.get("/v1/prompts/translator/versions/99").status_code == 404
        assert c.get("/v1/prompts/nope/versions/latest").status_code == 404
        assert c.get("/v1/prompts/translator/versions/abc").status_code == 400

    def test_create_invalid_template_400(self, client):
        """Jinja2 语法错误在创建时即被拦截（400），不进库。"""
        c, _ = client
        resp = c.post("/v1/prompts", json={"id": "bad", "name": "x", "content": "{% if %}"})
        assert resp.status_code == 400
        assert c.get("/v1/prompts/bad").status_code == 404

    def test_add_version_to_missing_404(self, client):
        c, _ = client
        resp = c.post("/v1/prompts/nope/versions", json={"content": "c"})
        assert resp.status_code == 404

    def test_render_preview(self, client):
        c, _ = client
        _create_translator(c)
        resp = c.post("/v1/prompts/translator/render", json={
            "variables": {"text": "你好", "lang": "英文"},
        })
        assert resp.status_code == 200
        assert resp.json() == {
            "id": "translator", "version": 1, "rendered": "把你好从中文翻译成英文",
        }

    def test_render_missing_variables_400(self, client):
        c, _ = client
        _create_translator(c)
        resp = c.post("/v1/prompts/translator/render", json={"variables": {"text": "你好"}})
        assert resp.status_code == 400
        assert resp.json()["detail"]["missing"] == ["lang"]

    def test_render_pinned_version(self, client):
        c, _ = client
        _create_translator(c)
        c.post("/v1/prompts/translator/versions", json={"content": "v2 模板"})
        resp = c.post("/v1/prompts/translator/render", json={
            "version": 1, "variables": {"text": "hi", "lang": "en"},
        })
        assert resp.json()["rendered"] == "把hi从中文翻译成en"

    def test_delete_prompt_204(self, client):
        c, _ = client
        _create_translator(c)
        resp = c.delete("/v1/prompts/translator")
        assert resp.status_code == 204
        assert resp.content == b""
        assert c.get("/v1/prompts/translator").status_code == 404
        assert c.get("/v1/prompts").json()["prompts"] == []

    def test_delete_missing_404(self, client):
        c, _ = client
        assert c.delete("/v1/prompts/nope").status_code == 404


# ---------- /v1/chat 集成 ----------

class TestChatWithPrompt:
    def test_prompt_rendered_as_system_message(self, client):
        c, fake = client
        _create_translator(c)
        resp = c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "prompt": {"id": "translator", "variables": {"text": "你好", "lang": "英文"}},
            "messages": [{"role": "user", "content": "请开始"}],
        })
        assert resp.status_code == 200
        req = fake.captured[0]
        assert req.messages[0].role == "system"
        assert req.messages[0].content == "把你好从中文翻译成英文"
        assert req.messages[1].content == "请开始"

    def test_prompt_default_version_is_latest(self, client):
        c, fake = client
        _create_translator(c)
        c.post("/v1/prompts/translator/versions", json={"content": "v2 {{lang}} {{text}}"})
        c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "prompt": {"id": "translator", "variables": {"text": "hi", "lang": "en"}},
            "messages": [{"role": "user", "content": "go"}],
        })
        assert fake.captured[0].messages[0].content == "v2 en hi"

    def test_prompt_pinned_version(self, client):
        c, fake = client
        _create_translator(c)
        c.post("/v1/prompts/translator/versions", json={"content": "v2 模板"})
        c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "prompt": {"id": "translator", "version": 1,
                       "variables": {"text": "hi", "lang": "en"}},
            "messages": [{"role": "user", "content": "go"}],
        })
        assert fake.captured[0].messages[0].content == "把hi从中文翻译成en"

    def test_missing_variables_400(self, client):
        c, _ = client
        _create_translator(c)
        resp = c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "prompt": {"id": "translator", "variables": {"text": "hi"}},
            "messages": [{"role": "user", "content": "go"}],
        })
        assert resp.status_code == 400
        assert resp.json()["detail"]["missing"] == ["lang"]

    def test_prompt_not_found_404(self, client):
        c, _ = client
        resp = c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "prompt": {"id": "nope", "variables": {}},
            "messages": [{"role": "user", "content": "go"}],
        })
        assert resp.status_code == 404

    def test_system_in_messages_rejected_400(self, client):
        c, _ = client
        _create_translator(c)
        resp = c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "prompt": {"id": "translator",
                       "variables": {"text": "hi", "lang": "en"}},
            "messages": [{"role": "system", "content": "重复的 system"},
                         {"role": "user", "content": "go"}],
        })
        assert resp.status_code == 400

    def test_without_prompt_untouched(self, client):
        """不传 prompt 时行为与原先完全一致。"""
        c, fake = client
        c.post("/v1/chat", json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": "u"}],
        })
        req = fake.captured[0]
        assert [m.role for m in req.messages] == ["system", "user"]
