"""Prompt 存储层：抽象接口（Protocol）+ SQLite 实现。

设计：
- PromptStore 协议定义存储能力，上层（server.py）只依赖协议，不绑定介质
- 版本不可变：代码里没有任何 UPDATE 版本内容的路径，"更新" = 追加新版本（版本号从 1 自增）
- 版本内容存 JSON（当前 schema: {"text": "..."}），未来扩展为 messages 数组模板时
  旧版本数据无需迁移（不可变版本天然兼容 schema 演进）
- SQLite：每个操作独立连接（FastAPI 同步端点跑在 threadpool，共享连接有线程问题）
- 版本号分配用单条 INSERT + 子查询取 MAX(version)+1，借助 SQLite 写锁保证原子性
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# ---------- 数据模型（内部抽象层，dataclass） ----------

@dataclass(slots=True)
class PromptMeta:
    id: str
    name: str
    description: str
    latest_version: int
    created_at: str


@dataclass(slots=True)
class PromptVersion:
    prompt_id: str
    version: int
    content: str            # 模板原文（已从 JSON 解包）
    variables: list[str]    # 模板声明的变量名（创建/追加版本时提取）
    created_at: str


# ---------- 异常 ----------

class PromptNotFoundError(Exception):
    """prompt 或指定版本不存在。"""


class PromptAlreadyExistsError(Exception):
    """prompt id 已存在。"""


# ---------- 抽象接口 ----------

class PromptStore(Protocol):
    """Prompt 存储协议：上层只依赖此接口，换介质时实现新类即可。"""

    def create_prompt(self, prompt_id: str, name: str, description: str,
                      content: str, variables: list[str]) -> int:
        """创建 prompt 并写入版本 1，返回版本号。id 已存在时抛 PromptAlreadyExistsError。"""
        ...

    def get_prompt(self, prompt_id: str) -> PromptMeta | None:
        """查询单个 prompt 元信息（含 latest_version），不存在返回 None。"""
        ...

    def list_prompts(self) -> list[PromptMeta]:
        """列出全部 prompt（按创建时间倒序）。"""
        ...

    def add_version(self, prompt_id: str, content: str, variables: list[str]) -> int:
        """追加新版本（版本号自增），返回新版本号。prompt 不存在时抛 PromptNotFoundError。"""
        ...

    def get_version(self, prompt_id: str, version: int) -> PromptVersion | None:
        """取指定版本，不存在返回 None。"""
        ...

    def get_latest_version(self, prompt_id: str) -> PromptVersion | None:
        """取最新版本，不存在返回 None。"""
        ...

    def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        """列出版本历史（新版本在前）。prompt 不存在返回空列表。"""
        ...


# ---------- SQLite 实现 ----------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_id  TEXT NOT NULL REFERENCES prompts(id),
    version    INTEGER NOT NULL,
    content    TEXT NOT NULL,   -- JSON: {"text": "..."}（当前单条模板 schema）
    variables  TEXT NOT NULL,   -- JSON 数组：模板声明的变量名
    created_at TEXT NOT NULL,
    PRIMARY KEY (prompt_id, version)
);
"""


def _pack_content(content: str) -> str:
    """模板原文 → 存储 JSON。未来扩展 messages 数组模板时改这里。"""
    return json.dumps({"text": content}, ensure_ascii=False)


def _unpack_content(raw: str) -> str:
    """存储 JSON → 模板原文。"""
    return json.loads(raw)["text"]


class SqlitePromptStore:
    """基于 sqlite3（标准库）的 PromptStore。db_path 由调用方传入。"""

    def __init__(self, db_path: str):
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")    # add_version 依赖外键拦截不存在的 prompt
        conn.execute("PRAGMA journal_mode = WAL")   # 读写并发友好
        return conn

    def create_prompt(self, prompt_id: str, name: str, description: str,
                      content: str, variables: list[str]) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO prompts (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                    (prompt_id, name, description, now),
                )
                conn.execute(
                    "INSERT INTO prompt_versions (prompt_id, version, content, variables, created_at)"
                    " VALUES (?, 1, ?, ?, ?)",
                    (prompt_id, _pack_content(content), json.dumps(variables), now),
                )
        except sqlite3.IntegrityError as e:
            raise PromptAlreadyExistsError(f"prompt 已存在: {prompt_id!r}") from e
        return 1

    def get_prompt(self, prompt_id: str) -> PromptMeta | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT p.id, p.name, p.description, p.created_at,"
                " COALESCE((SELECT MAX(v.version) FROM prompt_versions v WHERE v.prompt_id = p.id), 0)"
                " AS latest_version"
                " FROM prompts p WHERE p.id = ?",
                (prompt_id,),
            ).fetchone()
        if row is None:
            return None
        return PromptMeta(
            id=row["id"], name=row["name"], description=row["description"],
            latest_version=row["latest_version"], created_at=row["created_at"],
        )

    def list_prompts(self) -> list[PromptMeta]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT p.id, p.name, p.description, p.created_at,"
                " COALESCE((SELECT MAX(v.version) FROM prompt_versions v WHERE v.prompt_id = p.id), 0)"
                " AS latest_version"
                " FROM prompts p ORDER BY p.created_at DESC, p.id"
            ).fetchall()
        return [
            PromptMeta(
                id=r["id"], name=r["name"], description=r["description"],
                latest_version=r["latest_version"], created_at=r["created_at"],
            )
            for r in rows
        ]

    def add_version(self, prompt_id: str, content: str, variables: list[str]) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        # 版本号在单条 INSERT 内用子查询分配（原子性由 SQLite 写锁保证），
        # RETURNING 取回新版本号；外键拦截不存在的 prompt（_connect 已开 foreign_keys）
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "INSERT INTO prompt_versions (prompt_id, version, content, variables, created_at)"
                    " VALUES (?,"
                    " COALESCE((SELECT MAX(version) FROM prompt_versions WHERE prompt_id = ?), 0) + 1,"
                    " ?, ?, ?)"
                    " RETURNING version",
                    (prompt_id, prompt_id, _pack_content(content), json.dumps(variables), now),
                ).fetchone()
        except sqlite3.IntegrityError as e:
            raise PromptNotFoundError(f"prompt 不存在: {prompt_id!r}") from e
        return row[0]

    def get_version(self, prompt_id: str, version: int) -> PromptVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE prompt_id = ? AND version = ?",
                (prompt_id, version),
            ).fetchone()
        return self._to_version(row) if row else None

    def get_latest_version(self, prompt_id: str) -> PromptVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE prompt_id = ?"
                " ORDER BY version DESC LIMIT 1",
                (prompt_id,),
            ).fetchone()
        return self._to_version(row) if row else None

    def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_versions WHERE prompt_id = ? ORDER BY version DESC",
                (prompt_id,),
            ).fetchall()
        return [self._to_version(r) for r in rows]

    @staticmethod
    def _to_version(row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            prompt_id=row["prompt_id"],
            version=row["version"],
            content=_unpack_content(row["content"]),
            variables=json.loads(row["variables"]),
            created_at=row["created_at"],
        )
