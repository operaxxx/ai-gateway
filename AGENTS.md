# 仓库协作指南

默认使用中文沟通、编写说明与注释；代码标识符沿用现有英文命名。

## 开发入口

这是 Python 3.13+ 的 FastAPI LLM 网关，使用 uv 管理依赖。先读 `README.md` 了解启动与 API；涉及分层、请求生命周期或协议设计时读 `doc.md`，并以实际实现核对文档。

常用命令（在仓库根目录执行）：

```bash
uv sync
uv run uvicorn server:app --reload --port 8000
uv run pytest
uv run pytest tests/test_retry.py -q
uv run python scripts/verification/run_all.py
uv run python scripts/verification/run_all.py streaming retry
```

依赖和 pytest 配置以 `pyproject.toml` 为准；变更依赖时同步更新 `uv.lock`。当前未配置统一 lint 或类型检查命令，沿用所改文件的风格，避免无关格式化。

## 修改边界

- `server.py` 负责 HTTP 入参、Prompt 引用、限流、错误响应和 SSE 序列化；`gateway/gateway.py` 负责模型路由、重试编排、计时和结构化校验。
- `gateway/types.py` 定义协议无关的数据模型。上游载荷与事件差异在 `anthropic_adapter.py`、`responses_adapter.py` 内转换；新增模型时检查 `MODEL_ROUTES` 及两个调用路径（非流式和流式）。
- 保持统一 SSE 事件 `start / delta / done / error`，区分 `text` 和 `reasoning` 通道。计时由 Gateway 统一负责；首事件发出后不再重试，以免重复输出。
- 结构化输出的非流式校验失败返回 422；流式校验结论随 `done.validation` 返回。修改时同时核对内部字段和 HTTP 序列化。
- Prompt 历史版本不可变，修改内容通过追加版本实现；存储和渲染分别在 `prompt_store.py`、`prompt_render.py` 中维护。
- 限流有两层：客户端 IP 滑动窗口和模型级令牌桶。涉及限流时读取 `gateway/ratelimit.py` 与 `.claude/artifacts/designs/model-rate-limit.md`，同时检查拒绝响应、`Retry-After` 和指标。

## 前端工作

修改 `static/` 前先读 `.design-context.md`；涉及布局或视觉设计时再读 `.claude/artifacts/designs/ui-redesign.md`。保持原生 HTML/CSS/JS、无构建流程，遵循现有样式 token、DOM ID 绑定和用户数据安全插入约定。

## 验证与运行副作用

- 按改动运行相关 pytest 测试；影响共享请求模型、路由或网关编排时运行完整 pytest。涉及流式、重试、限流等链路行为时，补跑 `scripts/verification/run_all.py` 对应模块；该验证使用 mock 上游和 mock key。
- 测试沿用 `tmp_path`、`monkeypatch`、FakeGateway 或 mock HTTP 隔离数据、时间与网络。`tests/test_streaming.py` 是真实上游的手动演示脚本，不等同于离线流式回归测试；执行真实调用前确认任务需要和凭证配置。
- `server.py` 导入时会初始化全局对象、打开 Prompt 数据库并播种示例模板。需要隔离数据库时，在导入前设置 `PROMPTS_DB_PATH`；限流等环境配置也需在初始化前设置。
- `.env`、本地 SQLite 数据库和验证运行报告保持本地；报告写入 `scripts/verification/reports/`，`evidence/` 为已提交的历史证据，仅在明确需要更新证据时修改。
- 修改 API、环境配置或启动方式时同步维护 `README.md`。交付时说明实际执行的验证及结果，区分本次验证和历史证据。
