# AI Gateway

一个生产级模式的 LLM 接入网关：在下游应用与上游 LLM 供应商之间提供**统一 API**，屏蔽多供应商、多协议差异，并内置流式、结构化输出、Prompt 资产、可观测性、重试、限流六大治理能力。

当前上游默认指向 DeepSeek，通过两套协议接入：

| 模型 | 上游协议 | 说明 |
|---|---|---|
| `deepseek-v4-flash` | Anthropic Messages API | `/v1/messages` |
| `deepseek-v4-pro` | OpenAI Responses API | `/v1/responses` |

详细设计思考见 [doc.md](doc.md)（分层架构、请求生命周期、流式治理、协议适配矩阵）。

## 核心功能

1. **流式输出（SSE）**：统一极简事件流 `start / delta / done / error`，delta 分 `text` 与 `reasoning` 双通道（深度思考）；网关统一打点 `ttft_ms` / `elapsed_ms`，响应头携带 `Cache-Control: no-cache`、`X-Accel-Buffering: no` 防代理缓冲。
2. **结构化输出（JSON Schema）**：非流式校验失败返回 422 + 错误明细；流式同开时 JSON 增量实时透传，校验结论随 `done.validation` 下发。
3. **Prompt 模板引用**：`{{变量}}` 替换、`{% if %}` 条件、`{% for %}` 循环；版本管理（历史版本不可变）、渲染预览、chat 请求内联引用（`prompt + variables`，支持钉死版本）。
4. **可观测性**：`/v1/metrics` 指标快照（请求/错误/限流/重试计数、延迟分位数）、结构化 JSON 日志（全链路 `request_id`）、HTML 可视化报告。
5. **重试**：仅对可重试错误（network / server / 429 / 409）指数退避，尊重 `Retry-After`；流式仅在首事件发出前重试；不可重试错误快速失败。
6. **限流**：per 客户端 IP 滑动窗口，超限返回 429 + `Retry-After`，默认禁用。

## 项目结构

```
ai-gateway/
├── server.py            # FastAPI 接入层：/v1/chat、/v1/prompts、/v1/metrics、Web UI、SSE 出口
├── main.py              # 演示脚本：统一事件流消费 + 模板 v1/v2 渲染对比
├── gateway/             # 内核（协议无关领域模型 + 适配器 + 治理策略）
│   ├── types.py         #   统一模型：ChatRequest / ChatResponse / StreamEvent / Usage
│   ├── gateway.py       #   调度内核：模型路由表 MODEL_ROUTES、重试编排、事件流
│   ├── anthropic_adapter.py  # Anthropic Messages 协议适配
│   ├── responses_adapter.py  # OpenAI Responses 协议适配
│   ├── errors.py        #   GatewayError 统一错误（分类 / retryable / 状态码映射）
│   ├── retry.py         #   指数退避重试策略
│   ├── ratelimit.py     #   滑动窗口限流
│   ├── metrics.py       #   指标收集
│   ├── prompt_store.py  #   Prompt 模板存储（SQLite，版本不可变）
│   ├── prompt_render.py #   模板渲染（变量 / 条件 / 循环）
│   ├── structured_output.py  # JSON Schema 结构化输出
│   ├── sse.py           #   SSE 编解码
│   ├── logger_setup.py  #   JSON 日志 + request_id 贯穿
│   └── env.py           #   极简 .env 加载（已 export 的环境变量优先）
├── static/              # Web UI（原生 HTML/CSS/JS）
├── tests/               # pytest 单测与 API 集成测试
└── scripts/verification/  # 六大功能端到端验证 harness（mock 上游，无需真实 key）
    ├── run_all.py       #   总控：跑全部模块并汇总
    ├── verify_*.py      #   六个模块各自独立可跑
    ├── harness.py       #   进程编排 / SSE 解析 / 证据记录
    ├── mock_upstream.py #   可故障注入的 mock 上游
    ├── evidence/        #   已入库的六项验证证明（summary + 各模块 JSON）
    └── reports/         #   运行时产物（gitignore，含日志 / DB / HTML 报告）
```

## 快速开始

依赖：Python ≥ 3.13。包管理推荐 [uv](https://docs.astral.sh/uv/)；没有 uv 时可用标准 venv + pip 替代（见下）。

### 方式一：uv（推荐）

```bash
# 安装 uv（二选一）
curl -LsSf https://astral.sh/uv/install.sh | sh          # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 1. 安装依赖
uv sync
```

### 方式二：venv + pip（无 uv）

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install fastapi httpx jinja2 uvicorn   # 运行时依赖（pyproject.toml 同款）
pip install pytest               # 仅跑测试时需要
```

### 配置密钥并启动

```bash
# 2. 配置密钥（.env 或直接 export，已存在的环境变量优先）
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
EOF

# 3. 启动网关
uv run uvicorn server:app --reload --port 8000     # uv
uvicorn server:app --reload --port 8000            # venv + pip
```

- Web UI：<http://localhost:8000/>（选择模型、流式/深度思考/结构化输出开关、Prompt 模板管理、对比视图）
- 健康检查：<http://localhost:8000/health>

另附纯后端演示脚本（统一事件流 + 模板版本对比）：

```bash
uv run python main.py    # uv
python main.py           # venv + pip
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 必填 | Anthropic Messages 协议（flash 模型）鉴权 |
| `OPENAI_API_KEY` | 必填 | OpenAI Responses 协议（pro 模型）鉴权 |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_PATH` | `https://api.deepseek.com/anthropic` / `/v1/messages` | 上游地址与路径，支持中转站 |
| `OPENAI_BASE_URL` / `RESPONSES_API_PATH` | `https://api.deepseek.com` / `/v1/responses` | 同上 |
| `RETRY_MAX_ATTEMPTS` | `3` | 总尝试次数上限（含首次），1 = 禁用重试 |
| `RETRY_BACKOFF_BASE_S` / `RETRY_BACKOFF_MAX_S` | `0.5` / `8.0` | 指数退避基数与单次上限（秒） |
| `RATE_LIMIT_RPM` / `RATE_LIMIT_WINDOW_S` | `0`（禁用）/ `60` | 每窗口请求数与窗口长度（秒） |
| `PROMPTS_DB_PATH` | `prompts.db` | Prompt 模板 SQLite 路径 |
| `LOG_LEVEL` / `LOG_FILE` / `LOG_ENV` | `INFO` / stdout / `dev` | 日志级别 / JSON 日志落盘路径 / 环境标识 |
| `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` | — | 日志文件轮转配置 |

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | 健康检查 |
| `GET` | `/v1/models` | 可用模型列表 |
| `POST` | `/v1/chat` | 统一对话（`stream` / `thinking` / `response_format` / `prompt` 引用） |
| `POST` | `/v1/prompts` | 创建模板（自动提取变量） |
| `GET` | `/v1/prompts` | 模板列表 |
| `GET` | `/v1/prompts/{id}` | 模板详情（含版本） |
| `POST` | `/v1/prompts/{id}/versions` | 追加新版本 |
| `GET` | `/v1/prompts/{id}/versions/{version}` | 读取指定版本（`latest` 取最新） |
| `POST` | `/v1/prompts/{id}/render` | 渲染预览 |
| `DELETE` | `/v1/prompts/{id}` | 删除模板 |
| `GET` | `/v1/metrics` | 指标快照（JSON） |

非流式调用示例：

```bash
curl http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}'
```

模板引用调用示例：

```bash
curl http://localhost:8000/v1/prompts -H "Content-Type: application/json" \
  -d '{"id":"translator","name":"翻译","content":"把{{text}}从中文翻译成{{lang}}"}'

curl http://localhost:8000/v1/chat -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash",
       "prompt":{"id":"translator","variables":{"text":"你好","lang":"英文"}},
       "messages":[{"role":"user","content":"请开始"}]}'
```

流式响应为 SSE，下游只需实现一个解析器处理 4 种事件：

```
event: start
data: {"type":"start"}

event: delta
data: {"type":"delta","text":"你好","channel":"text"}

event: done
data: {"type":"done","usage":{"input_tokens":15,"output_tokens":5},"stop_reason":"stop","ttft_ms":320.5,"elapsed_ms":1850.2}

event: error
data: {"type":"error","error":"消息","elapsed_ms":120.3}
```

## 测试与验证

```bash
# 单元 / 集成测试
uv run pytest                # venv + pip: python -m pytest

# 六大功能端到端验证（mock 上游 + mock key，无需真实密钥）
uv run python scripts/verification/run_all.py            # 全部模块
uv run python scripts/verification/run_all.py streaming  # 指定模块
uv run python scripts/verification/verify_retry.py       # 单脚本独立运行
# venv + pip 环境下将 uv run python 换成 python 即可
```

验证脚本覆盖：流式输出、结构化输出、模板引用、可观测数据、重试机制、限流六大模块，每个脚本独立拉起 mock 上游 + 网关，自动判定 PASS/FAIL 并产出证据报告；任一模块非 PASS 时退出码为 1，可直接接入 CI。最新一轮全量通过的证明已存档于 [scripts/verification/evidence/](scripts/verification/evidence/)。
