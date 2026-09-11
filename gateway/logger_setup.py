"""日志体系入口：JsonFormatter + setup_logging + request_id 注入。

设计要点：
- 零新依赖（只用 stdlib logging），后续切 structlog / python-json-logger 无侵入
- JSON 格式：每行一个 JSON 对象，天然兼容 Loki / ELK / OTel Collector
- request_id 用 contextvar 传：middleware 生成 → LoggerFilter 注入 LogRecord → JsonFormatter 输出
- 环境变量配置：LOG_LEVEL / LOG_FILE / LOG_MAX_BYTES / LOG_BACKUP_COUNT
- 生产默认：stdout + 文件（RotatingFileHandler，10MB × 5）；开发仅 stdout

字段 Schema（与 OTel 语义约定对齐）：
  service, env, timestamp, level, logger, file, line, request_id,
  model, provider, stream, input_tokens, output_tokens, elapsed_ms, ttft_ms, stop_reason,
  error_category, error_type, error_status, error_message,
  msg
"""

import json
import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# ---------- request_id 传递 ----------
# 用 contextvar 跨中间件→handler→logger 传递 request_id，
# Python 3.7+ 内置，asyncio / 多线程都能正确隔离
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """生成并设置一个新的 request_id（UUID4 去横线）。"""
    rid = uuid.uuid4().hex
    request_id_ctx.set(rid)
    return rid


def get_request_id() -> str:
    return request_id_ctx.get()


# ---------- JsonFormatter ----------

class JsonFormatter(logging.Formatter):
    """把 LogRecord 格式化成一行 JSON（机器可读）。

    注入的 extra 字段会自动出现在 JSON 顶层；record 上不存在的字段用 '-' 兜底。
    异常时带 traceback（tb 字段），生产排查必备。
    """

    _RESOURCE_FIELDS = ("service", "env")

    def __init__(self, service: str = "ai-gateway", env: str = "dev"):
        super().__init__()
        self._service = service
        self._env = env

    def format(self, record: logging.LogRecord) -> str:
        # —— 时间戳（UTC ISO 8601，毫秒精度）
        ts = self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.") + f"{int(record.msecs):03d}Z"

        # —— 核心字段
        entry: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "service": self._service,
            "env": self._env,
            "logger": record.name,
            "file": record.filename,
            "line": record.lineno,
            "request_id": get_request_id(),
            "msg": record.getMessage(),
        }

        # —— 异常栈（record.exc_info 在 handler 里已 set）
        if record.exc_info and record.exc_info[0] is not None:
            entry["tb"] = self.formatException(record.exc_info)

        # —— 把所有非标准字段（来自 logger.info("...", extra={...}) 或 record.extra）注入
        _inject_extra(entry, record)

        return json.dumps(entry, ensure_ascii=False, default=str)


def _inject_extra(entry: dict, record: logging.LogRecord) -> None:
    """把 record 上的自定义属性（不在 LogRecord 标准字段里的）合并进 entry。

    跳过 None 值——避免 middleware 里没传 model/provider 时每行日志
    都带上一堆 "model": null "provider": null 这种噪声字段。
    """
    std_attrs = frozenset({
        "name", "msg", "args", "created", "relativeCreated", "msecs", "thread",
        "threadName", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "process",
        "processName", "taskName",
    })
    for key, value in record.__dict__.items():
        if key in std_attrs:
            continue
        if key.startswith("_"):
            continue
        if value is None:
            continue
        entry[key] = value


# ---------- LogRecord 清洗 Filter ----------

class RecordFilter(logging.Filter):
    """确保缺失的 LLM 领域字段有兜底（通过 _inject_extra 的 None 跳过机制
    实际上已经不需要了，这里保留做未来扩展——比如对特定字段做强类型校验）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return True


# ---------- setup_logging ----------

def setup_logging(
    *,
    service: str = "ai-gateway",
    env: str | None = None,
    level: str | None = None,
    log_file: str | None = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> None:
    """一次性配置 root logger。在入口模块（server.py 顶层）调一次即可。

    环境变量覆盖（方便生产部署不重新打包）：
      LOG_LEVEL        默认 INFO
      LOG_FILE         不设则仅 stdout
      LOG_MAX_BYTES    默认 10485760
      LOG_BACKUP_COUNT 默认 5
      LOG_ENV          默认 dev
    """
    level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    env = env or os.environ.get("LOG_ENV", "dev")
    log_file = log_file or os.environ.get("LOG_FILE")
    max_bytes = int(os.environ.get("LOG_MAX_BYTES", max_bytes))
    backup_count = int(os.environ.get("LOG_BACKUP_COUNT", backup_count))

    # —— 避免重复配置（uvicorn reload 时 import 链会重跑）
    root = logging.getLogger()
    if getattr(root, "_ai_gateway_configured", False):
        return
    root._ai_gateway_configured = True  # type: ignore[attr-defined]

    root.setLevel(getattr(logging, level, logging.INFO))

    formatter = JsonFormatter(service=service, env=env)
    flt = RecordFilter()

    # —— stdout handler（所有部署形态都需要，Docker/OTel Collector 从这里抓）
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(formatter)
    stdout.addFilter(flt)
    root.addHandler(stdout)

    # —— 可选文件 handler（本地备份；生产用 OTel Collector 从 stdout 抓即可不依赖）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(flt)
        root.addHandler(file_handler)

    # —— 第三方库降噪（uvicorn.access / httpx INFO 太吵）
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
